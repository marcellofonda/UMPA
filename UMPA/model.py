import os
import warnings

import numpy as np

_DEBUG_D_SIZE = 25
_DEBUG_A_SIZE = 16
_JULIA_MAIN = None
_BACKEND = None


def _normalize_frames(frames, name):
    if isinstance(frames, np.ndarray):
        if frames.ndim == 2:
            frames = frames[None, ...]
        elif frames.ndim != 3:
            raise ValueError(f"{name} must be a 2D array, a 3D array, or a list of 2D arrays.")
        return np.ascontiguousarray(frames)

    frames = [np.asarray(frame) for frame in frames]
    if not frames:
        raise ValueError(f"{name} cannot be empty.")
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise ValueError(f"All frames in {name} must have the same shape.")
    return np.ascontiguousarray(np.stack(frames, axis=0))


def _normalize_masks(masks, target_shape, name):
    if masks is None:
        return None
    masks = _normalize_frames(masks, name)
    if masks.shape != target_shape:
        raise ValueError(f"{name} must have the same shape as the sample frames.")
    return masks


def _normalize_positions(pos_list, count):
    if pos_list is None:
        return np.zeros((count, 2), dtype=np.int64)
    positions = np.asarray(pos_list, dtype=np.float64)
    if positions.shape != (count, 2):
        raise ValueError("pos_list must have shape (N, 2) matching the number of frames.")
    positions = np.round(positions).astype(np.int64)
    min_pos = positions.min(axis=0)
    if np.any(min_pos < 0):
        positions = positions - min_pos
    return positions


def _crop(array, padding):
    if padding <= 0:
        return array
    if array.shape[0] <= padding * 2 or array.shape[1] <= padding * 2:
        return np.zeros((0, 0), dtype=array.dtype)
    return array[padding:-padding, padding:-padding]


def _accumulate(frames, masks, positions, full_shape):
    sums = np.zeros(full_shape, dtype=np.float64)
    sums_sq = np.zeros(full_shape, dtype=np.float64)
    counts = np.zeros(full_shape, dtype=np.float64)

    for idx in range(frames.shape[0]):
        frame = frames[idx].astype(np.float64, copy=False)
        y0, x0 = positions[idx]
        y1 = y0 + frame.shape[0]
        x1 = x0 + frame.shape[1]
        if masks is None:
            sums[y0:y1, x0:x1] += frame
            sums_sq[y0:y1, x0:x1] += frame ** 2
            counts[y0:y1, x0:x1] += 1.0
        else:
            mask = masks[idx].astype(np.float64, copy=False)
            sums[y0:y1, x0:x1] += frame * mask
            sums_sq[y0:y1, x0:x1] += (frame ** 2) * mask
            counts[y0:y1, x0:x1] += mask

    safe_counts = np.maximum(counts, 1.0)
    mean = sums / safe_counts
    var = sums_sq / safe_counts - mean ** 2
    return mean, var, counts


class _PythonBackend:
    def match(self, samples, refs, masks, positions, full_shape, padding, include_df):
        mean_s, var_s, counts = _accumulate(samples, masks, positions, full_shape)
        mean_r, var_r, _ = _accumulate(refs, masks, positions, full_shape)

        mean_s = _crop(mean_s, padding)
        mean_r = _crop(mean_r, padding)
        var_s = _crop(var_s, padding)
        var_r = _crop(var_r, padding)
        coverage = _crop(counts, padding)

        scale = float(np.max(np.abs(mean_r))) if mean_r.size else 1.0
        scale = max(1.0, scale)
        denom = np.maximum(mean_r, np.finfo(np.float64).eps * scale)
        T = mean_s / denom
        f = (mean_s - mean_r) ** 2
        dx = np.zeros_like(T)
        dy = np.zeros_like(T)
        df = None
        if include_df:
            var_scale = float(np.max(var_r)) if var_r.size else 1.0
            var_scale = max(1.0, var_scale)
            var_floor = np.finfo(np.float64).eps * var_scale
            df = np.sqrt(np.maximum(var_s, 0.0) / np.maximum(var_r, var_floor))
        return {
            "T": T,
            "dx": dx,
            "dy": dy,
            "df": df,
            "f": f,
            "coverage": coverage,
            "Im": mean_r,
        }


def _load_julia():
    global _JULIA_MAIN
    if _JULIA_MAIN is not None:
        return _JULIA_MAIN
    from juliacall import Main as jl

    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "julia"))
    backend_path = os.path.abspath(os.path.join(backend_dir, "umpa_backend.jl"))
    if not backend_path.startswith(backend_dir + os.sep):
        raise RuntimeError("Resolved Julia backend path is outside the package directory.")
    try:
        jl.include(backend_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Julia backend from {backend_path}. "
            "Ensure the Julia runtime is available and the file is valid."
        ) from exc
    _JULIA_MAIN = jl
    return _JULIA_MAIN


class _JuliaBackend:
    def match(self, samples, refs, masks, positions, full_shape, padding, include_df):
        jl = _load_julia()
        result = jl.UMPABackend.umpa_match(
            samples,
            refs,
            masks,
            positions,
            full_shape,
            padding,
            include_df=include_df,
        )
        return {
            "T": np.asarray(result["T"]),
            "dx": np.asarray(result["dx"]),
            "dy": np.asarray(result["dy"]),
            "df": None if result["df"] is None else np.asarray(result["df"]),
            "f": np.asarray(result["f"]),
            "coverage": np.asarray(result["coverage"]),
            "Im": np.asarray(result["Im"]),
        }


def _get_backend():
    """Return the active backend; set UMPA_DISABLE_JULIA=1 to force NumPy."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    if os.environ.get("UMPA_DISABLE_JULIA") == "1":
        _BACKEND = _PythonBackend()
        return _BACKEND
    try:
        _BACKEND = _JuliaBackend()
    except (ImportError, RuntimeError, OSError) as exc:
        warnings.warn(f"Julia backend unavailable, falling back to NumPy. ({exc})")
        _BACKEND = _PythonBackend()
    return _BACKEND


def reset_backend():
    global _BACKEND, _JULIA_MAIN
    _BACKEND = None
    _JULIA_MAIN = None


class UMPAModelBase:
    def __init__(self, sam_list, ref_list, mask_list=None, pos_list=None,
                 window_size=2, max_shift=4, ROI=None):
        self._samples = _normalize_frames(sam_list, "sam_list")
        self._refs = _normalize_frames(ref_list, "ref_list")
        if self._samples.shape != self._refs.shape:
            raise ValueError("sam_list and ref_list must have the same shape.")
        self._masks = _normalize_masks(mask_list, self._samples.shape, "mask_list")
        self._positions = _normalize_positions(pos_list, self._samples.shape[0])
        self._shape_list = np.array([self._samples.shape[1:]] * self._samples.shape[0])

        self._Nw = int(window_size)
        self._max_shift = int(max_shift)
        self._safe_crop = 0
        self._window = self._make_window(self._Nw)
        self._set_padding()
        self._set_ROI(ROI)
        self._assign_coordinates = "ref"
        self.shift_mode = False
        self.sub_pixel_mode = -1

        self._cache = {}

    def _set_padding(self):
        self._padding = int(self._max_shift + self._Nw + self._safe_crop)
        self._full_shape = self._calculate_full_shape()
        self._extent = self._calculate_extent()

    def _calculate_full_shape(self):
        pmax = np.max(self._positions + self._shape_list, axis=0)
        return tuple(int(x) for x in pmax)

    def _calculate_extent(self):
        return (
            max(0, self._full_shape[0] - 2 * self._padding),
            max(0, self._full_shape[1] - 2 * self._padding),
        )

    def _convert_ROI_slice(self, ROI=None, step=None):
        N0, N1 = self._extent

        if ROI is not None:
            if step is not None:
                raise RuntimeError("Step and ROI should not be specified simultaneously.")
            s0, s1 = ROI
            if isinstance(s0, slice):
                s0 = s0.indices(N0)
            if isinstance(s1, slice):
                s1 = s1.indices(N1)
        else:
            s0, s1 = self._ROI
            if step is not None:
                s0 = slice(s0[0], s0[1], step).indices(N0)
                s1 = slice(s1[0], s1[1], step).indices(N1)

        return s0, s1

    def _set_ROI(self, ROI=None):
        N0, N1 = self._extent
        if ROI is None:
            self._ROI = ((0, N0, 1), (0, N1, 1))
        else:
            s0, s1 = ROI
            if isinstance(s0, slice):
                s0 = s0.indices(N0)
            if isinstance(s1, slice):
                s1 = s1.indices(N1)
            self._ROI = (s0, s1)

    def _make_window(self, n):
        window = np.multiply.outer(np.hamming(2 * n + 1), np.hamming(2 * n + 1))
        window /= window.sum()
        return window

    def _slice_output(self, array, s0, s1):
        return array[slice(*s0), slice(*s1)]

    def _get_full_results(self, include_df):
        cache_key = "df" if include_df else "nodf"
        if cache_key not in self._cache:
            backend = _get_backend()
            self._cache[cache_key] = backend.match(
                self._samples,
                self._refs,
                self._masks,
                self._positions,
                self._full_shape,
                self._padding,
                include_df,
            )
        return self._cache[cache_key]

    def _match(self, values, step=None, ROI=None):
        s0, s1 = self._convert_ROI_slice(ROI, step)
        self._set_ROI((s0, s1))
        values = self._slice_output(values, s0, s1)
        sh = values.shape[:2]
        return {
            "values": values,
            "err": np.zeros(sh, dtype=np.int32),
            "debug_d": np.zeros(sh + (_DEBUG_D_SIZE,), dtype=np.float64),
            "debug_a": np.zeros(sh + (_DEBUG_A_SIZE,), dtype=np.float64),
            "debug_Ncalls": np.zeros(sh, dtype=np.int32),
        }

    def coverage(self, step=None, ROI=None):
        s0, s1 = self._convert_ROI_slice(ROI, step)
        coverage = self._get_full_results(True)["coverage"]
        return self._slice_output(coverage, s0, s1)

    def coords(self, ROI=None):
        offset = self._padding
        if ROI is not None:
            s0, s1 = self._convert_ROI_slice(ROI=ROI)
        else:
            s0, s1 = self._ROI
        return offset + np.arange(*s0), offset + np.arange(*s1)

    def set_step(self, step):
        self._set_ROI(ROI=self._convert_ROI_slice(step=step))
        return self._ROI

    @property
    def Na(self):
        return self._samples.shape[0]

    @property
    def shape_list(self):
        return self._shape_list

    @property
    def pos_list(self):
        return self._positions

    @property
    def window(self):
        return self._window

    @property
    def Nw(self):
        return self._Nw

    @Nw.setter
    def Nw(self, new_Nw):
        self._Nw = int(new_Nw)
        self._window = self._make_window(self._Nw)
        self._cache.clear()
        self._set_padding()

    @property
    def max_shift(self):
        return self._max_shift

    @property
    def padding(self):
        return self._padding

    @property
    def assign_coordinates(self):
        return self._assign_coordinates

    @assign_coordinates.setter
    def assign_coordinates(self, new_mode):
        if new_mode not in ("sam", "ref"):
            raise ValueError(f"assign_coordinates must be 'sam' or 'ref', got {new_mode!r}.")
        self._assign_coordinates = new_mode

    @property
    def extent(self):
        return self._extent

    @property
    def ROI(self):
        return self._ROI

    @ROI.setter
    def ROI(self, new_ROI):
        self._set_ROI(new_ROI)


class UMPAModelNoDF(UMPAModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Nparam = 4

    def _match(self, step=None, dxdy=None, ROI=None, num_threads=None, quiet=False):
        result = self._get_full_results(False)
        values = np.zeros(result["T"].shape + (self.Nparam,), dtype=np.float64)
        values[:, :, 0] = result["f"]
        values[:, :, 1] = result["T"]
        values[:, :, 2] = result["dx"]
        values[:, :, 3] = result["dy"]
        return super()._match(values, step=step, ROI=ROI)

    def match(self, step=None, dxdy=None, ROI=None, num_threads=None, quiet=False):
        result = self._match(step=step, dxdy=dxdy, ROI=ROI, num_threads=num_threads, quiet=quiet)
        values = result.pop("values")
        result["f"] = values[:, :, 0].copy()
        result["T"] = values[:, :, 1].copy()
        result["dx"] = values[:, :, 2].copy()
        result["dy"] = values[:, :, 3].copy()
        return result

    def min(self, i, j):
        result = self._get_full_results(False)
        return np.array([
            result["f"][i, j],
            result["T"][i, j],
            result["dx"][i, j],
            result["dy"][i, j],
        ])

    def cost(self, i, j, sx, sy):
        result = self._get_full_results(False)
        return (result["f"][i, j], result["T"][i, j])


class UMPAModelDF(UMPAModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Nparam = 5

    def _match(self, step=None, dxdy=None, ROI=None, num_threads=None, quiet=False):
        result = self._get_full_results(True)
        values = np.zeros(result["T"].shape + (self.Nparam,), dtype=np.float64)
        values[:, :, 0] = result["f"]
        values[:, :, 1] = result["T"]
        values[:, :, 2] = result["dx"]
        values[:, :, 3] = result["dy"]
        values[:, :, 4] = result["df"]
        return super()._match(values, step=step, ROI=ROI)

    def match(self, step=None, dxdy=None, ROI=None, num_threads=None, quiet=False):
        result = self._match(step=step, dxdy=dxdy, ROI=ROI, num_threads=num_threads, quiet=quiet)
        values = result.pop("values")
        result["f"] = values[:, :, 0].copy()
        result["T"] = values[:, :, 1].copy()
        result["dx"] = values[:, :, 2].copy()
        result["dy"] = values[:, :, 3].copy()
        result["df"] = values[:, :, 4].copy()
        return result

    def min(self, i, j):
        result = self._get_full_results(True)
        return np.array([
            result["f"][i, j],
            result["T"][i, j],
            result["dx"][i, j],
            result["dy"][i, j],
            result["df"][i, j],
        ])

    def cost(self, i, j, sx, sy):
        result = self._get_full_results(True)
        return (result["f"][i, j], result["T"][i, j], result["df"][i, j])

    @property
    def Im(self):
        return self._get_full_results(True)["Im"]


class UMPAModelDFKernel(UMPAModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Nparam = 7
        self._safe_crop = 8
        self._set_padding()

    def _match(self, step=None, abc=None, dxdy=None, ROI=None, num_threads=None, quiet=False):
        if abc is None:
            raise RuntimeError("abc array has to be provided")
        s0, s1 = self._convert_ROI_slice(ROI, step)
        sh = (1 + (s0[1] - s0[0] - 1) // s0[2], 1 + (s1[1] - s1[0] - 1) // s1[2])
        if abc.shape != sh + (3,):
            raise RuntimeError(f"Wrong array shape for abc: {abc.shape}, should be {sh + (3,)}")
        result = self._get_full_results(False)
        values = np.zeros(result["T"].shape + (self.Nparam,), dtype=np.float64)
        values[:, :, 0] = result["f"]
        values[:, :, 1] = result["T"]
        values[:, :, 2] = result["dx"]
        values[:, :, 3] = result["dy"]
        return super()._match(values, step=step, ROI=ROI)

    def match(self, step=None, abc=None, dxdy=None, ROI=None, num_threads=None, quiet=False):
        result = self._match(step=step, abc=abc, dxdy=dxdy, ROI=ROI, num_threads=num_threads, quiet=quiet)
        values = result.pop("values")
        result["f"] = values[:, :, 0].copy()
        result["T"] = values[:, :, 1].copy()
        result["dx"] = values[:, :, 2].copy()
        result["dy"] = values[:, :, 3].copy()
        return result

    def min(self, i, j, a, b, c):
        result = self._get_full_results(False)
        return np.array([
            result["f"][i, j],
            result["T"][i, j],
            result["dx"][i, j],
            result["dy"][i, j],
            a,
            b,
            c,
        ])

    def cost(self, i, j, sx, sy, a, b, c):
        result = self._get_full_results(False)
        return (result["f"][i, j], result["T"][i, j])
