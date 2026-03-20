from setuptools import setup

NAME = "UMPA"
VERSION = "0.2"
DESCR = "Unified Modulated Pattern Analysis"
REQUIRES = ['numpy', 'juliacall>=0.9.31']

AUTHOR = "Pierre Thibault, Fabio De Marco, Sara Savatovic, Ronan Smith"
EMAIL = "pthibault@units.it"

LICENSE = "GPL 3.0"

SRC_DIR = "UMPA"
PACKAGES = [SRC_DIR]

if __name__ == "__main__":
    setup(install_requires=REQUIRES,
          packages=PACKAGES,
          zip_safe=False,
          name=NAME,
          version=VERSION,
          description=DESCR,
          author=AUTHOR,
          author_email=EMAIL,
          #url=URL,
          license=LICENSE,
          include_package_data=True,
          package_data={SRC_DIR: ['julia/*.jl', 'test/logo.npy']}
          )
