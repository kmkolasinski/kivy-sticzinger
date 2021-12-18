from pathlib import Path
from os import getenv
import sys


from setuptools import setup, Extension
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext
import numpy

VERSION = "1.1"
FILES = list(Path(".").rglob("*.pyx")) + list(Path(".").rglob("*.pxi"))
INSTALL_REQUIRES = []
SETUP_REQUIRES = []
PLATFORM = sys.platform

if getenv("NDKPLATFORM") and getenv("LIBLINK"):
    PLATFORM = "android"

# detect cython
if PLATFORM != "android":
    SETUP_REQUIRES.append("cython")
    INSTALL_REQUIRES.append("cython")
else:
    FILES = [fn.with_suffix(".c") for fn in FILES]

# create the extension
setup(
    name="sticzinger_ops",
    version=VERSION,
    cmdclass={"build_ext": build_ext},
    install_requires=INSTALL_REQUIRES,
    setup_requires=SETUP_REQUIRES,
    include_dirs=[numpy.get_include()],
    ext_modules=cythonize(
        [
            Extension(
                "sticzinger_ops",
                ["sticzinger_ops.pyx", "fast_ops.c"],
                libraries=["blas"],
                extra_compile_args=[
                    "-Ofast",
                    "-mavx",
                    "-ftree-vectorize",
                    "-march=native",
                    "-finline-functions",
                    "-fopt-info-vec-optimized",
                ],
            )
        ],
        # annotate=True,
    ),
    extras_require={"dev": [], "ci": []},
)
