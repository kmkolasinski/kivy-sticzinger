from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    name="py_cl_stub",
    include_dirs=["../cl_stub/include"],
    ext_modules=cythonize(
        [
            Extension(
                "py_cl_stub",
                ["py_cl_stub.pyx"],
                libraries=["cl_stub"],
                language="c++",
                extra_compile_args=["-Wl,--no-as-needed", "-ldl"],
                extra_link_args=["-L../cl_stub/build/"]
            )
        ],
        # annotate=False,
    ),
)

# from pathlib import Path
# from os import getenv
# import sys
#
#
# from setuptools import setup, Extension
# from setuptools.command.build_ext import build_ext
# import numpy
#
# VERSION = "1.1"
# FILES = list(Path(".").rglob("*.pyx")) + list(Path(".").rglob("*.pxi"))
# INSTALL_REQUIRES = []
# SETUP_REQUIRES = []
# PLATFORM = sys.platform
#
# if getenv("NDKPLATFORM") and getenv("LIBLINK"):
#     PLATFORM = "android"
#
# # detect cython
# if PLATFORM != "android":
#     SETUP_REQUIRES.append("cython")
#     INSTALL_REQUIRES.append("cython")
# else:
#     FILES = [fn.with_suffix(".cc") for fn in FILES]
#
# setup(
#     name="cl_stubs",
#     version=VERSION,
#     cmdclass={"build_ext": build_ext},
#     install_requires=INSTALL_REQUIRES,
#     setup_requires=SETUP_REQUIRES,
#     include_dirs=[numpy.get_include()],
#     ext_modules=[Extension(
#         "cl_stubs", [str(fn) for fn in FILES]),
#     ],
# )
