from pathlib import Path
from os import getenv
import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

VERSION = '1.0'

FILES = list(Path('.').rglob('*.pyx')) + list(Path('.').rglob('*.pxi'))

INSTALL_REQUIRES = []
SETUP_REQUIRES = []

# detect Python for android
PLATFORM = sys.platform

if getenv('NDKPLATFORM') and getenv('LIBLINK'):
    PLATFORM = 'android'

# detect cython
if PLATFORM != 'android':
    SETUP_REQUIRES.append('cython')
    INSTALL_REQUIRES.append('cython')
else:
    FILES = [fn.with_suffix('.c') for fn in FILES]

# create the extension
setup(
    name='sticzinger_ops',
    version=VERSION,
    cmdclass={'build_ext': build_ext},
    install_requires=INSTALL_REQUIRES,
    setup_requires=SETUP_REQUIRES,
    ext_modules=[
        Extension(
            'sticzinger_ops',
            [str(fn) for fn in FILES],
        )
    ],
    extras_require={
        'dev': [],
        'ci': [],
    },
)