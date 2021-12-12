from os.path import join
import hashlib
from re import match

import sh
import shutil
import fnmatch
from pythonforandroid.recipe import CppCompiledComponentsPythonRecipe, Recipe
from pythonforandroid.logger import (logger, info, warning, debug, shprint, info_main)
from pythonforandroid.util import (current_directory, ensure_dir,
                                   BuildInterruptingException)
from pythonforandroid.util import load_source as import_recipe

class MRTPRecipe(CppCompiledComponentsPythonRecipe):
    version = '1.1.1'
    url = 'https://github.com/kmkolasinski/mrpt/archive/refs/tags/release-1.1.1.tar.gz'
    depends = ['ipdb', 'numpy']

    def get_recipe_env(self, arch):
        env = super().get_recipe_env(arch)
        # we need the includes from our installed numpy at site packages
        # because we need some includes generated at numpy's compile time
        env['NUMPY_INCLUDES'] = join(
            self.ctx.get_python_install_dir(), "numpy/core/include",
        )

        # this flag below is to fix a runtime error:
        #   ImportError: dlopen failed: cannot locate symbol
        #   "_ZTVSt12length_error" referenced by
        #   "/data/data/org.test.matplotlib_testapp/files/app/_python_bundle
        #   /site-packages/pandas/_libs/window/aggregations.so"...
        env['LDFLAGS'] += f' -landroid  -l{self.stl_lib_name}'




        return env

    def should_build(self, arch):
        name = self.folder_name
        # if self.ctx.has_package(name):
        #     info('Python package already exists in site-packages')
        #     return False

        return True

    def build_arch(self, arch):
        '''Build any cython components, then install the Python module by
        calling setup.py install with the target Python dir.
        '''
        info('> BUILDING !!!')
        import ipdb
        ipdb.set_trace()
        super(MRTPRecipe, self).build_arch(arch)


recipe = MRTPRecipe()