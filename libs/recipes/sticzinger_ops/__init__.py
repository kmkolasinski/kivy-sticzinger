# https://github.com/tshirtman/android_cython_example
from pythonforandroid.recipe import IncludedFilesBehaviour, CythonRecipe


class SticzingerOpsRecipe(IncludedFilesBehaviour, CythonRecipe):
    src_filename = '../../../libs/sticzinger_ops'
    url = None
    version = '1.0'
    name = 'sticzinger_ops'
    site_package_name = 'sticzinger_ops'
    depends = ['setuptools']
    call_hostpython_via_targetpython = False
    install_in_hostpython = True


recipe = SticzingerOpsRecipe()