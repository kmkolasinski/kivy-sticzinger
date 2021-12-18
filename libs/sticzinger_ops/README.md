# Compilation

pip install -e .
python setup.py build_ext --inplace
python setup_cythonize.py build_ext --inplace