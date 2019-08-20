from setuptools import setup, Extension
# from Cython.Build import cythonize

# CYTHON COMPILATION
# compile with $> python setup.py build_ext --inplace
# change file extensions to .pyx to add type annotations
# ext_modules = cythonize([
#     Extension("qradient.circuit_logic", ["qradient/circuit_logic.py"]),
#     Extension("qradient.physical_components", ["qradient/physical_components.py"])
# ])

setup(
    # ext_modules=ext_modules,
    name='qradient',
    version='1.0.0',
    description='A package for efficient simulation and differentiation of parametrized quantum circuits.',
    author='Frederik Wilde',
    author_email='wilde.pysics@gmail.com',
    url='https://github.com/frederikwilde/qradient',
    # license='MIT',
    packages=['qradient'],
    install_requires=[
        'scipy',
        'tqdm',
        'Cython'
    ],
    # zip_safe=False
)
