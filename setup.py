from setuptools import find_packages, setup

setup(
    name='tabularepimdl',
    packages=find_packages(include=['tabularepimdl']),
    version='0.1.0',
    description='A tabular approach to epidemic simulation',
    author='Justin Lessler',
    license='MIT',
    url='https://github.com/UNCIDD/tabularepimdl/tree/tabularepimdl_unittest',
    install_requires=[
        "numpy",  
        "pandas",
        "PyYAML",
        "plotly"
    ],
    python_requires=">=3.6"
)