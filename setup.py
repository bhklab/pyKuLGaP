from setuptools import setup, find_packages

setup(name='kulgap',
      version='0.0.1',
      description="Functions for statistical analysis of treatment response curves in patient derived xenograph (PDX) models of cancer.",
      url='https://github.com/bhklab/pyKuLGaP/tree/pypi',
      author='Janosch Ortmann, Christopher Eeles, Benjamin Haibe-Kains',
      author_email='janosch.ortmann@gmail.com, christopher.eeles@uhnresearch.ca, benjamin.haibe.kains@utoronto.ca',
      license='MIT',
      packages=['kulgap'],
      package_dir={'kulgap': 'kulgap'},
      package_data={'kulgap': 'data/*'},
      zip_safe=False
      )