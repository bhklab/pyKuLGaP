from setuptools import setup, find_packages

with open("README.md") as fh:
    readme = fh.read()

setup(name='pykulgap',
      version='0.0.13',
      description="Functions for statistical analysis of treatment response curves in patient derived xenograph (PDX) models of cancer.",
      long_description=readme,
      long_description_content_type='text/markdown',
      url='https://github.com/bhklab/pyKuLGaP/',
      author='Janosch Ortmann, Christopher Eeles, Anna Goldenberg, Benjamin Haibe-Kains',
      author_email='janosch.ortmann@gmail.com, christopher.eeles@uhnresearch.ca, NA, benjamin.haibe.kains@utoronto.ca',
      license='MIT',
      packages=find_packages(),
      zip_safe=False
      )
