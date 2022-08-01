# gammapy-ogip-spectra

This folder contains ... 

## Installation and set-up

These instructions assume that you have previously installed a version of [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) on your machine.

### Case 1: you do not need the XSpec models library 


### Case2: you do need the XSpec models library

#### Create the working environment 
from https://threeml.readthedocs.io/en/v2.1.1/installation.html
```
conda create --name GammapyX -c conda-forge python=3.7 numpy scipy matplotlib
conda activate GammapyX
conda install -c conda-forge -c threeml astromodels threeml
```
#### Gammapy installation
```conda install gammapy```
