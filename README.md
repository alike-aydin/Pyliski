

# Pyliski 

This package implements a Python version of the Iliski woftware, previously described in [Aydin et al. (2021)](https://doi.org/10.1371/journal.pcbi.1008614) and available at [https://github.com/alike-aydin/Iliski](https://github.com/alike-aydin/Iliski). Shortly, Pyliski provides an easy and out-of-the-box way of computing transfer functions between signals, notably biological ones from brain imaging techniques such as fMRI or fUS.


## Requirements


This package requires:

- Numpy

- Scipy

- Seaborn


## Installation


## Usage


There are two classes, `PyliskiSolver` to input data and compute transfer functions and, `PyliskiPlotter` to visualize and go through computed functions and convoluted results. See the well-documented `test.py` file to get a grasp of how to use them.

### Key-bindings using `PyliskiPlotter`
`PyliskiPlotter` builds upon Matploblib and Seaborn to allow for an exploration of the computed transfer functions. Use <kbd>←</kbd> or <kbd>→</kbd> to navigate, <kbd>Esc</kbd> to close.
