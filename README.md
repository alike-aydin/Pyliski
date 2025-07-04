# Pyliski

This package implements a Python version of the Iliski woftware, previously described in [Aydin et al. (2021)](https://doi.org/10.1371/journal.pcbi.1008614) and available at [https://github.com/alike-aydin/Iliski](https://github.com/alike-aydin/Iliski). Shortly, Pyliski provides an easy and out-of-the-box way of computing transfer functions between signals, notably biological ones from brain imaging techniques such as fMRI or fUS.

## Requirements

This package requires:

- Numpy

- Scipy

- Seaborn

## Installation

## Usage

There are two classes, `PyliskiSolver` to input data and compute transfer functions and, `PyliskiPlotter` to visualize and go through computed functions and convoluted results. See the well-documented `test.py` file to get a grasp of how to use them or see below.

### Key-bindings using `PyliskiPlotter`

`PyliskiPlotter` builds upon Matploblib and Seaborn to allow for an exploration of the computed transfer functions. Use <kbd>←</kbd> or <kbd>→</kbd> to navigate, <kbd>Esc</kbd> to close.

### Code example

```python
pyliski = PyliskiSolver()

# Defining the input boxcar parameters in seconds
# dt: time step, baseline: baseline value, up: up value, total: total duration
dt = 0.1
baseline = 10.0
up = 5.0
total = 29.5
# Setting the input boxcar parameters and computing the boxcar function
pyliski.set_input_boxcar(dt, baseline, up, total)

# Loading the output data from a file, keeping only the second column
output_data = np.loadtxt("XV4_RelativeRBC_ET_200mV_5sec.txt")
output_data = np.array([o[1] for o in output_data])
# Setting the output data for the Pyliski solver
pyliski.set_output_data(output_data)

# Visualizing the input boxcar and output data
pyliski.visualize_data()

# Setting the transfer model to the gamma function
# The gamma function is defined in the transfer_utils module and takes time and parameters as inputs
pyliski.set_transfer_model(gamma)

# Setting the parameters for the optimization
# Boundaries are mandatory to run the optimization
# Here we set the bounds for the four parameters of the gamma function
# Other parameters can be set using in the options dictionary, see the Pyliski documentation for more details.
options = {
    "bounds": [(0.001, 10.0), (0.001, 10.0), (0.001, 10.0), (0.001, 10.0)],
    "x0": [1.0, 1.0, 1.0, 1.0],  # Initial guess for the parameters
}
# Setting the options for the Pyliski solver
pyliski.set_options(options)

# Running two iterations of the optimization
pyliski.run(20)

# Creating a plotter instance to visualize the results
# The PyliskiPlotter class is used to plot the results of the optimization
# It takes the PyliskiSolver instance as input
plotter = PyliskiPlotter(pyliski)
plotter.plot_results()

print("All tests passed.")
```
