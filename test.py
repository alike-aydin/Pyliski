# -*- coding: utf-8 -*-

"""
test.py

"""

import numpy as np
from pyliski import PyliskiSolver, PyliskiPlotter
from transfer_utils import gamma


def test_pyliski():
    """
    Test the Pyliski class methods.
    """
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


if __name__ == "__main__":
    test_pyliski()
    print("Pyliski module tests completed successfully.")
