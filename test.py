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

    # Test setInputBoxcar
    dt = 0.1
    baseline = 10.0
    up = 5.0
    total = 29.5
    boxcar = pyliski.set_input_boxcar(dt, baseline, up, total)

    # Check if boxcar is a numpy array and has the correct length
    assert isinstance(boxcar, np.ndarray), "Boxcar should be a numpy array."
    assert len(boxcar) == int(
        total / dt
    ), "Boxcar length does not match expected length."

    # Test setOutputData with valid data
    output_data = np.loadtxt("XV4_RelativeRBC_ET_200mV_5sec.txt")
    output_data = np.array([o[1] for o in output_data])
    pyliski.set_output_data(output_data)

    # Check if outputData is set correctly
    assert hasattr(pyliski, "outputData"), "Output data was not set."
    assert np.array_equal(
        pyliski.outputData, output_data
    ), "Output data does not match input data."

    pyliski.visualize_data()

    pyliski.set_transfer_model(gamma)
    options = {"bounds": [(0.001, 10.0), (0.001, 10.0), (0.001, 10.0), (0.001, 10.0)]}
    pyliski.set_options(options)

    pyliski.run(2)

    plotter = PyliskiPlotter(pyliski)
    plotter.plot_results()

    print("All tests passed.")


if __name__ == "__main__":
    test_pyliski()
    print("Pyliski module tests completed successfully.")
