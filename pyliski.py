# -*- coding: utf-8 -*-
"""
pyliski.py

"""

from typing import Callable, List, Tuple
from scipy.optimize import OptimizeResult

import numpy as np


class PyliskiSovlver:
    """
    Pyliski class for handling input boxcar, output data, and transfer model.
    This class provides methods to set input boxcar, output data, visualize data,
    and optimize the transfer model.
    """

    def __init__(self):
        """
        Initialize the PyliskiSovlver class.
        """
        self.time = np.array([])
        self.boxcar = np.array([])
        self.outputData = np.array([])
        self.transferModel = None
        self.bounds = np.array([])
        self.minim_options = np.array([])

        self.last_optimized = []

    def set_input_boxcar(
        self, dt: float, baseline: float, up: float, total: float
    ) -> np.ndarray:
        """
        Set the time array and generate a boxcar function based on the provided parameters.

        :param dt: Time step.
        :param baseline: Baseline duration.
        :param up: Up period duration.
        :param total: Total duration.
        """
        self.time = np.arange(0, total, dt)
        self.boxcar = np.zeros_like(self.time)
        start = int(baseline / dt)
        end = int((baseline + up) / dt)
        self.boxcar[start:end] = 1.0

        return self.boxcar

    def set_output_data(self, data: np.ndarray):
        """
        Set the output data for the Pyliski module.

        :param data: Output data as a numpy array, single dimension, time step should correspond to dt of boxcar.
        :raises ValueError: If the data is not a numpy array or if the dimensions do not match.
        """
        if self.time.size == 0:
            raise ValueError("Time array must be set before setting output data.")
        if not isinstance(data, np.ndarray):
            raise ValueError("Output data must be a numpy array.")
        if data.ndim != 1:
            raise ValueError("Output data must be a single dimension array.")
        if len(data) != len(self.time):
            raise ValueError("Output data length must match the time array length.")

        self.outputData = data

    def visualize_data(self):
        """
        Visualize the boxcar function and output data using matplotlib.
        Attention: This method will stop the execution of the program to show the plot.
        :raises ValueError: If boxcar or output data is not set.
        """
        import matplotlib.pyplot as plt

        if self.boxcar.size == 0 or self.outputData.size == 0:
            raise ValueError("Boxcar and output data must be set before visualization.")

        plt.figure(figsize=(10, 5))
        # Boxcar is normalized to outputData for better visualization
        factor = np.max(self.outputData) / np.max(self.boxcar)  # / 1
        plt.plot(self.time, self.boxcar * factor, label="Boxcar Function", color="blue")
        plt.plot(self.time, self.outputData, label="Output Data", color="red")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title("Boxcar Function and Output Data")
        plt.legend()
        plt.grid()
        plt.show()

    def set_options(self, options: dict):
        """
        Set options for the Pyliski module.

        :param options: Dictionary of options. Should contains at least bounds for optimization.
            Can contain a key named minim_options for additional settings transfered to the minimizer.
        :raises ValueError: If options is not a dictionary or do not contain 'bounds'.
        """
        if not isinstance(options, dict):
            raise ValueError("Options must be a dictionary.")
        if "bounds" not in options:
            raise ValueError("Options must contain 'bounds' for optimization.")

        self.bounds = options["bounds"]
        self.minim_options = options.get("minim_options", None)

    def set_transfer_model(self, model_func: Callable):
        """
        Set the transfer model for the Pyliski module.

        :param model_func: Transfer model function as a Callable.
        """
        if not callable(model_func):
            raise ValueError("Transfer model must be a callable function.")
        if self.boxcar.size == 0:
            raise ValueError("Boxcar must be set before setting the transfer model.")
        if self.outputData.size == 0:
            raise ValueError(
                "Output data must be set before setting the transfer model."
            )
        self.transferModel = model_func

    def _optimize(self) -> OptimizeResult:
        """
        Optimize the transfer model parameters using the provided bounds.

        :param bounds: List of tuples specifying the bounds for each parameter.
        :return: Optimized parameters.
        """
        from scipy.optimize import dual_annealing

        from transfer_utils import get_residuals

        if not hasattr(self, "transferModel"):
            raise ValueError("Transfer model must be set before optimization.")
        if not hasattr(self, "bounds"):
            raise ValueError("Bounds must be set before optimization.")

        result = dual_annealing(
            get_residuals,
            bounds=self.bounds,
            args=(
                self.time,
                self.transferModel,
                self.boxcar,
                self.outputData,
            ),
            minimizer_kwargs=self.minim_options,
        )

        if not result.success:
            raise RuntimeError("Optimization failed: " + result.message)

        import matplotlib.pyplot as plt

        print(f"Optimization successful: {result.x}")
        foundTF = self.transferModel(self.time, result.x)  # type: ignore
        plt.plot(self.time, foundTF, label="Optimized Model")
        plt.plot(self.time, self.outputData, label="Output Data", color="red")
        plt.plot(
            self.time,
            np.convolve(foundTF, self.boxcar)[1 : len(self.time) + 1],
            label="Convolved Model",
            color="green",
        )
        plt.legend()
        plt.show()

        return result

    def run(self, n: int) -> List[OptimizeResult]:
        """
        Run the optimization process multiple times.

        :param n: Number of optimization runs.
        :return: List of optimization results.
        """
        from tqdm import tqdm

        results = []
        for i in tqdm(range(n)):
            result = self._optimize()
            results.append(result)

        self.last_optimized = self._sort_results(results)
        return self.last_optimized

    def _sort_results(self, results: List[OptimizeResult]) -> List[OptimizeResult]:
        """
        Sort the optimization results based on the residuals.

        :param results: List of optimization results.
        :return: Sorted list of optimization results.
        """
        return sorted(results, key=lambda res: res.fun)

    def save_solver(self, filename: str):
        """
        Save the PyliskiSovlver instance to a file.

        :param filename: Name of the file to save the instance.
        """
        import pickle

        with open(filename, "wb") as f:
            pickle.dump(self, f)


class PyliskiPlotter:
    """
    Pyliski class for plotting the results of the optimization.
    This class provides methods to plot the optimized transfer function and the output data.
    """

    def __init__(self, pyliski_solver: PyliskiSovlver):
        """
        Initialize the PyliskiPlotter class.

        :param pyliski_solver: Instance of PyliskiSovlver to plot results from.
        """
        self.pyliski_solver = pyliski_solver

    def plot_results(self):
        """
        Plot the results of the optimization.
        """
        if not hasattr(self.pyliski_solver, "last_optimized"):
            raise ValueError("No optimization results available to plot.")

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        sns.set_theme(style="darkgrid")

        # FIRST SUBPLOT: Input Data
        sns.lineplot(
            x=self.pyliski_solver.time,
            y=self.pyliski_solver.boxcar,
            ax=axs[0],
            label="Boxcar Function",
        )
        sns.lineplot(
            x=self.pyliski_solver.time,
            y=self.pyliski_solver.outputData,
            ax=axs[0],
            label="Output Data",
        )
        axs[0].set_title("Input Data")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Amplitude (a.u.)")

        # SECOND SUBPLOT: Optimized Transfer Function
        best_result = self.pyliski_solver.last_optimized[
            0
        ]  # Assuming the first result is the best
        optimized_tf = self.pyliski_solver.transferModel(  # type: ignore
            self.pyliski_solver.time, best_result.x
        )
        sns.lineplot(
            x=self.pyliski_solver.time,
            y=optimized_tf,
            ax=axs[1],
            label="Optimized Transfer Function",
        )
        axs[1].set_title("Optimized Transfer Function")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Amplitude (a.u.)")
        axs[1].legend()
        axs[1].grid()

        # THIRD SUBPLOT: Convolved Model
        convolved_model = np.convolve(optimized_tf, self.pyliski_solver.boxcar)[
            1 : len(self.pyliski_solver.time) + 1
        ]
        sns.lineplot(
            x=self.pyliski_solver.time,
            y=self.pyliski_solver.outputData,
            ax=axs[2],
            label="Output Data",
        )
        sns.lineplot(
            x=self.pyliski_solver.time,
            y=convolved_model,
            ax=axs[2],
            label="Convolved Model",
        )
        axs[2].set_title("Applied Model")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Amplitude (a.u.)")
        axs[2].legend()
        axs[2].grid()

        plt.tight_layout()
        plt.show()
        plt.pause(5)
