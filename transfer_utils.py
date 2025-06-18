# -*- coding: utf-8 -*-
"""
transfer_utils.py

"""
from typing import Callable, Tuple, List
import numpy as np


def gamma(time: np.ndarray, params: List[float]) -> np.ndarray:
    """
    Gamma function for modeling transfer functions.

    :param time: Time array as a numpy array.
    :param params: List of four parameters [p1, p2, p3, p4].
    :return: Computed gamma function values as a numpy array.
    :raises ValueError: If time is not a numpy array or if params does not have exactly four elements.
    """

    from scipy.special import gamma as scipy_gamma

    if not isinstance(time, np.ndarray):
        raise ValueError("Time must be a numpy array.")
    if time.ndim != 1:
        raise ValueError("Time must be a single dimension array.")

    if len(params) != 4:
        raise ValueError("Params must be a list of four parameters: [p1, p2, p3, p4].")

    p1, p2, p3, p4 = params

    time_p3 = np.subtract(time, p3)

    gamma_result = np.multiply(
        np.multiply(time_p3 >= 0, p4),
        np.divide(
            np.multiply(
                np.multiply(np.power(time_p3, p1 - 1), np.power(p2, p1)),
                np.exp(np.multiply(-p2, time_p3)),
            ),
            scipy_gamma(p1),
        ),
    )

    return np.nan_to_num(gamma_result)


def get_residuals(
    params: List[float],
    time: np.ndarray,
    transferFunc: Callable[[np.ndarray, List[float]], np.ndarray],
    boxcar: np.ndarray,
    outputData: np.ndarray,
) -> float:
    """
    Calculate the residuals between the modeled output and the actual output data.
    :param params: List of parameters for the transfer function.
    :param time: Time array as a numpy array.
    :param transferFunc: Transfer function to be used for modeling.
    :param boxcar: Boxcar function as a numpy array.
    :param outputData: Actual output data as a numpy array.
    :return: Sum of squared residuals.
    """
    transfer_Func_values = transferFunc(time, params)
    modeled_output = np.convolve(transfer_Func_values, boxcar)[1 : len(time) + 1]
    residuals = outputData - modeled_output
    residuals_squared = np.square(residuals)
    return np.sum(residuals_squared)
