import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod


class WeightBase:
    """
    A base class for statistical weight calculators
    """

    __metaclass__ = ABCMeta

    name = None

    def __init__(self,
                 data: pd.DataFrame):
        """
        :param data:        Input data
        """
        self._data = data

    @abstractmethod
    def calculate(self) -> np.ndarray:
        """
        Assigns a weight to each row of the dataset such that
        the sum of all weights is equal to the number of rows
        in the dataset.

        :returns:       stats weights as a new column for the dataset
        """
        pass
