import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod


class Weight:
    """
    Represents a statistical weight
    """

    __metaclass__ = ABCMeta

    def __init__(self,
                 data: pd.DataFrame,
                 name: str):
        self._data = data

        self.name = name

    @abstractmethod
    def calculate(self) -> np.ndarray:
        """
        Assigns a weight to each row of the dataset such that
        the sum of all weights is equal to the number of rows
        in the dataset.

        :returns:       stats weigths as a new column for the dataset
        """
        pass
