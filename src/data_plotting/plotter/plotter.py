import pandas as pd

from abc import ABCMeta, abstractmethod


class Plotter:
    """
    Plots a specific feature of the given dataset
    """

    __metaclass__ = ABCMeta

    def __init__(self,
                 data: pd.DataFrame,
                 plotting_dir: str):
        """
        :param data:                full data
        :param plotting_dir:        global plotting dir
        """
        self._data = data
        self._plotting_dir = plotting_dir

    @abstractmethod
    def plot(self):
        """
        Makes the plot and saves it to the given file
        """
        pass
