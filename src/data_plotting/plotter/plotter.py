import pathlib

import pandas as pd

from abc import ABCMeta, abstractmethod


class Plotter:
    """
    Plots a specific feature of the given dataset
    """

    __metaclass__ = ABCMeta

    def __init__(self,
                 data: pd.DataFrame,
                 plotting_dir: str,
                 subdir_name: str):
        """
        :param data:                full data
        :param plotting_dir:        global plotting dir
        """
        self._data = data
        self._plotting_dir = plotting_dir

        self.subdir_name = subdir_name

        self._assure_subdir()

    @abstractmethod
    def plot(self):
        """
        Makes the plot and saves it to the given file
        """
        pass

    def _assure_subdir(self):
        """
        Creates the separation_power subdir in the main plotting dir
        if it doesn't already exist
        """
        pathlib.Path(
            f'{self._plotting_dir}/{self.subdir_name}'
        ).mkdir(
            parents=True,
            exist_ok=True
        )
