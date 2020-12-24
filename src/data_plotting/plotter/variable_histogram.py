import logging

import pandas as pd
import matplotlib.pyplot as pl

from .plotter import Plotter

logger = logging.getLogger(__name__)


class VariableHistogram(Plotter):
    """
    Makes separation plots for all
    """

    def __init__(self,
                 data: pd.DataFrame,
                 plotting_dir: str):
        """
        :param data:                full data
        :param plotting_dir:        global plotting dir
        """
        super(VariableHistogram, self).__init__(
            data=data,
            plotting_dir=plotting_dir,
            subdir_name='variable_histograms'
        )

    def plot(self):
        """
        Makes simple histograms for all the variables
        from the given dataset
        """
        for column in self._data.columns:
            self._plot_single_variable(column)

    def _plot_single_variable(self,
                              variable_name: str):
        """
        Plots histogram for the given variable

        :param variable_name:   name of the variable column
        """
        logger.info(
            f'Plotting a histogram for: {variable_name}'
        )
        fig, ax = pl.subplots()

        ax.hist(
            self._data[variable_name],
            color='tab:green'
        )
        ax.set_title(
            'Histogram',
            fontsize=20
        )
        ax.set_xlabel(
            variable_name,
            fontsize=16
        )
        pl.savefig(
            f'{self._plotting_dir}/{self.subdir_name}/{variable_name}.png'
        )
        pl.close(fig)

        logger.info(
            f'Image saved as: {self._plotting_dir}/{self.subdir_name}/'
            f'{variable_name}.png'
        )
