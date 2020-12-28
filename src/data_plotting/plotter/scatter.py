import logging
import itertools

import pandas as pd
import matplotlib.pyplot as pl

from .plotter import Plotter

logger = logging.getLogger(__name__)


class Scatter(Plotter):
    """
    Makes scatter plots for all combinations of input variables
    """

    def __init__(self,
                 data: pd.DataFrame,
                 plotting_dir: str):
        """
        :param data:                full data
        :param plotting_dir:        global plotting dir
        """
        super(Scatter, self).__init__(
            data=data,
            plotting_dir=plotting_dir,
            subdir_name='scatter'
        )

    def plot(self):
        variables = self._data.columns

        for c in itertools.combinations(variables, r=2):
            self._plot_combination(
                *c
            )

    def _plot_combination(self,
                          variable_a: str,
                          variable_b: str):
        """
        Plots separation power for the given variable

        :param variable_a:      name of the x variable
        :param variable_b:      name of the y variable
        """
        logger.info(
            f'Plotting scatter for: {variable_b} vs. {variable_a}'
        )
        fig, ax = pl.subplots()

        ax.scatter(
            self._data[variable_a],
            self._data[variable_b],
            color='tab:grey',
            alpha=0.5
        )

        ax.set_title(
            'Scatter plot',
            fontsize=20
        )
        ax.set_xlabel(
            variable_a,
            fontsize=16
        )
        ax.set_ylabel(
            variable_b,
            fontsize=16
        )
        pl.savefig(
            f'{self._plotting_dir}/{self.subdir_name}/'
            f'{variable_a}-{variable_b}.png'
        )
        pl.close(fig)

        logger.info(
            f'Image saved as: {self._plotting_dir}/{self.subdir_name}/'
            f'{variable_a}-{variable_b}.png'
        )
