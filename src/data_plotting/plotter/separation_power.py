import logging

import pandas as pd
import matplotlib.pyplot as pl

from .plotter import Plotter

logger = logging.getLogger(__name__)


class SeparationPower(Plotter):
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
        super(SeparationPower, self).__init__(
            data=data,
            plotting_dir=plotting_dir,
            subdir_name='separation_power'
        )

    def plot(self):
        """
        Makes separation plots for all input variables
        """
        median_charges = self._data.charges.median()

        pos_data = self._data.loc[self._data.charges <= median_charges]
        neg_data = self._data.loc[self._data.charges > median_charges]

        for column in self._data.columns:
            if column == 'charges':
                continue

            self._plot_single_variable(
                column,
                pos_data,
                neg_data
            )

    def _plot_single_variable(self,
                              variable_name: str,
                              pos_data: pd.DataFrame,
                              neg_data: pd.DataFrame):
        """
        Plots separation power for the given variable

        :param variable_name:   name of the variable column
        :param pos_data:        data corresponding to bellow median costs
        :param neg_data:        data corresponding to above median costs
        """
        logger.info(
            f'Plotting separation power for: {variable_name}'
        )
        fig, ax = pl.subplots()

        ax.hist(
            [
                pos_data[variable_name],
                neg_data[variable_name]
            ],
            color=[
                'tab:blue',
                'tab:orange'
            ],
            label=[
                'bellow median',
                'above median'
            ]
        )
        ax.legend()
        ax.set_title(
            'Separation power',
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
