import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import matplotlib.patches as mpatches

from .plotter import Plotter
from data_processing.weights import (
    WeiAge,
    WeiSmoker,
    WeiFinal
)

logger = logging.getLogger(__name__)


class Reweighting(Plotter):
    """
    Plots effects of reweighting on the input variables
    """

    def __init__(self,
                 data: pd.DataFrame,
                 plotting_dir: str):
        """
        :param data:                full data
        :param plotting_dir:        global plotting dir
        """
        super(Reweighting, self).__init__(
            data=data,
            plotting_dir=plotting_dir,
            subdir_name='reweighting'
        )

        self._weights = [
            WeiSmoker,
            WeiAge,
            WeiFinal
        ]

        self._wei_names = []

        self._available_colors = [
            'k',
            'tab:red',
            'tab:green',
            'tab:blue',
            'tab:orange',
            'tab:purple',
            'tab:brown',
            'tab:cyan',
            'tab:pink',
        ]

    def plot(self):
        for wei_cls in self._weights:
            wei = wei_cls(self._data)

            self._data[wei.name] = wei.calculate()

            self._wei_names.append(wei.name)

        for column in self._data.columns:
            if column in self._wei_names:
                continue

            self._plot_single_variable(column)

    def _plot_single_variable(self,
                              variable_name: str):
        """
        Plots effects of reweighting on the given variable

        :param variable_name:       name of the corresponding column
        """
        logger.info(
            f'Plotting reweighting for: {variable_name}'
        )
        fig, ax = pl.subplots()

        values = self._data[variable_name].values
        patches = []

        for i, wei_name in enumerate([None] + self._wei_names):
            try:
                color = self._available_colors[i]

            except IndexError:
                raise IndexError(
                    'Please define more plot colors in _available_colors'
                )

            if wei_name is None:
                weights = None
                name = 'nominal'

            else:
                weights = self._data[wei_name]
                name = wei_name

            patches.append(
                mpatches.Patch(
                    color=color,
                    label=name
                )
            )

            if values.dtype in (int, float):
                hist_data = self._prepare_numeric_histogram(
                    values,
                    weights=weights
                )

                ax.bar(
                    hist_data['bin_centres'],
                    hist_data['bin_heights'],
                    width=hist_data['bin_width'],
                    fill=False,
                    edgecolor=color
                )

            else:
                hist_data = self._prepare_categorical_histogram(
                    values,
                    weights=weights
                )

                ax.bar(
                    hist_data['bin_centres'],
                    hist_data['bin_heights'],
                    width=hist_data['bin_width'],
                    fill=False,
                    edgecolor=color
                )
                ax.set_xticks(hist_data['bin_centres'])
                ax.set_xticklabels(hist_data['bin_names'])

        ax.legend(
            handles=patches
        )

        ax.set_title(
            'Reweighting',
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

    @staticmethod
    def _prepare_numeric_histogram(values,
                                   weights=None) -> dict:
        """
        Prepares a histogram of numerical values (e.g. integers)

        :param values:      to fill the histogram
        :param weights:     corresponding stats. weights
        :return:            data for the plot
        """
        n_unique = len(np.unique(values))

        hist, bins = np.histogram(
            values,
            weights=weights,
            bins=10 if n_unique > 10 else n_unique
        )

        return {
            'bin_heights': hist,
            'bin_centres': (bins[1:] + bins[:-1]) / 2,
            'bin_width': (bins[1:] - bins[:-1])[0],
            'bin_names': (bins[1:] + bins[:-1]) / 2
        }

    @staticmethod
    def _prepare_categorical_histogram(values,
                                       weights=None) -> dict:
        """
        Prepares a histogram of categorical values

        :param values:      to fill the histogram
        :param weights:     corresponding stats. weights
        :return:            data for the plot
        """
        labels = []
        enum_map = {}

        for i, category in enumerate(np.unique(values)):
            labels.append(category)
            enum_map[category] = i

        hist, bins = np.histogram(
            [enum_map[v] for v in values],
            bins=len(labels),
            weights=weights
        )

        return {
            'bin_heights': hist,
            'bin_centres': (bins[1:] + bins[:-1]) / 2,
            'bin_width': (bins[1:] - bins[:-1])[0],
            'bin_names': labels
        }
