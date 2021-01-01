import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

from .plotter import Plotter
from data_processing.metrics import (
    MSE,
    MAE,
)
from data_processing.weights import WeiFinal

logger = logging.getLogger(__name__)


class CrossValidation(Plotter):
    """
    Plots results of the cross validation
    """

    def __init__(self,
                 data: pd.DataFrame,
                 plotting_dir: str):
        """
        :param data:                full data
        :param plotting_dir:        global plotting dir
        """
        super(CrossValidation, self).__init__(
            data,
            plotting_dir,
            subdir_name='cross_validation'
        )

        self._metrics = (
            MSE,
            MAE,
        )

        self._plot_columns = (
            'age',
            'sex',
            'bmi',
        )

    def plot(self):
        for metric_cls in self._metrics:
            metric = metric_cls(
                target_column='severity',
                predicted_column='predicted',
                weight_column='wei_final'
            )
            self._data[metric.name] = metric.evaluate_rows(
                self._data
            )

            self._plot_metric_histogram(metric.name)

    def _plot_metric_histogram(self,
                               metric_name: str):
        logger.info(
            f'Plotting error histogram for: {metric_name}'
        )
        fig, ax = pl.subplots()

        hist, bins = np.histogram(
            self._data[metric_name],
            weights=self._data[WeiFinal.name],
            bins=100
        )

        ax.set_title(
            'Error histogram',
            fontsize=20
        )
        ax.set_xlabel(
            metric_name,
            fontsize=16
        )

        ax.bar(
            (bins[1:] + bins[:-1]) / 2,
            hist,
            width=(bins[1:] - bins[:-1])[0]
        )

        pl.savefig(
            f'{self._plotting_dir}/{self.subdir_name}/{metric_name}-histo.png'
        )
        pl.close(fig)

        logger.info(
            f'Image saved as: {self._plotting_dir}/{self.subdir_name}/'
            f'{metric_name}-histo.png'
        )
