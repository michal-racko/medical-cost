import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod


class MetricBase:
    """
    A base class for evaluation metrics.

    Can be applied to a result DataFrame
    """
    __metaclass__ = ABCMeta

    name = None

    def __init__(self,
                 target_column: str,
                 predicted_column: str,
                 weight_column: str):
        """
        :param target_column:       Name of the target column
                                    (must be present in the given data)
        :param predicted_column:    Name of the column with regressor results
                                    (must be present in the given data)
        :param weight_column:       Name of the stat weight column
        """
        self._target_column = target_column
        self._predicted_column = predicted_column
        self._weight_column = weight_column

    @abstractmethod
    def evaluate(self,
                 data: pd.DataFrame) -> float:
        """
        Evaluates the metric for the whole DataFrame
        (sums over all rows)

        :param data:        regressor results
        :returns:           value of the metric
        """
        pass

    @abstractmethod
    def evaluate_rows(self,
                      data: pd.DataFrame) -> np.ndarray:
        """
        Evaluates the metric for each row of the given DataFrame

        :param data:        regressor results
        :returns:           values of the metric
        """
        pass
