import numpy as np
import pandas as pd

from .metric_base import MetricBase


class MSE(MetricBase):
    """
    Mean square error
    """

    name = 'mse'

    def __init__(self,
                 target_column: str,
                 predicted_column: str,
                 weight_column: str):
        super(MSE, self).__init__(
            target_column=target_column,
            predicted_column=predicted_column,
            weight_column=weight_column
        )

    def evaluate(self,
                 data: pd.DataFrame) -> float:
        return (
                self.evaluate_rows(data) * data[self._weight_column]
        ).mean()

    def evaluate_rows(self,
                      data: pd.DataFrame) -> np.ndarray:
        return (
                       data[self._predicted_column] -
                       data[self._target_column]
               ) ** 2
