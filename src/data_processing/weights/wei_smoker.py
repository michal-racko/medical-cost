import numpy as np
import pandas as pd

from .weight_base import WeightBase


class WeiSmoker(WeightBase):
    """
    Calculates weights which will make
    the smoker distribution uniform
    """

    name = 'wei_smoker'

    def __init__(self,
                 data: pd.DataFrame):
        super(WeiSmoker, self).__init__(
            data,
        )

    def calculate(self) -> np.ndarray:
        is_smoker = self._data.smoker.values == 'yes'

        n_rows = len(self._data)
        n_smokers = is_smoker.astype(int).sum()
        n_non_smokers = (~is_smoker).astype(int).sum()

        wei = np.zeros(n_rows)
        wei[is_smoker] = n_rows / (2 * n_smokers)
        wei[~is_smoker] = n_rows / (2 * n_non_smokers)

        return wei
