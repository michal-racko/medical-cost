import numpy as np
import pandas as pd

from .weight import Weight


class WeiAge(Weight):
    """
    Calculates weights which will make
    the age distribution uniform
    """

    def __init__(self,
                 data: pd.DataFrame):
        super(WeiAge, self).__init__(
            data,
            name='wei_age'
        )

    def calculate(self) -> np.ndarray:
        values = self._data.age

        ages, counts = np.unique(
            values,
            return_counts=True
        )
        n_rows = len(self._data)
        n_categories = len(ages)

        wei = np.zeros(len(self._data))

        for age, count in zip(ages, counts):
            wei[values == age] = n_rows / (
                    (values == age).astype(int).sum() * n_categories
            )

        return wei
