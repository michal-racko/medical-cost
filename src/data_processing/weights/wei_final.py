import numpy as np
import pandas as pd

from .weight import Weight
from .wei_age import WeiAge
from .wei_smoker import WeiSmoker


class WeiFinal(Weight):
    """
    Combines all weights into one
    """

    def __init__(self,
                 data: pd.DataFrame):
        super(WeiFinal, self).__init__(
            data,
            name='wei_final'
        )

    def calculate(self) -> np.ndarray:
        wei_age = WeiAge(self._data).calculate()
        wei_smoker = WeiSmoker(self._data).calculate()

        wei = wei_age * wei_smoker

        return wei
