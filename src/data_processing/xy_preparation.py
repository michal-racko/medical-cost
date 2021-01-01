import numpy as np
import pandas as pd


def prepare_x(data: pd.DataFrame) -> np.ndarray:
    """
    Prepares the X matrix for the regressor
    """
    columns = [
        data.age.values,
        data.bmi.values,
        (data.smoker == 'yes').astype(int)
    ]

    return np.vstack(columns).T


def prepare_y(data: pd.DataFrame) -> np.ndarray:
    """
    Prepares the y vector for the regressor
    """
    return data.severity.values
