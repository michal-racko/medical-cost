import numpy as np
import pandas as pd


def get_severity(data: pd.DataFrame) -> np.ndarray:
    """
    Calculates healt issue severity
    in range 0 to 1 based on the charges.

    :param data:        Input data
    :return:            Severity in the range 0 to 1
    """
    df = data[['charges']].copy()

    df.sort_values(
        by='charges',
        inplace=True
    )

    df['severity'] = np.arange(
        len(df)
    ) / len(df)

    df.sort_index(
        inplace=True
    )

    return df.severity.values
