import logging

import pandas as pd

from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor

from data_processing import prepare_x, prepare_y
from data_processing.weights import WeiFinal
from data_processing.metrics import (
    MAE,
    MSE
)
from data_processing import get_severity

logger = logging.getLogger(__name__)


def _evaluate_metrics(data: pd.DataFrame,
                      metric_classes):
    result = {}

    for metric_cls in metric_classes:
        metric = metric_cls(
            target_column='severity',
            predicted_column='predicted',
            weight_column='wei_final'
        )

        result[metric.name] = metric.evaluate(
            data
        )

    logger.info(f'Cross-validation results: {result}')


def cross_validate(data: pd.DataFrame,
                   model: GradientBoostingRegressor,
                   k_folds=5) -> pd.DataFrame:
    """
    Runs cross validation of the given model with the given data.

    :param data:        training data
    :param model:       regression model
    :param k_folds:     number of cross-val k-folds
    :returns:           a dataframe with the results written into
                        a column named "predicted"
    """
    wei = WeiFinal(data)
    data[wei.name] = wei.calculate()

    data['severity'] = get_severity(data)

    # Shuffle the data
    data = data.sample(n=len(data))

    kf = KFold(
        n_splits=k_folds
    )

    results = []

    for i_cross_val, (train, test) in enumerate(kf.split(data)):
        logger.info(f'Cross-val split: {i_cross_val}')

        train_df = data.iloc[train]
        test_df = data.iloc[test]

        model.fit(
            prepare_x(train_df),
            prepare_y(train_df),
            sample_weight=train_df[wei.name].values
        )

        test_df['predicted'] = model.predict(
            prepare_x(test_df)
        )

        results.append(
            test_df
        )

    result = pd.concat(results)

    metric_classes = (
        MAE,
        MSE,
    )

    _evaluate_metrics(
        result,
        metric_classes
    )

    return result
