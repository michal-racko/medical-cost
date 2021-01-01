import logging

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

from data_plotting import (
    SeparationPower,
    VariableHistogram,
    Scatter,
    Reweighting,
    CrossValidation
)
from tools.command_line import parse_args, OperationMode
from tools.config import Config
from training import cross_validate

logging.basicConfig(
    format='%(asctime)s %(levelname)-5s %(name)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None


def main_analysis(_data: pd.DataFrame,
                  _config: Config):
    """
    Makes all the plots needed for the analysis

    :param _data:           examined data
    :param _config:         configurations from the config file
    """
    plotters = [
        SeparationPower,
        VariableHistogram,
        Scatter,
        Reweighting
    ]

    for cls in plotters:
        plotter = cls(
            data=_data,
            plotting_dir=_config.plotting_dir
        )
        plotter.plot()


def main_cross_val(_data: pd.DataFrame,
                   _config: Config):
    """
    Runs cross-validation of the GBR

    :param _data:           examined data
    :param _config:         configurations from the config file
    """
    plotters = [
        CrossValidation
    ]

    model = GradientBoostingRegressor()
    # model = LinearRegression()

    results = cross_validate(
        _data,
        model
    )

    for cls in plotters:
        plotter = cls(
            data=results,
            plotting_dir=_config.plotting_dir
        )
        plotter.plot()


if __name__ == '__main__':
    args = parse_args()

    config = Config.read(
        args.config
    )

    logger.info('Reading data')
    data = pd.read_csv(config.input_data_file)

    if args.mode == OperationMode.ANALYSIS:
        logger.info('Analysis mode selected')
        main_analysis(data, config)

    elif args.mode == OperationMode.CROSS_VALIDATION:
        logger.info('Cross-validation mode selected')
        main_cross_val(data, config)

    logger.info('All done')
