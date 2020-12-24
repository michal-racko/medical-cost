import logging

import pandas as pd

from data_plotting import (
    SeparationPower,
    VariableHistogram
)
from tools.command_line import parse_args, OperationMode
from tools.config import Config

logging.basicConfig(
    format='%(asctime)s %(levelname)-5s %(name)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def main_plotting(_data: pd.DataFrame,
                  _config: Config):
    """
    Makes all the plots needed for the analysis

    :param _data:           examined data
    :param _config:         configurations from the config file
    """
    plotters = [
        SeparationPower,
        VariableHistogram
    ]

    for cls in plotters:
        plotter = cls(
            data=_data,
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

    if args.mode == OperationMode.PLOTTING:
        logger.info('Plotting mode selected')
        main_plotting(data, config)

    logger.info('All done')
