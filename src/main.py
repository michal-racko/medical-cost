import pandas as pd

from data_plotting import SeparationPower
from tools.command_line import parse_args, OperationMode
from tools.config import Config


def main_plotting(_data: pd.DataFrame,
                  _config: Config):
    """
    Makes all the plots needed for the analysis

    :param _data:           examined data
    :param _config:         configurations from the config file
    """
    plotters = [
        SeparationPower(
            data=_data,
            plotting_dir=_config.plotting_dir
        )
    ]

    for plotter in plotters:
        plotter.plot()


if __name__ == '__main__':
    args = parse_args()

    config = Config.read(
        args.config
    )

    data = pd.read_csv(config.input_data_file)

    if args.mode == OperationMode.PLOTTING:
        main_plotting(data, config)
