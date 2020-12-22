import pandas as pd

from data_plotting import SeparationPower

INPUT_DATA_FILE = '../data/insurance.csv'
PLOTTING_DIR = '../plotting'


def main_plotting(_data: pd.DataFrame):
    plotter = SeparationPower(
        data=_data,
        plotting_dir=PLOTTING_DIR
    )

    plotter.plot()


if __name__ == '__main__':
    data = pd.read_csv(INPUT_DATA_FILE)

    main_plotting(data)
