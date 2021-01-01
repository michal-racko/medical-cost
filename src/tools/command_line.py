import enum
import argparse


class OperationMode(enum.IntEnum):
    ANALYSIS = 1
    CROSS_VALIDATION = 2

    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return OperationMode[s.upper()]

        except KeyError:
            return s


def parse_args() -> argparse.Namespace:
    """
    Parses command line arguments

    :returns:       parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        prog='A simple ML tool for disease prevention'
    )

    parser.add_argument(
        '-c',
        '--config',
        required=True,
        help='Path to the config file'
    )

    parser.add_argument(
        '-m',
        '--mode',
        required=True,
        help='Operation mode',
        type=OperationMode.argparse,
        choices=list(OperationMode)
    )

    return parser.parse_args()
