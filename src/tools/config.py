import yaml


class ConfigurationError(Exception):
    pass


class Config:
    """
    Abstracts the config file.

    Configuration can be retrieved either by calling
    Config.get(key_0, key_1, ...), using getitem (config[key])
    or getattr (config.key)
    """

    def __init__(self,
                 config: dict):
        self._config = config

        self._required_keys = [
            'input_data_file',
            'plotting_dir'
        ]

    def _check_keys(self):
        """
        Checks if all required keys are present in the config.

        :raises ConfigurationError:     if a required key is missing
        """
        for k in self._required_keys:
            if k not in self._config:
                raise ConfigurationError(
                    f'"{k}" not found in the config file, but it is required'
                )

    @classmethod
    def read(cls,
             filepath: str):
        """
        Reads configurations from the given file.

        :param filepath:            Path to the config file

        :raises FileNotFoundError:  If no yml file is found at the given path
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = yaml.load(
                    f,
                    Loader=yaml.SafeLoader
                )

        except FileNotFoundError:
            raise FileNotFoundError(
                f'No config file found at: {filepath}'
            )

        return cls(config)

    def __getitem__(self, key):
        if key in self._config:
            return self._config[key]

        else:
            raise ConfigurationError(
                f'"{key}" not found in the config file'
            )

    def __getattr__(self, item):
        if item in self._config:
            return self._config[item]

        else:
            raise ConfigurationError(
                f'"{item}" not found in the config file'
            )

    def get(self,
            *keys,
            default=None):
        """
        Recursively goes through the given keys and returns
        the final value or default if no value found in
        the nested configurations.

        :param keys:            keys to be looped over
        :param default:         default to be returned if the key doesn't exist
        :return:                value at the desired position in the config
        """
        conf = self._config

        for key in keys:
            if key in conf:
                conf = conf[key]

            else:
                return default

        return conf
