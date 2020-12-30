## Medical cost

The goal of this project is to train a regression model which will forecast health issue severity based on BMI and
smoking habits.

#### Usage

```
python main.py

  -c CONFIG, --config CONFIG
                        Path to the config file
  -m {plotting}, --mode {plotting}
                        Operation mode
```

Operation modes:

- plotting - makes all plots into the directory specified in the config file

Config options:

```
input_data_file   - path to the input data file (.csv)

plotting_dir      - a directory where plots will be saved
```