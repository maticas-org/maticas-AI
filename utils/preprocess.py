import os
import numpy as np
import pandas as pd

CURRFILEDIR = os.path.dirname(os.path.abspath(__file__))
PROJECTPATH = os.path.dirname(CURRFILEDIR)

DATASETPATH  = '/datasets/ambient_measurements/2022/'
FULLDATAPATH = PROJECTPATH + DATASETPATH
OUTPUTDIR    = PROJECTPATH + '/datasets/ambient_measurements/2022/'

FILES = {
    'lux': 'lux.csv',
    'humidity': 'hum.csv',
    'wtemp': 'wtemp.csv',
    'temperature': 'temp.csv',
    'pressure': 'pressure.csv',
    'ec': 'ec.csv',
    'ph': 'ph.csv',
}

def read_files():
    """
    Read the data files for each alias and store them in a dictionary.

    Returns:
    data (dict): A dictionary containing the dataframes for each alias.
    """
    data = {}

    for alias, filename in FILES.items():
        data[alias] = pd.read_csv(FULLDATAPATH + filename)
        data[alias]['time'] = pd.to_datetime(data[alias]['time'])
        data[alias].set_index('time', inplace=True)

    return data


def get_maximum_sampling_period(data):
    """
    Get the maximum sampling period among the dataset points.

    Args:
    data (dict): A dictionary containing the dataframes for each alias.

    Returns:
    max_sampling_period (int): The maximum sampling period in seconds.
    starting_time (datetime): The starting time of the dataset.
    """
    samplings = []
    starting_time = []

    for alias, df in data.items():
        init = df.iloc[0].name
        end = df.iloc[1].name
        samplings.append((end - init).total_seconds())
        starting_time.append(init)

    return int(max(samplings)), min(starting_time)


def build_dataset(data:         dict,
                  window_sizes: list):
    """
    Build the dataset by calculating rolling features for each alias and window size.

    Args:
    data (dict): A dictionary containing the dataframes for each alias.
    window_sizes (list): A list of window sizes (as strings) in different units of time.

    Returns:
    None
    """

    rolling_features = {}

    for alias, df in data.items():
        for win_sz in window_sizes:

            rolling_window = f"{win_sz}"
            mean = df.rolling(rolling_window).mean()
            var = df.rolling(rolling_window).var()
            max_ = df.rolling(rolling_window).max()
            min_ = df.rolling(rolling_window).min()

            rolling_features[f'mean_{rolling_window}'] = mean
            rolling_features[f'var_{rolling_window}'] = var
            rolling_features[f'max_{rolling_window}'] = max_
            rolling_features[f'min_{rolling_window}'] = min_

        for title, feature in rolling_features.items():
            df[title] = feature

        df.to_csv(f'{OUTPUTDIR}{alias}_preprocessed.csv')


def check_window_sizes(minimum_window_size, candidate_window_sizes):
    """
    Check the window sizes and choose the ones that make sense.

    Args:
    minimum_window_size (int): The maximum sampling period among the dataset points.
    candidate_window_sizes (list): A list of candidate window sizes.

    Returns:
    valid_window_sizes (set): A set of valid window sizes.
    """
    valid_window_sizes = set()

    for win_sz in candidate_window_sizes:
        valid_window_sizes.add(max(minimum_window_size, win_sz))

    return valid_window_sizes


print("Reading files...")
data = read_files()


max_period, starting_time = get_maximum_sampling_period(data)
print(f"Maximum sampling period: {max_period} seconds")
print(f"The chosen sampling periods should be greater than {max_period} seconds.\n")

#the sampling periods should be bigger than the 'max_sampling_period'
sampling_periods = ['30min', '1d', '2d', '3d', '7d']
print(f"Default sampling periods {sampling_periods}")

print("Extracting features and building datasets...")
build_dataset(data, sampling_periods)

print(f"Saved datasets at: {OUTPUTDIR}")



