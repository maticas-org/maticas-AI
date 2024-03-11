import pandas as pd 
import numpy as np
from copy import deepcopy
from typing import List, Dict

def interpolate_and_resample(data: pd.DataFrame,
                             freq: str = '10min',
                             method: str = 'spline',
                             order: int = 3) -> pd.DataFrame:

    """
    Interpolate and resample the dataframe. 
    To fill missing values, the method uses the interpolation method and the order.

    Args:
    -----------
        - data (pd.DataFrame): The dataframe containing the data.
        - freq (str): The frequency to resample the data.
        - method (str): The method to interpolate the data.

    Returns:
    -----------
        - data (pd.DataFrame): The dataframe with the interpolated and resampled data.
    """

    data = deepcopy(data)
    data = data.interpolate(method=method, order=order).\
                resample(freq).mean().interpolate(method=method, order=order)
    return data 


def encode_date(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encode ciclically encode the date and time, separatelly.

    Args:
    -----------
        - data (pd.DataFrame): The dataframe containing the data.
        - alias (str): The alias of the dataframe.

    Returns:
    -----------
        - data (pd.DataFrame): The dataframe with the encoded date.
    """

    data = deepcopy(data)

    data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 23.0) 
    data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 23.0)
    data['day_of_year_sin'] = np.sin(2 * np.pi * data.index.dayofyear / 365.0) 
    data['day_of_year_cos'] = np.cos(2 * np.pi * data.index.dayofyear / 365.0)

    return data

def join_data(datasets: List[pd.DataFrame],
              drop_lack: bool = True) -> pd.DataFrame:

    """
        This function takes a list of dataframes and join them into a single dataframe.
        If for example the first dataframe has a column 'A' and the second dataframe has a column 'B',
        the resulting dataframe will have the columns 'A' and 'B'.

        If the drop_lack is set to True, the function will drop the rows that don't aling in the time index.
        for example, if the first dataframe has a row which is far in the past and the second dataframe has a row
        which is far in the future, the function will drop both rows, so that only the 'time intersection' is kept.

        Args:
        -----------
            - datasets (List[pd.DataFrame]): The list of dataframes to join.
            - drop_lack (bool): If True, drop the rows that don't aling in the time index.
        
        Returns:
        -----------
            - data (pd.DataFrame): The resulting dataframe.
    """

    datasets = [deepcopy(dataset) for dataset in datasets] 
    data = pd.concat(datasets, axis = 1)

    if drop_lack:
        data = data.dropna()

    return data


def add_rolling_features(data: pd.DataFrame, 
                         variables: List[str],
                         windows: List[str] = ['10min'],
                         features: List[str] = ['mean', 'std', 'max', 'min', 'median', 'sum']):

    """
        Add rolling features to the dataframe.

        Args:
        -----------
            - data (pd.DataFrame): The dataframe containing the data.
            - variables (List[str]): The variables to add the rolling features.
            - windows (List[str]): The windows to calculate the rolling features.
            - features (List[str]): The features to calculate.

        Returns:
        -----------
            - data (pd.DataFrame): The dataframe with the rolling features.
    """

    data = deepcopy(data)
    
    for variable in variables:
        for window in windows:
            for feature in features:

                rolling = data[variable].rolling(window)

                if feature == 'mean':
                    data[f'{variable}_{window}_{feature}'] = rolling.mean()
                elif feature == 'std':
                    data[f'{variable}_{window}_{feature}'] = rolling.std()
                elif feature == 'max':
                    data[f'{variable}_{window}_{feature}'] = rolling.max()
                elif feature == 'min':
                    data[f'{variable}_{window}_{feature}'] = rolling.min()
                elif feature == 'median':
                    data[f'{variable}_{window}_{feature}'] = rolling.median()
                elif feature == 'sum':
                    data[f'{variable}_{window}_{feature}'] = rolling.sum()
                else:
                    raise ValueError(f'Feature {feature} not implemented.')

    return data