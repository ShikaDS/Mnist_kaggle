import os
from random import shuffle
from typing import Tuple
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def get_data_from_csv(path_to_data_dir: str) -> Tuple[np.ndarray, ...]:
    """
    function for reading data from csv format

    Args:
        path_to_data_dir (str): directory with data from kaggle

    Returns:
        Tuple[np.ndarray, ...]: tuple of training and test data
    """
    train_file = os.path.join(path_to_data_dir, "train.csv")
    test_file = os.path.join(path_to_data_dir, "test.csv")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    train_labels = train_df["label"]
    train_labels = train_labels.values
    train_data = train_df.drop("label", axis=1)
    train_data = train_data.values
    test_data = test_df.values
    return train_data, train_labels, test_data


def scale_data(data: np.ndarray) -> np.ndarray:
    """
    Scale data to range [-1; +1]

    Args:
        data (np.ndarray): array with numbers from 0 to 255

    Returns:
        np.ndarray: array with numbers from -1 to 0
    """
    data = 2.0 * data / 255.0 - 1.0
    return data


def convert_data_for_model(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    conv_mod: bool = False,
) -> Tuple[np.ndarray, ...]:
    """
    A function that calls the data scaling method
    and the method for decomposing the sample
    into training and validation,
    in the case when conv_mod = True,
    the data is also reshaped into the desired form

    Args:
        train_data (np.ndarray): training data
        train_labels (np.ndarray): training data answers
        test_data (np.ndarray): test data
        conv_mod (bool, optional): switch for reshaping from (748,) -> (28, 28,1). Defaults to False.

    Returns:
        Tuple[np.ndarray, ...]: data tuple for full training pipeline
    """
    scale_train_data = scale_data(train_data)
    scale_test_data = scale_data(test_data)
    categorical_train_labels = to_categorical(train_labels)
    if conv_mod:
        scale_train_data = scale_train_data.reshape(
            scale_train_data.shape[0], 28, 28, 1
        )
        scale_test_data = scale_test_data.reshape(scale_test_data.shape[0], 28, 28, 1)
    X_train, X_val, y_train, y_val = train_test_split(
        scale_train_data, categorical_train_labels, test_size=0.3, shuffle=True
    )
    return X_train, X_val, y_train, y_val, scale_test_data


def get_data(path_to_data_dir: str, conv_mod: bool = False) -> Tuple[np.ndarray, ...]:
    """
    Function to get all data

    Args:
        path_to_data_dir (str): directory with data from kaggle
        conv_mod (bool, optional): switch for reshaping from (748,) -> (28, 28,1). Defaults to False.

    Returns:
        Tuple[np.ndarray, ...]: data tuple for full training pipeline
    """
    train_data, train_labels, test_data = get_data_from_csv(path_to_data_dir)
    X_train, X_val, y_train, y_val, scale_test_data = convert_data_for_model(
        train_data, train_labels, test_data, conv_mod
    )
    return X_train, X_val, y_train, y_val, scale_test_data
