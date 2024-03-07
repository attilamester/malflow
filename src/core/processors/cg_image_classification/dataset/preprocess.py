from typing import List, Tuple, Dict

import pandas as pd
import torch


def list_to_dict_keys(l: List[str]) -> Dict[str, int]:
    return {v: i for i, v in enumerate(sorted(l))}


def filter_ds_having_at_column_min_occurencies(df: pd.DataFrame, column: str, min_occurencies: int) \
        -> Tuple[pd.DataFrame, pd.Series, Dict[str, int]]:
    value_counts = df[column].value_counts()
    values_to_keep = value_counts[value_counts >= min_occurencies].index

    dataset_filtered = df[df[column].isin(values_to_keep)]
    values_filtered = dataset_filtered[column]

    return dataset_filtered, values_filtered, list_to_dict_keys(list(values_filtered.unique()))


def preprocess(x: torch.tensor, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y


def undo_preprocess(x: torch.tensor, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y
