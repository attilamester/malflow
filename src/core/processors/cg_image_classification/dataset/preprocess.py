from typing import List, Dict

import pandas as pd
import torch


def list_to_dict_keys(l: List[str]) -> Dict[str, int]:
    return {v: i for i, v in enumerate(sorted(l))}


def df_filter_having_at_column_min_occurencies(df: pd.DataFrame, column: str, min_occurencies: int) -> pd.DataFrame:
    value_counts = df[column].value_counts()
    values_to_keep = value_counts[value_counts >= min_occurencies].index

    return df[df[column].isin(values_to_keep)]


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
