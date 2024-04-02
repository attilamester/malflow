from typing import List, Dict

import pandas as pd


def list_to_dict_keys(l: List[str]) -> Dict[str, int]:
    return {v: i for i, v in enumerate(sorted(l))}


def df_filter_having_at_column_min_occurencies(df: pd.DataFrame, column: str, min_occurencies: int) -> pd.DataFrame:
    value_counts = df[column].value_counts()
    values_to_keep = value_counts[value_counts >= min_occurencies].index

    return df[df[column].isin(values_to_keep)]
