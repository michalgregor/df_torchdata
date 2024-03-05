import pandas as pd
import numpy as np
from typing import List, Union

def fewshot_split(
    df: pd.DataFrame,
    split_cols: Union[str, List[str]],
    num_in_support: int = 5,
    shuffle: bool = True,
    random_state: Union[np.random.RandomState, int] = None,
    return_index: bool = False,
):
    """
    Split a dataframe into support and query sets for few-shot learning.

    Args:
        df: DataFrame to split.
        split_cols: Column(s) to split the DataFrame on.
        num_in_support: Number of samples to put in the support set.
        shuffle: Whether to shuffle the support and query sets.
        random_state: Random state to use for shuffling.
        return_index: Whether to return the indices of the support and query sets.

    Returns:
        If return_index is False, returns a tuple of two DataFrames, the first
        containing the support set and the second containing the query set.

        If return_index is True, returns a tuple of iloc indices, the first
        for the support set and the second for the query set.
    """

    # we rely on .index to be an iloc compatible index later here, so we
    # reset the index here; note that we are not using the inplace
    # version of reset_index, so df is kept intact
    df_orig = df
    df = df.reset_index(drop=True)

    if shuffle:
        if random_state is None:
            random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)

        index = np.arange(df.shape[0])
        random_state.shuffle(index)
        df = df.iloc[index]

    groupby = df.groupby(split_cols, as_index=False)
    df_support_index = groupby.nth(slice(0, num_in_support)).index
    df_query_index = groupby.nth(slice(num_in_support, None)).index
    
    if return_index:
        return df_support_index, df_query_index
    else:
        return df_orig.iloc[df_support_index], df_orig.iloc[df_query_index]



