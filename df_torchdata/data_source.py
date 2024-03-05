from typing import List, Union
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import MapDataPipe
from .df_datapipe import Tensor, TensorDataFramePipe, ColumnDataFramePipe

import pandas as pd

class DataSourceTransform:
    def __call__(self, df, data_source=None):
        raise NotImplementedError

class ColumnTransformer(DataSourceTransform):
    def __init__(self, col_name, transformer):
        self.col_name = col_name
        self.transformer = transformer

    def fit(self, df):
        if isinstance(self.col_name, list):
            self.transformer.fit(df[self.col_name])
        else:
            self.transformer.fit(df[[self.col_name]])

    def __call__(self, df, data_source=None):
        if isinstance(self.col_name, list):
            return self.transformer.transform(df[self.col_name])
        else:
            ret = self.transformer.transform(df[[self.col_name]])
            assert ret.shape[1] == 1
            return ret[:, 0]

class ColumnInverseTransformer(DataSourceTransform):
    def __init__(self, col_name, transformer):
        self.col_name = col_name

        if isinstance(transformer, ColumnTransformer):
            self.transformer = transformer.transformer
        else:
            self.transformer = transformer

    def fit(self, df):
        pass

    def __call__(self, df, data_source=None):
        if isinstance(self.col_name, list):
            return self.transformer.inverse_transform(df[self.col_name])
        else:
            return self.transformer.inverse_transform(df[[self.col_name]])[:,0]

def make_ordinal_encoder(col_name):
    return ColumnTransformer(col_name, OrdinalEncoder())

def make_inv_ordinal_encoder(col_name, ordinal_encoder):
    return ColumnInverseTransformer(col_name, ordinal_encoder)

class DataSourceBase:
    def _make_pipe(self,
        tensor: Union[Tensor, pd.Series, MapDataPipe, str, DataSourceTransform],
        dataframe: pd.DataFrame
    ):
        if isinstance(tensor, Tensor) or isinstance(tensor, pd.Series):
            return TensorDataFramePipe(tensor)
        elif isinstance(tensor, MapDataPipe):
            return tensor
        elif isinstance(tensor, str):
            return ColumnDataFramePipe(dataframe, tensor)
        elif isinstance(tensor, DataSourceTransform):
            return TensorDataFramePipe(tensor(dataframe, data_source=self))
        else:
            raise TypeError(f"Expected Tensor, Dataset, str or DataSourceTransform, got {type(tensor)}")
    
    def to_supervised_dataset(self,
        *tensors: Union[
            Union[Tensor, pd.Series, MapDataPipe, str, DataSourceTransform],
            List[Union[Tensor, pd.Series, MapDataPipe, str, DataSourceTransform]]
        ],
        dataframe = None
    ):
        if dataframe is None:
            dataframe = self.full_dataframe

        items = []

        for t in tensors:
            if isinstance(t, list):
                items.append(
                    TensorDataFramePipe([self._make_pipe(tt, dataframe) for tt in t])
                )
            else:
                items.append(self._make_pipe(t, dataframe))

        return TensorDataFramePipe(items, dataframe=dataframe)
        
    def to_tensor_dataset(self, datapipe):
        return datapipe.to_tensor_dataset()

    def from_tensor_dataset(self, *tensors):
        return TensorDataFramePipe(*tensors)
