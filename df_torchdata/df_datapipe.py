import sys
assert sys.version_info >= (3, 7), "Python>=3.7 is required: we assume stable order in dicts."

from .common import DataFramePipe, func_datapipe, GroupByPipe, decollate
from .store import make_store, default_disk_store, default_store, identity
from torchdata.datapipes import DataChunk
from typing import Callable, Union, List, Optional, Any, Sized
from torch.utils.data import default_collate, MapDataPipe, IterDataPipe

import pandas as pd
import numpy as np
import torch

def loc_indexer(df, index):
    return df.index.get_indexer(index)

class _NonCached:
    pass

class ToFunc:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    def __call__(self, x):
        return x.to(*self.args, **self.kwargs)

def to_numpy(x):
    return x.numpy()

non_cached = _NonCached()
Tensor = Union[np.ndarray, torch.Tensor]
                             
class TensorDataFramePipe(DataFramePipe):
    def __init__(self,
        tensors: Union[
            Union[Tensor, MapDataPipe],
            List[Union[Tensor, MapDataPipe]]
        ],
        dataframe=None
    ):
        if isinstance(tensors, list):
            self.scalar = False
        else:
            tensors = [tensors]
            self.scalar = True

        if dataframe is None:
            dataframe = pd.DataFrame(index=range(len(tensors[0])), columns=[])
        
        for X in tensors:
            assert len(X) == len(dataframe), "The tensors must have the same length as the dataframe index."

        super().__init__(dataframe)
        self.tensors = tensors

    def __getitem__(self, idx):
        if self.scalar:
            return self.tensors[0][idx]
        else:
            return tuple(X[idx] for X in self.tensors)

@func_datapipe('index')
class IndexDataFramePipe(DataFramePipe):
    def __init__(self, datapipe, index, indexer=None):
        """Indexes a DataFramePipe, i.e. selects a subset of the rows.

        Args:
            datapipe (DataFramePipe): the DataFramePipe to index
            index (Iterable): the indices to select
            indexer (str/Callable, optional): the column to index. Defaults to None.
                * None: a numeric index is assumed (i.e. `df.iloc[index]`);
                * str: the specified column is used in place of the index;
                * Callable: a custom indexer function, which takes index and
                    transforms it into an iloc index; use e.g. loc_indexer
                    to index as `df.loc[index]`.
        """
        df = datapipe.dataframe

        if indexer is None:
            pass
        elif isinstance(indexer, str):
            index = pd.Index(df[indexer]).get_indexer(index)
        else:
            index = indexer(df, index)

        super().__init__(df.iloc[index])

        self.datapipe = datapipe
        self.index = index

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        return self.datapipe[self.index[idx]]

@func_datapipe('iloc')
class ILocDataFramePipe(IndexDataFramePipe):
    def __init__(self, datapipe, index):
        super().__init__(datapipe, index, indexer=None)

@func_datapipe('loc')
class LocDataFramePipe(IndexDataFramePipe):
    def __init__(self, datapipe, index):
        super().__init__(datapipe, index, indexer=loc_indexer)
        
@func_datapipe('pipeindex')
class PipeIndexDataFramePipe(DataFramePipe):
    def __init__(self,
        datapipe: DataFramePipe,
        index_datapipe: DataFramePipe,
        indexer: Optional[str] = None,
        index_column: Optional[str] = None,
    ):
        """Indexes a DataFramePipe, i.e. selects a subset of the rows; the
        index is computed by another DataFramePipe.

        Args:
            datapipe (DataFramePipe): the DataFramePipe to index
            index_datapipe (DataFramePipe): the DataFramePipe to compute the index
            indexer (str/Callable, optional): the column to index. Defaults to None.
                * None: a numeric index is assumed (i.e. `df.iloc[index]`);
                * str: the specified column is used in place of the index;
                * Callable: a custom indexer function; use e.g. loc_indexer
                    to index as `df.loc[index]`.
            index_column (str, optional): the column of index_datapipe to use
            as index. Defaults to None.
                * None: the index of index_datapipe is used;
                * str: the specified column is used as index.
        """
        if index_column is None:
            index = index_datapipe.dataframe.index
        else:
            index = index_datapipe.dataframe[index_column]

        df = datapipe.dataframe

        if indexer is None:
            pass
        elif isinstance(indexer, str):
            index = pd.Index(df[indexer]).get_indexer(index)
        else:
            index = indexer(df, index)

        df = df.iloc[index]
        super().__init__(df)

        self._datapipe = datapipe
        self._index = index

    def __len__(self):
        return len(self._index)
    
    def __getitem__(self, idx):
        return self._datapipe[self._index[idx]]       

@func_datapipe('map')
class MapDataFramePipe(DataFramePipe):
    def __init__(self, datapipe, map_datapipe_fn, map_dataframe_fn=None, include_df_row=False):
        df = datapipe.dataframe
        
        if not map_dataframe_fn is None:
            df = df.apply(map_dataframe_fn, axis=1)

        self.datapipe = datapipe
        self.map_datapipe_fn = map_datapipe_fn if not map_datapipe_fn is None else identity
        self.include_df_row = include_df_row

        super().__init__(df)

    def __getitem__(self, idx):
        if self.include_df_row:
            return self.map_datapipe_fn(self.dataframe.iloc[idx], self.datapipe[idx])
        else:
            return self.map_datapipe_fn(self.datapipe[idx])

@func_datapipe('batch')
class BatchDataFramePipe(DataFramePipe):
    def __init__(self,
        datapipe, batch_size, drop_last=False, wrapper_class=DataChunk,
        key_fn=None, key_col=None
    ):
        df = datapipe.dataframe
        self.datapipe = datapipe
        self.batch_size = batch_size
        self.drop_last = drop_last

        if wrapper_class is None:
            wrapper_class = DataChunk
        else:
            self.wrapper_class = wrapper_class

        dfs = []; keys = []
        for i in range(0, len(df), self.batch_size):
            df_batch = df.iloc[i:i+self.batch_size]
            dfs.append(df_batch)

            if not key_fn is None:
                keys.append(key_fn(df_batch))

        if key_fn is None:
            df = pd.DataFrame({'dataframe': dfs})
        else:
            df = pd.DataFrame({
                'dataframe': dfs,
                key_col if not key_col is None else 'key': keys
            })
        
        super().__init__(df)

    def __getitem__(self, index):
        batch = []
        indices = range(index * self.batch_size, (index + 1) * self.batch_size)

        try:
            for i in indices:
                batch.append(self.datapipe[i])
            return self.wrapper_class(batch)
        except IndexError as e:
            if not self.drop_last and len(batch) > 0:
                return self.wrapper_class(batch)
            else:
                raise IndexError(f"Index {index} is out of bound.") from e

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.datapipe) // self.batch_size
        else:
            return (len(self.datapipe) + self.batch_size - 1) // self.batch_size

@func_datapipe('batch_collate')
class BatchCollateDataFramePipe(BatchDataFramePipe):
    def __init__(self,
        datapipe, batch_size, drop_last=False,
        key_fn=None, key_col=None,
        collate_fn=default_collate
    ):
        super().__init__(
            datapipe, batch_size, drop_last,
            wrapper_class=collate_fn,
            key_fn=key_fn, key_col=key_col
        )

@func_datapipe('decollate')
class BatchCollateDataFramePipe(MapDataFramePipe):
    def __init__(self, datapipe):
        super().__init__(datapipe, decollate)
        
@func_datapipe('cache')
class CacheDataFramePipe(DataFramePipe):
    def __init__(self, datapipe, store=default_store, key_fn=None, clear_store=False):
        """
        A pipe that caches the results of a DataFramePipe so that they do
        not need to be recomputed on a subsequent call.

        Args:
            datapipe (DataFramePipe): the DataFramePipe to cache
            store (dc.Cache, optional): the storage to use. Defaults to None.
                * None: A dictionary is used and the cache is stored in memory.
                * str: A file is used and the cache is stored on disk.
                * Any: A custom storage with a dict-like interface.
            key_fn (callable/str, optional): A callable that maps an entry to
                a cache key or the name of the column to use as key.
                Defaults to None.
                * None: the index of the DataFramePipe is used as key.
                * str: the specified column is used as key.
                * callable: a custom key function.
        """
        super().__init__(datapipe.dataframe)

        if store is default_store:
            self.store = {}
        elif isinstance(store, str):
            self.store = default_disk_store(store)
        else:
            self.store = store

        if clear_store:
            self.store.clear()

        self.key_fn = key_fn
        self.datapipe = datapipe

    def __getitem__(self, idx):
        if self.store is None:
            return self.datapipe[idx]
         
        if self.key_fn is None:
            key = idx
        elif isinstance(self.key_fn, str):
            key = self.dataframe[self.key_fn].iloc[idx]
        else:
            key = self.key_fn(self.dataframe.iloc[idx])

        val = self.store.get(key, non_cached)

        if val is non_cached:
            self.store[key] = val = self.datapipe[idx]

        return val

@func_datapipe('cached_unbatch')
class CachedUnbatchDataFramePipe(DataFramePipe):
    def __init__(self,
        datapipe, store=default_store,
        key_fn=None,
        clear_store=False,
        remove_on_retrieval=False
    ):
        df = pd.concat(datapipe.dataframe["dataframe"].tolist())
        super().__init__(df)

        self.store = make_store(store, clear_store)
        self.key_fn = key_fn
        self.datapipe = datapipe
        self.remove_on_retrieval = remove_on_retrieval
        self.batch_size = self.datapipe.dataframe["dataframe"].iloc[0].shape[0]

    def __getitem__(self, index):
        if self.store is None:
            return self.datapipe[index]

        if self.key_fn is None:
            key = index
        elif isinstance(self.key_fn, str):
            key = self.dataframe[self.key_fn].iloc[index]
        else:
            key = self.key_fn(self.dataframe.iloc[index])

        val = self.store.get(key, non_cached)

        if val is non_cached:
            batch_index = index // self.batch_size
            index_in_batch = index % self.batch_size
            batch = self.datapipe[batch_index]
            batch_start = batch_index * self.batch_size
            indices = range(batch_index * self.batch_size, batch_start + len(batch))
            
            if self.key_fn is None:
                keys = indices
            elif isinstance(self.key_fn, str):
                keys = self.dataframe[self.key_fn].iloc[indices].values
            else:
                keys = [self.key_fn(self.dataframe.iloc[i]) for i in indices]

            for key, val in zip(keys, batch):
                self.store[key] = val

            val = batch[index_in_batch]

            if self.remove_on_retrieval:
                del self.store[keys[index_in_batch]]
        else:
            if self.remove_on_retrieval:
                del self.store[key]

        return val
    
@func_datapipe('discarding_unbatch')
class DiscardingUnbatchDataFramePipe(DataFramePipe):
    def __init__(self, datapipe):
        df = pd.concat(datapipe.dataframe["dataframe"].tolist())
        super().__init__(df)
        self.datapipe = datapipe
        self.batch_size = self.datapipe.dataframe["dataframe"].iloc[0].shape[0]

    def __getitem__(self, index):
        batch_index = index // self.batch_size
        index_in_batch = index % self.batch_size
        batch = self.datapipe[batch_index]
        val = batch[index_in_batch]
        return val

@func_datapipe('map_tensors')
class MapTensorsDataFramePipe(DataFramePipe):
    def __init__(self, datapipe, map_datapipe_fn, map_dataframe_fn=None, include_df_row=False):
        df = datapipe.dataframe
        
        if not map_dataframe_fn is None:
            df = df.apply(map_dataframe_fn, axis=1)

        self.datapipe = datapipe
        self.map_datapipe_fn = map_datapipe_fn if not map_datapipe_fn is None else identity
        self.include_df_row = include_df_row

        super().__init__(df)

    def _map_tensor(self, row, x):
        if isinstance(x, torch.Tensor):
            if self.include_df_row:
                return self.map_datapipe_fn(row, x)
            else:
                return self.map_datapipe_fn(x)
        elif isinstance(x, list) or isinstance(x, tuple):
            return type(x)(self._map_tensor(row, y) for y in x)
        else:
            return x

    def __getitem__(self, idx):
        return self._map_tensor(
            self.dataframe.iloc[idx],
            self.datapipe[idx]
        )

@func_datapipe('to')
class ToDataFramePipe(MapTensorsDataFramePipe):
    def __init__(self, datapipe, *args, **kwargs):
        super().__init__(datapipe, ToFunc(*args, **kwargs))

@func_datapipe('cpu')
class CpuDataFramePipe(MapTensorsDataFramePipe):
    def __init__(self, datapipe):
        super().__init__(datapipe, ToFunc('cpu'))

@func_datapipe('cuda')
class CudaDataFramePipe(MapTensorsDataFramePipe):
    def __init__(self, datapipe, device=None):
        super().__init__(datapipe, ToFunc(device))

@func_datapipe('numpy')
class NumpyDataFramePipe(MapTensorsDataFramePipe):
    def __init__(self, datapipe):
        super().__init__(datapipe, to_numpy)

@func_datapipe('zip')
class ZipDataFramePipe(DataFramePipe):
    def __init__(self, datapipe: DataFramePipe, *zip_pipes: Union[MapDataPipe, pd.DataFrame]):
        super().__init__(datapipe.dataframe)
        self.datapipe = datapipe
        self.zip_pipes = zip_pipes

        for i, zip_pipe in enumerate(zip_pipes):
            assert len(zip_pipe) == len(datapipe), "All pipes must have the same length."
            if isinstance(zip_pipe, pd.DataFrame):
                self.zip_pipes[i] = DataFramePipe(zip_pipe)

    def __getitem__(self, idx):
        return (self.datapipe[idx],) + tuple(
            zip_pipe[idx] for zip_pipe in self.zip_pipes
        )
    
@func_datapipe('to_tensor_dataset')
class ToTensorDatasetDataFramePipe(DataFramePipe):
    def __new__(
        cls,
        datapipe: IterDataPipe,
        collate_fn: Optional[Callable] = default_collate,
    ):
        return datapipe.batch_collate(
            batch_size=len(datapipe),
            collate_fn=collate_fn
        )[0]

@func_datapipe('col')
class ColumnDataFramePipe(DataFramePipe):
    def __init__(self, datapipe: Union[DataFramePipe, pd.DataFrame], column):
        if isinstance(datapipe, pd.DataFrame):
            df = datapipe
        elif isinstance(datapipe, DataFramePipe):
            df = datapipe.dataframe
        else:
            raise TypeError(f"Expected DataFramePipe or pd.DataFrame, got {type(datapipe)}")

        super().__init__(df)    
        self.column = column

    def __getitem__(self, idx):
        return self.dataframe[self.column].iloc[idx]

@func_datapipe('groupby')
class GroupByDataFramePipe(DataFramePipe):
    def __new__(
        cls,
        datapipe: DataFramePipe,
        *args,
        **kwargs
    ):
        return GroupByPipe(datapipe, *args, **kwargs)

@func_datapipe('fewshot_split')
class FewshotSplitDataFramePipe(DataFramePipe):
    def __new__(
        cls,
        datapipe: DataFramePipe,
        split_cols: Union[str, List[str]],
        num_in_support: int = 5,
        shuffle: bool = True,
        random_state: Union[np.random.RandomState, int] = None
    ):
        # we rely on .index later to be an iloc compatible index, so we
        # reset the index here; note that we are not using the inplace
        # version of reset_index, so df is kept intact
        df = datapipe.dataframe.reset_index(drop=True)

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
        
        return [
            IndexDataFramePipe(datapipe, df_support_index),
            IndexDataFramePipe(datapipe, df_query_index),
        ]
    
@func_datapipe('shuffle')
class IndexDataFramePipe(DataFramePipe):
    def __init__(self,
        datapipe,
        random_state: Union[np.random.RandomState, int] = None
    ):
        df = datapipe.dataframe

        if random_state is None:
            random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)

        index = np.arange(df.shape[0])
        random_state.shuffle(index)
        df = df.iloc[index]

        super().__init__(df)
        self.index = index
        self.datapipe = datapipe

    def __getitem__(self, idx):
        return self.datapipe[self.index[idx]]
    
@func_datapipe('concat')
class ConcaterDataFramePipe(DataFramePipe):
    def __init__(self, *datapipes: DataFramePipe):
        df = pd.concat([dp.dataframe for dp in datapipes], ignore_index=True)
        super().__init__(df)

        if len(datapipes) == 0:
            raise ValueError("Expected at least one DataPipe, but got nothing")
        if not all(isinstance(dp, DataFramePipe) for dp in datapipes):
            raise TypeError("Expected all inputs to be `MapDataPipe`")
        if not all(isinstance(dp, Sized) for dp in datapipes):
            raise TypeError("Expected all inputs to be `Sized`")
        self.datapipes = datapipes  # type: ignore[assignment]

    def __getitem__(self, index):
        offset = 0
        for dp in self.datapipes:
            if index - offset < len(dp):
                return dp[index - offset]
            else:
                offset += len(dp)
        raise IndexError(f"Index {index} is out of range.")

@func_datapipe('column_as_target')
class TargetColDataFramePipe(DataFramePipe):
    def __init__(self, datapipe: DataFramePipe, target_column: str):
        """A special `DataFramePipe` that designates a column of the dataframe
        as the target column and returns an `(input, target)` tuple, where
        `input` is the output of the wrapped `DataFramePipe` and `target` is
        the value of the designated column.
        
        Args:
            *datapipe: The source `DataFramePipe`.
            target_column: The name of the target column.
        """
        super().__init__(datapipe.dataframe)
        self.datapipe = datapipe
        self.target_column = target_column

    def __getitem__(self, index):
        input = self.datapipe[index]
        return input, self.dataframe[self.target_column].iloc[index]