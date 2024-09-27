from typing import Dict, Callable, Union, Tuple
from torch.utils.data import MapDataPipe, IterDataPipe, functional_datapipe
import functools

def _register_datapipe_as_func_cls(decorator, base_cls, name, cls):
    base_cls.register_datapipe_as_function(name, cls)

def _register_datatype_fallback(decorator, base_cls, name, cls):
    functional_datapipe.__call__(decorator, cls)

class DataFramePipe(MapDataPipe):
    functions: Dict[str, Callable] = {}

    def __init__(self, dataframe):
        super().__init__()
        self.dataframe = dataframe

    def __getattr__(self, attribute_name):
        if attribute_name in DataFramePipe.functions:
            f = DataFramePipe.functions[attribute_name]
        elif attribute_name in MapDataPipe.functions:
            f = MapDataPipe.functions[attribute_name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attribute_name}")
        
        function = functools.partial(f, self)
        return functools.update_wrapper(wrapper=function, wrapped=f, assigned=("__doc__",))
        
    def __dir__(self):
        # for auto-completion in a REPL (e.g. Jupyter notebook)
        return (
            list(super().__dir__()) +
            list(DataFramePipe.functions.keys()) +
            list(MapDataPipe.functions.keys())
        )

    def __getitem__(self, idx):
        return self.dataframe.iloc[idx]
        
    def __len__(self):
        return len(self.dataframe)

class GroupByPipe:
    functions: Dict[str, Callable] = {}
    register_datapipe_as_function = classmethod(DataFramePipe.register_datapipe_as_function.__func__)

    def __init__(self, datapipe, key_fn=None, value_fn=None):
        """A special datapipe that iterates over the input datapipe, and reports
        each sample and its group key to a specified callback function.

        Usually used in conjunction with reduction operations such as .mean()
        and .sum() to perform group-wise reductions and return results in the
        form of a dictionary.
        
        Args:
            datapipe: A datapipe that yields samples.

            key_fn: A function that extracts the group key from each sample.
                The following types are supported:
                    * When None, the sample is assumed to be a tuple of
                      (input tensor 1, ..., label), and the last element is
                      used as the group key.
                    * When an integer, the sample is expected to be a tuple
                      and the element at the specified index is used as the
                      group key.
                    * When a slice, the sample is expected to be a tuple and
                      the elements in the specified range are used as the
                      group key.
                    * When a string or a list of strings, the sample is
                      expected to indexable by the string/list of strings.
                    * When a callable, the sample is passed to it and the
                      returned value is used.

            value_fn: A function that extracts the value to operate on from
                each sample. The following types are supported:
                    * When None, the sample is assumed to be a tuple of
                      (input tensor 1, ..., label), and all elements
                      but the last are used as the value; if there is only one
                      such element, it is returned as an item, not as a tuple.
                    * When an integer, the sample is expected to be a tuple
                      and the element at the specified index is used as the
                      group key.
                    * When a slice, the sample is expected to be a tuple and
                      the elements in the specified range are used as the
                      group key.
                    * When a string or a list of strings, the sample is
                      expected to indexable by the string/list of strings.
                    * When a callable, the sample is passed to it and the
                      returned value is used.
        """
        super().__init__()
        self.datapipe = datapipe
        self.key_fn = key_fn
        self.value_fn = value_fn

    def __getattr__(self, attribute_name):
        if attribute_name in GroupByPipe.functions:
            f = GroupByPipe.functions[attribute_name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attribute_name}")
        
        function = functools.partial(f, self)
        return functools.update_wrapper(wrapper=function, wrapped=f, assigned=("__doc__",))
        
    def __dir__(self):
        # for auto-completion in a REPL (e.g. Jupyter notebook)
        return (
            list(super().__dir__()) +
            list(GroupByPipe.functions.keys())
        )
    
    def _make_extraction_func(self, extract_func):
        if (
            isinstance(extract_func, str) or
            isinstance(extract_func, int) or
            isinstance(extract_func, list) or
            isinstance(extract_func, slice)
        ):
            return lambda x: x[extract_func]
        elif isinstance(extract_func, Callable):
            return extract_func
        else:
            raise TypeError(f"Unknown type of extract_func argument '{extract_func}'")

    def __call__(self, callback_func):
        """Args:
            callback_func: a function that takes two arguments,
                a group key and a tensor;
        """

        if self.key_fn is None:
            key_fn = lambda x: x[-1]
        else:
            key_fn = self._make_extraction_func(self.key_fn)

        if self.value_fn is None:
            value_fn = lambda x: x[:-1] if len(x) > 2 else x[0]
        else:
            value_fn = self._make_extraction_func(self.value_fn)

        for sample in self.datapipe:
            key = key_fn(sample)
            value = value_fn(sample)
            callback_func(key, value)
 
_func_datapipe_cls_registry = {
    DataFramePipe: _register_datapipe_as_func_cls,
    GroupByPipe: _register_datapipe_as_func_cls,
    IterDataPipe: _register_datatype_fallback,
    MapDataPipe: _register_datatype_fallback,
}

class func_datapipe(functional_datapipe):
    def __call__(self, cls):
        for base_cls, reg_func in _func_datapipe_cls_registry.items():
            if issubclass(cls, base_cls):
                reg_func(self, base_cls, self.name, cls)
                break

        return cls

#---------------------
# Decollate
#---------------------

def convert_dict_to_list(dict_data):
    keys = dict_data.keys()
    values = dict_data.values()
    
    # Use zip and list comprehension to create a list of dictionaries
    list_data = [dict(zip(keys, item)) for item in zip(*values)]
    
    return list_data

def decollate(batch: Union[Tuple, dict]):
    if isinstance(batch, dict):
        return convert_dict_to_list(batch)
    elif isinstance(batch, tuple):
        return list(zip(*batch))
    else:
        raise ValueError(f"Unsupported batch type: {type(batch)}")
