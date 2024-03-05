from typing import Any, Callable, Optional
from torch.utils.data import IterDataPipe
from .common import func_datapipe, GroupByPipe

@func_datapipe('reduce')
class ReduceIterDataPipe(IterDataPipe):
    def __new__(
        cls,
        datapipe: IterDataPipe,
        reduce_fn: Callable,
        batch_reduce_fn: Optional[Callable] = None,
        initializer:Any = None
    ):
        if batch_reduce_fn is None:
            it = iter(datapipe)

            if initializer is None:
                ret = next(it)
            else:
                ret = initializer

            for x in it:
                ret = reduce_fn(ret, x)

        else:
            it = iter(datapipe)

            if initializer is None:
                ret = next(it)
                ret = batch_reduce_fn(ret)
            else:
                ret = initializer 

            for x in it:
                ret = reduce_fn(ret, batch_reduce_fn(x))

        return ret
    
@func_datapipe('sum')
class SumIterDataPipe(IterDataPipe):
    def __new__(
        cls,
        datapipe: IterDataPipe,
        batched: bool = False,
    ):
        ret = 0
        
        if batched:
            for batch in datapipe:
                ret += batch.sum(axis=0)
        else:
            for x in datapipe:
                ret += x

        return ret
    
@func_datapipe('mean')
class MeanIterDataPipe(IterDataPipe):
    def __new__(
        cls,
        datapipe: IterDataPipe,
        batched: bool = False,
    ):
        norm = len(datapipe)
        ret = 0
        
        if batched:
            for batch in datapipe:
                batch /= norm
                ret += batch.sum(axis=0)
        else:
            for x in datapipe:
                x /= norm
                ret += x

        return ret

@func_datapipe('groupby')
class GroupByIterDataPipe(IterDataPipe):
    def __new__(
        cls,
        datapipe: IterDataPipe,
        *args,
        **kwargs
    ):
        return GroupByPipe(datapipe, *args, **kwargs)
    