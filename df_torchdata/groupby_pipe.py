from typing import Any, Callable, Optional
from .common import func_datapipe, GroupByPipe
from abc import abstractmethod

class GroupbyCallback:
    @abstractmethod
    def __call__(self, key, tensor):
        raise NotImplementedError

    def initialize(self, groupby: GroupByPipe):
        pass

    def finalize(self):
        pass

class GroupbyCallbackSum(GroupbyCallback):
    def __call__(self, key, tensor):
        group_sum = self.group_sums.get(key, None)
        
        if group_sum is None:
            self.group_sums[key] = tensor
        else:
            self.group_sums[key] = group_sum + tensor

    def initialize(self, groupby: GroupByPipe):
        self.group_sums = {}

    def finalize(self):
        ret = self.group_sums
        self.group_sums = {}
        return ret

class GroupbyCallbackMean(GroupbyCallback):
    def __call__(self, key, tensor):
        group_sum_count = self.group_sum_counts.get(key, None)
        
        if group_sum_count is None:
            self.group_sum_counts[key] = (tensor, 1)
        else:
            group_sum_count = (
                group_sum_count[0] + tensor,
                group_sum_count[1] + 1
            )
        
            self.group_sum_counts[key] = group_sum_count

    def initialize(self, groupby: GroupByPipe):
        self.group_sum_counts = {}
        
    def finalize(self):
        ret = self.group_sum_counts
        self.group_sum_counts = {}

        ret = {k: v[0] / v[1] for k, v in ret.items()}
        return ret

@func_datapipe('reduce')
class ReduceGroupByPipe(GroupByPipe):
    def __new__(
        cls,
        groupby: GroupByPipe,
        callback: GroupbyCallback
    ):
        callback.initialize(groupby)
        groupby(callback)
        return callback.finalize()

@func_datapipe('sum')
class SumGroupByPipe(GroupByPipe):
    def __new__(
        cls, groupby: GroupByPipe,
    ):
        callback = GroupbyCallbackSum()

        callback.initialize(groupby)
        groupby(callback)
        return callback.finalize()
    
@func_datapipe('mean')
class MeanGroupByPipe(GroupByPipe):
    def __new__(
        cls, groupby: GroupByPipe,
    ):
        callback = GroupbyCallbackMean()
        
        callback.initialize(groupby)
        groupby(callback)
        return callback.finalize()
    