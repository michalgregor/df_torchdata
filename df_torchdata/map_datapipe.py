from torch.utils.data import MapDataPipe
from .common import func_datapipe, GroupByPipe

@func_datapipe('index')
class IndexMapDataPipe(MapDataPipe):
    def __init__(self, datapipe, index):
        super().__init__()

        self.datapipe = datapipe
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index):
        return self.datapipe[self.index[index]]

@func_datapipe('groupby')
class GroupByIterDataPipe(MapDataPipe):
    def __new__(
        cls,
        datapipe: MapDataPipe,
        *args,
        **kwargs
    ):
        return GroupByPipe(datapipe, *args, **kwargs)