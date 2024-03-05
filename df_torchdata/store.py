import diskcache as dc

class _default_store:
    pass

default_store = _default_store()

def identity(x):
    return x

def default_disk_store(path):
    return dc.Cache(
        path, eviction_policy='none'
    )

def make_store(store, clear_store=False):
    if store is default_store:
        store = {}
    elif isinstance(store, str):
        store = default_disk_store(store)
    
    if clear_store and not store is None:
        store.clear()

    return store