import os

import pandas as pd
import pyterrier as pt

from datasets import load_dataset


CUR_DIR = os.path.dirname(os.path.realpath(__file__))


def start_indexing(
        collections: pd.DataFrame,
        index_name: str = "./dataset/index",
        meta: dict[str, int] = {"docno": 8, "title": 256, "text": (1<<15)}
    ) -> None:
    if not os.path.isabs(index_name):
        index_name = os.path.abspath(os.path.join(CUR_DIR, index_name))
    
    print("Start indexing...")
    indexer = pt.IterDictIndexer(index_name, meta=meta, overwrite=True)
    indexer.index(collections.to_dict(orient="records"))
    print(f"Indexing finished! Saved at {index_name}")


def get_index(index_path: str = "./dataset/index/data.properties") -> pt.IndexRef:
    if not os.path.isabs(index_path):
        index_path = os.path.abspath(os.path.join(CUR_DIR, index_path))

    if not os.path.exists(index_path):
        raise RuntimeError(f"The path '{index_path}' is incorrect or the index have not been made")
    return pt.IndexRef.of(index_path)


def init():
    cache_dir = os.path.abspath(os.path.join(CUR_DIR, "./dataset"))
    ds = load_dataset("mteb/cqadupstack-programmers", "corpus", cache_dir=cache_dir)
    ds = ds["corpus"].to_pandas()
    ds = ds.rename(columns={"_id": "docno"})
    ds['content'] = ds['title'] + ' ' + ds['text']

    meta = {"docno": 8, "title": 256, "text": (1<<15), "content": (1<<15)}
    start_indexing(ds, meta=meta)


if __name__ == "__main__":
    if not pt.java.started():
        pt.java.init()
    
    cache_dir = os.path.abspath(os.path.join(CUR_DIR, "./dataset"))
    ds = load_dataset("mteb/cqadupstack-programmers", "corpus", cache_dir=cache_dir)
    ds = ds["corpus"].to_pandas()
    ds = ds.rename(columns={"_id": "docno"})
    ds['content'] = ds['title'] + ' ' + ds['text']

    meta = {"docno": 8, "title": 256, "text": (1<<15), "content": (1<<15)}
    # start_indexing(ds, "dataset/index_windows", meta=meta)
    start_indexing(ds, meta=meta)
