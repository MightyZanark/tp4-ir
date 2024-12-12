import pandas as pd
import pyterrier as pt

from datasets import load_dataset


def start_indexing(collections: pd.DataFrame, index_name: str = "./dataset/index"):
    indexer = pt.IterDictIndexer(index_name, meta={"docno": 32, "title": 256, "text": 4096})
    indexer.index(collections.to_dict(orient="records"))

if __name__ == "__main__":
    if not pt.java.started():
        pt.java.init()
    
    ds = load_dataset("mteb/cqadupstack-programmers", "corpus", cache_dir="./dataset")
    ds = ds["corpus"].to_pandas()
    ds = ds.rename(columns={"_id": "docno"})
    start_indexing(ds)
