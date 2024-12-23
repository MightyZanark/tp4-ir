import os

import pandas as pd
import pyterrier as pt

from datasets import load_dataset


CUR_DIR = os.path.dirname(os.path.realpath(__file__))

def start_indexing(collections: pd.DataFrame, index_name: str = "./dataset/index"):
    index_name = os.path.join(CUR_DIR, index_name)
    print("Start indexing...")
    indexer = pt.IterDictIndexer(
        index_name, 
        meta={"docno": 8, "title": 256, "text": (1<<15)}, 
        overwrite=True
    )
    indexer.index(collections.to_dict(orient="records"))
    print(f"Indexing finished! Saved at {index_name}")


def get_index(index_path: str = "./dataset/index/data.properties"):
    index_path = os.path.join(CUR_DIR, index_path)
    if not os.path.exists(index_path):
        raise RuntimeError("The index_path is incorrect of the index have not been made")
    return pt.IndexRef.of(index_path)


if __name__ == "__main__":
    if not pt.java.started():
        pt.java.init()
    
    ds = load_dataset("mteb/cqadupstack-programmers", "corpus", cache_dir="./dataset")
    ds = ds["corpus"].to_pandas()
    ds = ds.rename(columns={"_id": "docno"})
    # print(ds["docno"].str.len().max())
    # print(ds["title"].str.len().max())
    # print(ds["text"].str.len().max())
    start_indexing(ds)
    # indexref = get_index()
    # print(pt.IndexFactory.of(indexref).getMetaIndex())
