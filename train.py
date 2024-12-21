import pickle

import pandas as pd
import pyterrier as pt
from datasets import load_dataset

from index import get_index


def get_bm25(indexref: pt.IndexRef, k: int = 30, save: bool = True, save_loc: str = "./model/bm25.pkl"):
    bm25 = pt.terrier.Retriever(indexref, wmodel="BM25", metadata=["title", "text"]) % k
    if save:
        with open(save_loc, "wb") as f:
            pickle.dump(bm25, f)
    return bm25


if __name__ == "__main__":
    if not pt.java.started():
        pt.java.init()
    
    indexref = get_index()
    get_bm25(indexref)
