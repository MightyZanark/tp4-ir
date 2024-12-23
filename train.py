import os
import pickle

import pandas as pd
import pyterrier as pt
from datasets import load_dataset

from index import get_index


CUR_DIR = os.path.dirname(os.path.realpath(__file__))

def get_bm25(indexref: pt.IndexRef, save: bool = True, save_loc: str = "./model/bm25.pkl"):
    print("Getting BM25 retriever...")
    save_loc = os.path.join(CUR_DIR, save_loc)
    bm25 = pt.terrier.Retriever(indexref, wmodel="BM25", metadata=["title", "text"])
    if save:
        print(f"Saving BM25 to {save_loc}")
        with open(save_loc, "wb") as f:
            pickle.dump(bm25, f)
    print("Grabbed BM25 retriever!")
    return bm25


if __name__ == "__main__":
    if not pt.java.started():
        pt.java.init()
    
    indexref = get_index()
    get_bm25(indexref)
