import os
import pickle
from typing import Any

import pandas as pd
import pyterrier as pt

from index import get_index
from sentence_transformers import CrossEncoder


CUR_DIR = os.path.dirname(os.path.realpath(__file__))

def save_model(model: Any, save_loc: str):
    if not os.path.isabs(save_loc):
        save_loc = os.path.abspath(os.path.join(CUR_DIR, save_loc))
    
    print(f"Saving model {model.__class__.__name__} to {save_loc}")
    with open(save_loc, "wb") as f:
        pickle.dump(model, f)


def get_bm25(
        indexref: pt.IndexRef,
        metadata: list[str] = ["docno", "title", "text", "content"],
        save: bool = True,
        save_loc: str = "./model/bm25.pkl"
    ) -> pt.terrier.Retriever:
    print("Getting BM25 retriever...")
    bm25 = pt.terrier.Retriever(indexref, wmodel="BM25", metadata=metadata)
    if save:
        save_model(bm25, save_loc)
    print("Grabbed BM25 retriever!")
    return bm25


def get_cross_encoder(
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        save: bool = True,
        save_loc: str = "./model/crossenc.pkl"
    ) -> CrossEncoder:
    print(f"Getting CrossEncoder with model {model_name}")
    cross_enc = CrossEncoder(model_name)
    if save:
        save_model(cross_enc, save_loc)
    print("Grabbed CrossEncoder model!")
    return cross_enc


if __name__ == "__main__":
    if not pt.java.started():
        pt.java.init()
    
    # indexref = get_index(r"C:\Users\ASUSTeK\Documents\Fasilkom\SEM 5\TBI\TP\TP4\TEPEE4\tp4-ir\./dataset/index/data_1.properties"
    indexref = get_index()
    bm25 = get_bm25(indexref)
    crossenc = get_cross_encoder()
    print(crossenc)
