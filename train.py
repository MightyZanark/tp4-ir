import os
import pickle

import pandas as pd
import pyterrier as pt
from datasets import load_dataset

from index import get_index
from sentence_transformers import CrossEncoder


CUR_DIR = os.path.dirname(os.path.realpath(__file__))

def get_bm25(indexref: pt.IndexRef, save: bool = True, save_loc: str = "./model/bm25.pkl"):
    print("Getting BM25 retriever...")
    save_loc = os.path.join(CUR_DIR, save_loc)
    bm25 = pt.terrier.Retriever(indexref, wmodel="BM25", metadata=["docno", "content"])
    if save:
        print(f"Saving BM25 to {save_loc}")
        with open(save_loc, "wb") as f:
            pickle.dump(bm25, f)
    print("Grabbed BM25 retriever!")
    return bm25

def rerank_cross_encoder(query: str, candidates: pd.DataFrame, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = 30):
    print(f"Loading CrossEncoder model: {model_name}")
    cross_encoder = CrossEncoder(model_name)

    query_doc_pairs = [(query, doc) for doc in candidates["content"]]

    print("Scoring documents with the CrossEncoder...")
    scores = cross_encoder.predict(query_doc_pairs)

    candidates["score"] = scores

    reranked_candidates = candidates.sort_values(by="score", ascending=False).head(top_k)

    return reranked_candidates[["content", "score"]].values.tolist()


if __name__ == "__main__":
    if not pt.java.started():
        pt.java.init()
    
    indexref = get_index(r"C:\Users\ASUSTeK\Documents\Fasilkom\SEM 5\TBI\TP\TP4\TEPEE4\tp4-ir\./dataset/index/data_1.properties")
    get_bm25(indexref)
