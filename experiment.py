import os
import pyterrier as pt

from retrieve import get_model, remove_nonalnum
from index import get_index
from pyterrier.measures import *
from datasets import load_dataset

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
crossmodel = get_model("./model/crossenc.pkl")
CUT_OFF = 30

def _crossencoder_apply(dataframe):
    queries = dataframe['query']
    contents = dataframe['content']

    pairs = list(zip(queries, contents))

    scores = crossmodel.predict(pairs)
    return scores


if __name__ == "__main__":
    if not pt.java.started():
        pt.java.init()
    
    cache_dir = os.path.abspath(os.path.join(CUR_DIR, "./dataset"))
    ds = load_dataset("mteb/cqadupstack-programmers", "default", cache_dir=cache_dir)
    ds = ds["test"].to_pandas()
    ds = ds.rename(columns={"query-id": "qid", "corpus-id": "docno", "score": "label"})
    ds["label"] = ds["label"].astype(int)

    ds_queries = load_dataset("mteb/cqadupstack-programmers", "queries", cache_dir=cache_dir)
    ds_queries = ds_queries["queries"].to_pandas()
    ds_queries = ds_queries.rename(columns={"_id": "qid", "text": "query"})
    ds_queries["query"] = ds_queries["query"].apply(remove_nonalnum)

    # BM25 model
    indexref = get_index()
    bm25_model = get_model()
    cross_encT = pt.apply.doc_score(_crossencoder_apply, batch_size=128)
    bm25_crossencoder = (bm25_model % CUT_OFF) >> pt.text.get_text(get_index(
        r"C:\Users\ASUSTeK\Documents\Fasilkom\SEM 5\TBI\TP\TP4\TEPE4LAGI\tp4-ir\dataset\index_windows\data.properties"),
        ["content"]) >> cross_encT
    
    eval_results = pt.Experiment([bm25_model, bm25_crossencoder], \
                            ds_queries, \
                            ds, \
                            eval_metrics=[P@5, "map", nDCG@5], \
                            names=["BM25", "BM25 >> Bi-encoder"], \
                            baseline=0) # statistical significance test

