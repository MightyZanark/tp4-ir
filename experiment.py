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

    df_qrels = load_dataset("mteb/cqadupstack-programmers", "default", cache_dir=cache_dir)
    df_qrels = df_qrels["test"].to_pandas()
    df_qrels = df_qrels.rename(columns={"query-id": "qid", "corpus-id": "docno", "score": "label"})
    df_qrels["label"] = df_qrels["label"].astype(int)

    df_queries = load_dataset("mteb/cqadupstack-programmers", "queries", cache_dir=cache_dir)
    df_queries = df_queries["queries"].to_pandas()
    df_queries = df_queries.rename(columns={"_id": "qid", "text": "query"})
    df_queries["query"] = df_queries["query"].apply(remove_nonalnum)
    
    test_queries = df_queries[:50].copy()
    test_qrels = df_qrels[df_qrels["qid"].isin(test_queries["qid"])]

    # BM25 model
    indexref = get_index()
    bm25_model = get_model()
    cross_encT = pt.apply.doc_score(_crossencoder_apply, batch_size=4096)
    # indexref = get_index(r"C:\Users\ASUSTeK\Documents\Fasilkom\SEM 5\TBI\TP\TP4\TEPE4LAGI\tp4-ir\dataset\index_windows\data.properties")
    indexref = get_index()
    bm25_crossencoder = (bm25_model % CUT_OFF) >> pt.text.get_text(indexref, ["content"]) >> cross_encT
    
    eval_results = pt.Experiment([bm25_model, bm25_crossencoder], \
                            test_queries, \
                            test_qrels, \
                            eval_metrics=[P@5, "map", nDCG@5], \
                            names=["BM25", "BM25 >> Cross-encoder"], \
                            baseline=0) # statistical significance test
    print(eval_results)

