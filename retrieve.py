import os
import re
import pickle

import nltk
import pandas as pd
import pyterrier as pt
from datasets import load_dataset
from nltk.corpus import stopwords
from sentence_transformers import CrossEncoder


CUR_DIR = os.path.dirname(os.path.realpath(__file__))
NONALNUM = re.compile("[\W_]+")

def remove_nonalnum(text: str) -> str:
    return NONALNUM.sub('', text)


def get_model(model_loc: str = "./model/bm25.pkl") -> pt.terrier.Retriever:
    model_loc = os.path.join(CUR_DIR, model_loc)
    if not os.path.exists(model_loc):
        raise RuntimeError(f"Model not found at {model_loc}")

    with open(model_loc, "rb") as f:
        return pickle.load(f)


def get_serp(model: pt.terrier.Retriever, query: str, k: int = 30):
    query = remove_nonalnum(query)
    stemmer = pt.TerrierStemmer.porter
    query = [stemmer.stem(word) for word in query.split()]
    
    nltk.download("stopwords")
    stw = stopwords.words("english")
    for i, word in enumerate(stw):
        stw[i] = remove_nonalnum(word)
    stw = set(stw)
    query = ' '.join([word for word in query if word not in stw])

    return (model % k).search(query)

def get_rerank_results(query: str, serp: pd.DataFrame, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", k: int = 30):
    print(f"Loading CrossEncoder model: {model_name}")
    cross_encoder = CrossEncoder(model_name)

    print("Scoring results with the CrossEncoder...")
    query_doc_pairs = [(query, doc) for doc in serp["content"]]
    scores = cross_encoder.predict(query_doc_pairs)

    serp["score"] = scores
    reranked = serp.sort_values(by="score", ascending=False).head(k)

    return reranked

if __name__ == "__main__":
    if not pt.java.started():
        pt.java.init()

    print("Loading BM25 model...")
    model = get_model()

    query = "javascript"
    print(f"Retrieving SERP for query: {query}")
    serp = get_serp(model, query)

    print("Reranking SERP with cross-encoder...")
    reranked = get_rerank_results(query, serp)

    print("Top Reranked Results:")
    print(reranked)