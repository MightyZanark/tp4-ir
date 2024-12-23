import os
import re
import pickle

import nltk
import pandas as pd
import pyterrier as pt
from nltk.corpus import stopwords
from sentence_transformers import CrossEncoder


CUR_DIR = os.path.dirname(os.path.realpath(__file__))
NONALNUM = re.compile("[\W_ ]+")


def remove_nonalnum(text: str) -> str:
    return NONALNUM.sub(' ', text)


def get_model(model_loc: str = "./model/bm25.pkl") -> pt.terrier.Retriever | CrossEncoder:
    if not os.path.isabs(model_loc):
        model_loc = os.path.abspath(os.path.join(CUR_DIR, model_loc))

    if not os.path.exists(model_loc):
        raise RuntimeError(f"Model not found at {model_loc}")

    with open(model_loc, "rb") as f:
        return pickle.load(f)


def get_serp(model: pt.terrier.Retriever, query: str, k: int = 30, rerank: bool = True):
    query = remove_nonalnum(query)
    stemmer = pt.TerrierStemmer.porter
    query = [stemmer.stem(word) for word in query.split()]
    
    nltk.download("stopwords", quiet=True)
    stw = stopwords.words("english")
    for i, word in enumerate(stw):
        stw[i] = remove_nonalnum(word)
    stw = set(stw)
    query = ' '.join([word for word in query if word not in stw])

    serp = (model % k).search(query)
    if not rerank:
        return serp

    return rerank_serp(query, serp)


def rerank_serp(query: str, serp: pd.DataFrame, reranker_loc: str = "./model/crossenc.pkl"):
    print(f"Loading CrossEncoder model: {reranker_loc}")
    cross_encoder = get_model(reranker_loc)

    print("Scoring results with the CrossEncoder...")
    query_doc_pairs = [(query, doc) for doc in serp["content"]]
    scores = cross_encoder.predict(query_doc_pairs)

    serp["score"] = scores
    reranked = serp.sort_values(by="score", ascending=False)

    return reranked


if __name__ == "__main__":
    if not pt.java.started():
        pt.java.init()

    print("Loading BM25 model...")
    model = get_model()

    query = "is python bad?"
    print(f"Retrieving SERP for query: {query}")
    serp = get_serp(model, query)
    serp = serp[["title", "text", "score"]].to_dict(orient="records")

    # print("Reranking SERP with cross-encoder...")
    # reranked = rerank_serp(query, serp)

    print("Top Reranked Results:")
    print(serp)