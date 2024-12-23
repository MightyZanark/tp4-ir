import os
import re
import pickle

import nltk
import pandas as pd
import pyterrier as pt
from datasets import load_dataset
from nltk.corpus import stopwords


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


if __name__ == "__main__":
    if not pt.java.started():
        pt.java.init()

    print(get_serp(get_model(), "javascript"))    
