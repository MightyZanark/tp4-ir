import os

import pyterrier as pt
from flask import Flask, render_template, request

import index
import train
from retrieve import get_model, get_serp


app = Flask(__name__)


@app.route("/")
def home():
    query = request.args.get("query")
    if query is None:
        return render_template("index.html")
    
    serp = get_serp(get_model(), query)
    serp = serp[["title", "text"]].to_dict(orient="records")
    return render_template("index.html", result=serp)


if __name__ == "__main__":
    if not pt.java.started():
        pt.java.init()
    
    index.init()
    train.init()

    debug = not os.getenv("PRODUCTION", False)
    port = 8080 if debug else 80
    app.run("0.0.0.0", port, debug=debug)
