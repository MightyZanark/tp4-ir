from flask import Flask, render_template
import pyterrier as pt

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    if not pt.java.started():
        pt.java.init()

    app.run("0.0.0.0", 8080)
