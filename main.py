from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "<h1>Hi</h1>"

if __name__ == "__main__":
    app.serve("0.0.0.0", 8080)