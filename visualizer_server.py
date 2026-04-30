import json
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/architecture")
def architecture():
    model = request.args.get("model", "4b")
    if model not in ("4b", "8b"):
        model = "4b"
    with open(f"architecture_{model}.json") as f:
        return jsonify(json.load(f))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
