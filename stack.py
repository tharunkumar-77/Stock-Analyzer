from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():

    stock = request.form["stock"]
    return render_template(
        "index.html",
        stock=stock
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)