from flask import Flask, render_template, request
import yfinance as yf

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():

    stock = request.form["stock"]

    try:
        ticker=yf.Ticker(stock)
        info=ticker.info

        name=info.get("longName","Not Available")
        symbol=info.get("symbol",stock)
        price=info.get("currentPrice","Not Available")
        asset_type=info.get("quoteType","Note available")


        return render_template("index.html",stock=stock,name=name,symbol=symbol,price=price,asset_type=asset_type)
    except Exception:
        return render_template("index.html",error="Invalid symbol or no Data")
    
    



if __name__ == "__main__":
    app.run(debug=True, port=5002)