from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():

    stock_df=pd.read_csv("stocks.csv")
    stock = request.form["stock"].strip()
    search_item=stock.lower()
    matched=stock_df[stock_df["name"].str.lower()==search_item]

    if not matched.empty:
        stock = matched.iloc[0]["ticker"]

    try:
        ticker=yf.Ticker(stock)
        info=ticker.info

        name = (
            info.get("longName")
            or info.get("shortName")
            or "Not Available"
        )

        symbol = (
            info.get("symbol")
            or stock
        )

        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or "Not Available")

        asset_type = (
                info.get("quoteType")
                or "Not Available"
                )
        explanation=info.get("longBusinessSummary","No Data Available")
        explanation=explanation[:500]+"..."

    
        return render_template("index.html",stock=stock,name=name,symbol=symbol,price=price,asset_type=asset_type,
                               explanation=explanation)

    
    
    except Exception:
        return render_template("index.html",error="Invalid symbol or no Data")
    
    




if __name__ == "__main__":
    app.run(debug=True, port=5002)