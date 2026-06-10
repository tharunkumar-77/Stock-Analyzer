from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd

app = Flask(__name__)


def calculate_return(history):

    if history.empty:
        return "N/A"

    start_price = history["Close"].iloc[0]
    end_price = history["Close"].iloc[-1]

    return ((end_price - start_price) / start_price) * 100


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():

    stock_df = pd.read_csv("stocks.csv")

    stock = request.form["stock"].strip()

    search_item = stock.lower()

    matched = stock_df[
        stock_df["name"].str.lower() == search_item
    ]

    if not matched.empty:
        stock = matched.iloc[0]["ticker"]

    try:

        ticker = yf.Ticker(stock)

        info = ticker.info

        history_1y = ticker.history(period="1y")
        history_3y = ticker.history(period="3y")
        history_5y = ticker.history(period="5y")

        return_1y = round(calculate_return(history_1y), 2)
        return_3y = round(calculate_return(history_3y), 2)
        return_5y = round(calculate_return(history_5y), 2)

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
            or "Not Available"
        )

        asset_type = (
            info.get("quoteType")
            or "Not Available"
        )

        explanation = info.get(
            "longBusinessSummary",
            "No Data Available"
        )

        if explanation != "No Data Available":
            explanation = explanation[:500] + "..."

        return render_template(
            "index.html",
            stock=stock,
            name=name,
            symbol=symbol,
            price=price,
            asset_type=asset_type,
            explanation=explanation,
            return_1y=return_1y,
            return_3y=return_3y,
            return_5y=return_5y
        )

    except Exception:
        return render_template(
            "index.html",
            error="Invalid symbol or no Data"
        )
    
@app.route("/calculator", methods=["POST"])
def calculator():

    stock_df = pd.read_csv("stocks.csv")

    stock = request.form["stock"].strip()

    amount = float(request.form["amount"])

    date = request.form["date"]

    matched = stock_df[
        stock_df["name"].str.lower() == stock.lower()
    ]

    if not matched.empty:
        stock = matched.iloc[0]["ticker"]

    try:

        ticker = yf.Ticker(stock)

        history = ticker.history(start=date)

        if history.empty:
            return render_template(
                "index.html",
                error="No historical data available."
            )

        buy_price = history["Close"].iloc[0]

        current_price = history["Close"].iloc[-1]

        shares = amount / buy_price

        current_value = round(
            shares * current_price,
            2
        )

        profit = round(
            current_value - amount,
            2
        )

        return render_template(
            "index.html",
            investment_amount=amount,
            buy_price=round(buy_price, 2),
            current_price=round(current_price, 2),
            current_value=current_value,
            profit=profit
        )

    except Exception:

        return render_template(
            "index.html",
            error="Calculation failed."
        )
    



if __name__ == "__main__":
    app.run(debug=True, port=5002)