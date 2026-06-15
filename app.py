from flask import Flask, jsonify, request,render_template
import yfinance as yf
import pandas as pd

app=Flask(__name__)

def calculate_return(history):
    if history.empty:
        return "Not available"
    
    start_price=history["Close"].iloc[0]
    end_price=history["Close"].iloc[-1]

    return ((end_price -start_price) / start_price) *100


def get_detail(stock):
    stock_df = pd.read_csv("stocks.csv")
    stock = stock.strip()

    search_item = stock.lower()

    matched = None

    for _, row in stock_df.iterrows():
        if row["name"].lower() == search_item:
            matched = row
            break

    if matched is not None:
        stock = matched["ticker"]

    try:
        ticker=yf.Ticker(stock)
        info=ticker.info

        name=info.get("longName") or info.get("shortName") or "Not Available"

        symbol=info.get("symbol") or "Not Available"

        price=info.get("currentPrice") or info.get("regularMarketPrice") or "Not Available"

        asset_type=info.get("quoteType") or "Not Available"

        explanation=info.get("longBusinessSummary") or "Not Available"
        if explanation != "Not Available":
            explanation=explanation[:330] +"..."

        history_1y=ticker.history(period="1y")
        history_3y=ticker.history(period="3y")
        history_5y=ticker.history(period="5y")

        returns_1y=round(calculate_return(history_1y),2)
        returns_3y=round(calculate_return(history_3y),2)
        returns_5y=round(calculate_return(history_5y),2)

        return {
    "name": name,
    "symbol": symbol,
    "price": price,
    "asset_type":asset_type,
    "explanation": explanation,
    "return_1y": returns_1y,
    "return_3y": returns_3y,
    "return_5y": returns_5y,
    }



    except Exception:
        return {"error":"Invalid Symbol -_-"}
    

@app.route("/")
def home():
    
    return render_template("home.html")

@app.route("/search",methods=["POST"])
def search():
    stock = request.form["stock"]
    details = get_detail(stock)
    return render_template("home.html", **details)


@app.route("/historical_calculator",methods=["POST"])
def historical_calulator():
    
    stock=request.form["stock"]
    details = get_detail(stock)
    stock=request.form["stock"]
    amount=float(request.form["amount"])
    start_date=request.form["start_date"]
    end_date=request.form["end_date"]

    ticker=yf.Ticker(stock)
    history=ticker.history(start=start_date,end=end_date)

    if history.empty:
        return render_template("home.html",returns="No past Data found for the given date")
    
    buy_price=history["Close"].iloc[0]
    sell_price=history["Close"].iloc[-1]

    shares=amount/buy_price
    final_returns=round(shares*sell_price,2)
    profit=round(final_returns-amount,2)


    return render_template(
    "home.html",
    **details,
    amount=amount,
    start_date=start_date,
    end_date=end_date,
    buy_price=round(buy_price, 2),
    sell_price=round(sell_price, 2),
    returns=profit
)

if __name__ == "__main__":
    app.run(debug=True,port=5001)





    

