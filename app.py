from flask import Flask, request, render_template
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)


try:
    from transformers import pipeline
    sentiment_pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")
except Exception:
    sentiment_pipe = None


def get_sentiment(symbol):
    try:
        if sentiment_pipe is None:
            return None, None
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if not news:
            return None, None
        headlines = [item.get("content", {}).get("title", "") or item.get("title", "") for item in news[:8]]
        headlines = [h for h in headlines if h]
        if not headlines:
            return None, None
        results = sentiment_pipe(headlines)
        score_map = {"positive": 1, "negative": -1, "neutral": 0}
        scores = [score_map.get(r["label"], 0) * r["score"] for r in results]
        avg = sum(scores) / len(scores)
        if avg > 0.15:
            label = "Positive"
        elif avg < -0.15:
            label = "Negative"
        else:
            label = "Neutral"
        return label, round(avg, 2)
    except Exception:
        return None, None


def make_chart(history, color):
    if history.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(history.index, history["Close"], color=color, linewidth=1.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_growth_chart_monthly(monthly_history, amount, final_price):
    if monthly_history.empty:
        return None
    total_units = 0
    portfolio_values = []
    total_invested_values = []
    running_invested = 0
    for _, row in monthly_history.iterrows():
        total_units += amount / row["Close"]
        running_invested += amount
        portfolio_values.append(total_units * row["Close"])
        total_invested_values.append(running_invested)
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(monthly_history.index, portfolio_values, color="#198754", linewidth=1.5, label="Portfolio value")
    ax.plot(monthly_history.index, total_invested_values, color="#dc3545", linestyle="--", linewidth=1, label="Total invested")
    ax.set_title(f"Monthly SIP Growth (₹{amount:,.0f}/month)")
    ax.set_ylabel("Value (₹)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_growth_chart(history, amount):
    if history.empty:
        return None
    shares = amount / history["Close"].iloc[0]
    portfolio = history["Close"] * shares
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(history.index, portfolio, color="#198754", linewidth=1.5)
    ax.axhline(y=amount, color="#dc3545", linestyle="--", linewidth=1, label="Invested amount")
    ax.set_title(f"Growth of ₹{amount:,.0f}")
    ax.set_ylabel("Portfolio Value (₹)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_comparison_chart(h1, h2, label1, label2):
    if h1.empty or h2.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(h1.index, h1["Close"] / h1["Close"].iloc[0] * 100, label=label1, color="#0d6efd")
    ax.plot(h2.index, h2["Close"] / h2["Close"].iloc[0] * 100, label=label2, color="#fd7e14")
    ax.set_ylabel("Normalized Growth (Base 100)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def calculate_return(history):
    if history.empty:
        return "Not available"
    start_price = history["Close"].iloc[0]
    end_price = history["Close"].iloc[-1]
    return ((end_price - start_price) / start_price) * 100


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
        ticker = yf.Ticker(stock)
        info = ticker.info

        name = info.get("longName") or info.get("shortName") or "Not Available"
        symbol = info.get("symbol") or "Not Available"
        price = info.get("currentPrice") or info.get("regularMarketPrice") or "Not Available"
        asset_type = info.get("quoteType") or "Not Available"
        fund_objective = info.get("longBusinessSummary") or info.get("category") or "Not Available"
        expense_ratio = info.get("annualReportExpenseRatio") or info.get("netExpenseRatio")
        if expense_ratio:
            expense_ratio = round(expense_ratio * 100, 2)
        else:
            expense_ratio = "Not Available"

        explanation = info.get("longBusinessSummary") or "Not Available"
        if explanation != "Not Available":
            explanation = explanation[:513] + "..."

        history_1y = ticker.history(period="1y")
        history_3y = ticker.history(period="3y")
        history_5y = ticker.history(period="5y")

        returns_1y = round(calculate_return(history_1y), 2)
        returns_3y = round(calculate_return(history_3y), 2)
        returns_5y = round(calculate_return(history_5y), 2)

        chart_1y = make_chart(history_1y, "#0d6efd")
        chart_3y = make_chart(history_3y, "#6610f2")
        chart_5y = make_chart(history_5y, "#fd7e14")

        return {
            "name": name,
            "symbol": symbol,
            "price": price,
            "asset_type": asset_type,
            "explanation": explanation,
            "return_1y": returns_1y,
            "return_3y": returns_3y,
            "return_5y": returns_5y,
            "chart_1y": chart_1y,
            "chart_3y": chart_3y,
            "chart_5y": chart_5y,
            "fund_objective": fund_objective,
            "expense_ratio": expense_ratio,
        }

    except Exception:
        return {"error": "Invalid Symbol -_-"}


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/search", methods=["POST"])
def search():
    stock = request.form["stock"]
    details = get_detail(stock)
    sentiment_label, sentiment_score = get_sentiment(details.get("symbol", stock))
    return render_template("home.html", **details, sentiment_label=sentiment_label, sentiment_score=sentiment_score)


@app.route("/historical_calculator", methods=["POST"])
def historical_calulator():
    stock = request.form["stock"]
    details = get_detail(stock)
    amount = float(request.form["amount"])
    start_date = request.form["start_date"]
    end_date = request.form["end_date"]
    monthly = request.form.get("monthly") == "on"

    ticker = yf.Ticker(stock)
    history = ticker.history(start=start_date, end=end_date)
    history.index = history.index.tz_localize(None)

    if history.empty:
        return render_template("home.html", **details, returns="No past data found for the given date range")

    if monthly:
        monthly_history = history.resample("MS").first()
        total_invested = 0
        total_units = 0
        for _, row in monthly_history.iterrows():
            total_units += amount / row["Close"]
            total_invested += amount
        final_value = round(total_units * history["Close"].iloc[-1], 2)
        profit = round(final_value - total_invested, 2)
        buy_price = round(history["Close"].iloc[0], 2)
        sell_price = round(history["Close"].iloc[-1], 2)
        growth_chart = generate_growth_chart_monthly(monthly_history, amount, history["Close"].iloc[-1])
    else:
        buy_price = history["Close"].iloc[0]
        sell_price = history["Close"].iloc[-1]
        shares = amount / buy_price
        final_value = round(shares * sell_price, 2)
        profit = round(final_value - amount, 2)
        total_invested = amount
        buy_price = round(buy_price, 2)
        sell_price = round(sell_price, 2)
        growth_chart = generate_growth_chart(history, amount)

    return render_template(
        "home.html",
        **details,
        amount=amount,
        start_date=start_date,
        end_date=end_date,
        buy_price=buy_price,
        sell_price=sell_price,
        returns=profit,
        total_invested=round(total_invested, 2),
        monthly=monthly,
        growth_chart=growth_chart,
    )


@app.route("/compare", methods=["POST"])
def compare():
    stock1 = request.form["stock1"]
    stock2 = request.form["stock2"]

    details1 = get_detail(stock1)
    details2 = get_detail(stock2)

    comparison_chart = None
    winner = None
    diff = None

    if "error" not in details1 and "error" not in details2:
        try:
            t1 = yf.Ticker(details1["symbol"])
            t2 = yf.Ticker(details2["symbol"])
            h1 = t1.history(period="1y")
            h2 = t2.history(period="1y")
            comparison_chart = generate_comparison_chart(h1, h2, details1["symbol"], details2["symbol"])
        except Exception:
            pass

        if details1["return_1y"] > details2["return_1y"]:
            winner = details1["symbol"]
        elif details2["return_1y"] > details1["return_1y"]:
            winner = details2["symbol"]
        else:
            winner = "Tie"
        diff = round(abs(details1["return_1y"] - details2["return_1y"]), 2)

    return render_template(
        "compare.html",
        details1=details1,
        details2=details2,
        winner=winner,
        diff=diff,
        comparison_chart=comparison_chart,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)