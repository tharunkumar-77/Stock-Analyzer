from flask import Flask, request, render_template, jsonify
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

try:
    from transformers import pipeline
    sentiment_pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")
except Exception:
    sentiment_pipe = None


def get_suggestions():
    try:
        return pd.read_csv("stocks.csv")["name"].tolist()
    except Exception:
        return []


def get_sentiment(symbol):
    try:
        if sentiment_pipe is None:
            return None, None
        ticker = yf.Ticker(symbol)
        news = ticker.news or []
        headlines = [
            (item.get("content", {}).get("title") or item.get("title", ""))
            for item in news[:8]
        ]
        headlines = [h for h in headlines if h]
        if not headlines:
            return None, None
        results = sentiment_pipe(headlines)
        score_map = {"positive": 1, "negative": -1, "neutral": 0}
        scores = [score_map.get(r["label"].lower(), 0) * r["score"] for r in results]
        avg = sum(scores) / len(scores)
        label = "Positive" if avg > 0.15 else "Negative" if avg < -0.15 else "Neutral"
        return label, round(avg, 2)
    except Exception:
        return None, None


def calculate_return(history):
    if history.empty or len(history) < 2:
        return 0.0
    return ((history["Close"].iloc[-1] - history["Close"].iloc[0]) / history["Close"].iloc[0]) * 100


def calculate_projection(amount, return_5y, sentiment_score, horizon_days):
    try:
        annual_rate = (1 + float(return_5y) / 100) ** (1 / 5) - 1
    except Exception:
        annual_rate = 0.08
    years = horizon_days / 365.25
    sentiment_weight = max(0.0, 1.0 - (horizon_days / 365))
    sentiment_nudge = (sentiment_score or 0) * sentiment_weight * 0.05
    adjusted_rate = annual_rate + sentiment_nudge
    projected = round(amount * (1 + adjusted_rate) ** years, 2)
    uncertainty = max(0.08, 0.15 * (1 - years))
    return projected, round(projected * (1 - uncertainty), 2), round(projected * (1 + uncertainty), 2)


def _encode_figure(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_chart(history, color):
    if history.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(history.index, history["Close"], color=color, linewidth=1.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _encode_figure(fig)


def generate_growth_chart(history, amount):
    if history.empty:
        return None
    shares = amount / history["Close"].iloc[0]
    portfolio = history["Close"] * shares
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(history.index, portfolio, color="#198754", linewidth=1.5, label="Portfolio value")
    ax.axhline(y=amount, color="#dc3545", linestyle="--", linewidth=1, label="Invested amount")
    ax.set_title(f"Growth of ₹{amount:,.0f}")
    ax.set_ylabel("Portfolio Value (₹)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _encode_figure(fig)


def generate_growth_chart_monthly(monthly_history, amount, final_price):
    if monthly_history.empty:
        return None
    total_units, running_invested = 0, 0
    portfolio_values, total_invested_values = [], []
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
    return _encode_figure(fig)


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
    return _encode_figure(fig)


def get_detail(stock):
    suggestions = get_suggestions()
    stock = stock.strip()
    try:
        stock_df = pd.read_csv("stocks.csv")
        for _, row in stock_df.iterrows():
            if row["name"].lower() == stock.lower():
                stock = row["ticker"]
                break
    except Exception:
        pass
    try:
        ticker = yf.Ticker(stock)
        with ThreadPoolExecutor() as ex:
            f_info = ex.submit(lambda: ticker.info)
            f_1y   = ex.submit(ticker.history, period="1y")
            f_3y   = ex.submit(ticker.history, period="3y")
            f_5y   = ex.submit(ticker.history, period="5y")
            info       = f_info.result()
            history_1y = f_1y.result()
            history_3y = f_3y.result()
            history_5y = f_5y.result()

        name        = info.get("longName") or info.get("shortName") or "Not Available"
        symbol      = info.get("symbol") or stock.upper()
        price       = info.get("currentPrice") or info.get("regularMarketPrice") or "Not Available"
        asset_type  = info.get("quoteType") or "Not Available"
        raw_expense = info.get("annualReportExpenseRatio") or info.get("netExpenseRatio")
        expense_ratio = round(raw_expense * 100, 2) if raw_expense else "Not Available"
        raw_exp     = info.get("longBusinessSummary", "")
        explanation = (raw_exp[:510] + "...") if len(raw_exp) > 510 else raw_exp or "Not Available"

        return {
            "suggestions":   suggestions,
            "name":          name,
            "symbol":        symbol,
            "price":         price,
            "asset_type":    asset_type,
            "explanation":   explanation,
            "expense_ratio": expense_ratio,
            "return_1y":     round(calculate_return(history_1y), 2),
            "return_3y":     round(calculate_return(history_3y), 2),
            "return_5y":     round(calculate_return(history_5y), 2),
            "chart_1y":      make_chart(history_1y, "#0d6efd"),
            "chart_3y":      make_chart(history_3y, "#6610f2"),
            "chart_5y":      make_chart(history_5y, "#fd7e14"),
        }
    except Exception as e:
        return {"error": f"Could not fetch data: {e}", "suggestions": suggestions}


@app.route("/")
def home():
    return render_template("home.html", suggestions=get_suggestions())


@app.route("/api/search", methods=["POST"])
def api_search():
    stock = request.json.get("stock", "").strip()
    if not stock:
        return jsonify({"error": "No stock provided"})
    details = get_detail(stock)
    sentiment_label, sentiment_score = get_sentiment(details.get("symbol", stock))
    details["sentiment_label"] = sentiment_label
    details["sentiment_score"] = sentiment_score
    return jsonify(details)


@app.route("/api/historical", methods=["POST"])
def api_historical():
    data       = request.json
    stock      = data.get("stock", "").strip()
    amount     = float(data.get("amount", 0))
    start_date = data.get("start_date", "")
    end_date   = data.get("end_date", "")
    monthly    = data.get("monthly", False)

    if not stock:
        return jsonify({"error": "No stock provided"})
    try:
        ticker  = yf.Ticker(stock)
        history = ticker.history(start=start_date, end=end_date)
        history.index = history.index.tz_localize(None)

        if history.empty:
            return jsonify({"error": "No data found for the selected date range"})

        if monthly:
            monthly_history = history.resample("MS").first()
            total_units, total_invested = 0, 0
            for _, row in monthly_history.iterrows():
                total_units    += amount / row["Close"]
                total_invested += amount
            final_value  = round(total_units * history["Close"].iloc[-1], 2)
            profit       = round(final_value - total_invested, 2)
            buy_price    = round(history["Close"].iloc[0], 2)
            sell_price   = round(history["Close"].iloc[-1], 2)
            growth_chart = generate_growth_chart_monthly(monthly_history, amount, history["Close"].iloc[-1])
        else:
            buy_price    = round(history["Close"].iloc[0], 2)
            sell_price   = round(history["Close"].iloc[-1], 2)
            shares       = amount / buy_price
            final_value  = round(shares * sell_price, 2)
            profit       = round(final_value - amount, 2)
            total_invested = amount
            growth_chart = generate_growth_chart(history, amount)

        return jsonify({
            "buy_price":      buy_price,
            "sell_price":     sell_price,
            "returns":        profit,
            "total_invested": round(total_invested, 2),
            "growth_chart":   growth_chart,
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/future", methods=["POST"])
def api_future():
    data    = request.json
    stock   = data.get("stock", "").strip()
    amount  = float(data.get("amount", 0))
    horizon = int(data.get("horizon", 30))

    if not stock:
        return jsonify({"error": "No stock provided"})

    details = get_detail(stock)
    if "error" in details:
        return jsonify(details)

    sentiment_label, sentiment_score = get_sentiment(details["symbol"])
    projected, low, high = calculate_projection(amount, details["return_5y"], sentiment_score, horizon)

    horizon_labels = {7:"1 Week",14:"2 Weeks",30:"1 Month",90:"3 Months",180:"6 Months",365:"1 Year"}

    return jsonify({
        "projected":       projected,
        "low":             low,
        "high":            high,
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "horizon_label":   horizon_labels.get(horizon, f"{horizon} days"),
    })


@app.route("/api/compare", methods=["POST"])
def api_compare():
    data   = request.json
    stock1 = data.get("stock1", "").strip()
    stock2 = data.get("stock2", "").strip()

    details1 = get_detail(stock1)
    details2 = get_detail(stock2)

    if "error" in details1:
        return jsonify({"error": details1["error"]})
    if "error" in details2:
        return jsonify({"error": details2["error"]})

    try:
        h1 = yf.Ticker(details1["symbol"]).history(period="1y")
        h2 = yf.Ticker(details2["symbol"]).history(period="1y")
        comparison_chart = generate_comparison_chart(h1, h2, details1["symbol"], details2["symbol"])
    except Exception:
        comparison_chart = None

    r1, r2 = details1["return_1y"], details2["return_1y"]
    winner = details1["symbol"] if r1 > r2 else details2["symbol"] if r2 > r1 else "Tie"
    diff   = round(abs(r1 - r2), 2)

    return jsonify({
        "details1":         details1,
        "details2":         details2,
        "winner":           winner,
        "diff":             diff,
        "comparison_chart": comparison_chart,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)