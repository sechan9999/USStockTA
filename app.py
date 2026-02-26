"""
US Stock Technical Analyzer — Flask Backend
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, jsonify, request, render_template, send_from_directory
import yfinance as yf
import numpy as np
import os
import os.path
import tempfile
from openai import OpenAI
import json
import requests

# Set yfinance cache directory to /tmp to prevent Vercel's read-only error
yf_cache_dir = os.path.join(tempfile.gettempdir(), 'yfinance_cache')
os.makedirs(yf_cache_dir, exist_ok=True)
try:
    yf.set_tz_cache_location(yf_cache_dir)
except AttributeError:
    pass

# Setup a session with a real browser User-Agent to prevent Yahoo from blocking the request
yf_session = requests.Session()
yf_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
})

app = Flask(__name__, template_folder="templates", static_folder="static")


def safe(v):
    """Convert numpy/pandas values to plain Python, None if NaN."""
    if v is None:
        return None
    try:
        if np.isnan(float(v)):
            return None
        return float(v)
    except Exception:
        return None


def safe_int(v):
    try:
        f = float(v)
        return int(f) if not np.isnan(f) else 0
    except Exception:
        return 0


# ─── ROUTES ────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stock")
def get_stock():
    symbol   = request.args.get("symbol", "AAPL").upper().strip()
    interval = request.args.get("interval", "1d")
    period   = request.args.get("period",   "6mo")

    try:
        ticker = yf.Ticker(symbol, session=yf_session)
        hist   = ticker.history(period=period, interval=interval, auto_adjust=True)
        info   = ticker.fast_info          # faster than .info

        if hist is None or hist.empty:
            return jsonify({"error": f"No data found for '{symbol}'. Check the ticker."}), 404

        # Remove timezone from index for clean timestamps
        hist.index = hist.index.tz_localize(None) if hist.index.tzinfo is None else hist.index.tz_convert(None)

        # Build response
        timestamps = [int(t.timestamp()) for t in hist.index]
        opens   = [safe(v) for v in hist["Open"]]
        highs   = [safe(v) for v in hist["High"]]
        lows    = [safe(v) for v in hist["Low"]]
        closes  = [safe(v) for v in hist["Close"]]
        volumes = [safe_int(v) for v in hist["Volume"]]

        # Meta — use fast_info first, fall back gracefully
        price    = safe(getattr(info, "last_price",            None)) or closes[-1]
        prev     = safe(getattr(info, "previous_close",        None)) or closes[-2]
        mktcap   = safe(getattr(info, "market_cap",            None))
        h52      = safe(getattr(info, "year_high",             None))
        l52      = safe(getattr(info, "year_low",              None))
        shares   = safe(getattr(info, "shares",                None))

        # Try slower .info for extra fields (PE, beta, name)
        slow = {}
        try:
            slow = ticker.info or {}
        except Exception:
            pass

        meta = {
            "symbol":              symbol,
            "longName":            slow.get("longName") or slow.get("shortName") or symbol,
            "regularMarketPrice":  price,
            "previousClose":       prev,
            "marketCap":           mktcap,
            "fiftyTwoWeekHigh":    h52,
            "fiftyTwoWeekLow":     l52,
            "trailingPE":          safe(slow.get("trailingPE")),
            "beta":                safe(slow.get("beta")),
            "dividendYield":       safe(slow.get("dividendYield")),
            "sector":              slow.get("sector", ""),
            "industry":            slow.get("industry", ""),
        }

        return jsonify({
            "symbol":     symbol,
            "meta":       meta,
            "timestamps": timestamps,
            "open":       opens,
            "high":       highs,
            "low":        lows,
            "close":      closes,
            "volume":     volumes,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/search")
def search_ticker():
    """Return basic info for autocomplete / quick validation."""
    q = request.args.get("q", "").upper().strip()
    if not q:
        return jsonify([])
    try:
        # Avoid yf.Search because it might crash with default session on vercel
        # but yfinance doesn't easily let us pass a session to Search. We'll try just the Ticker approach
        res = yf.Search(q, max_results=6)
        quotes = res.quotes or []
        return jsonify([
            {"symbol": r.get("symbol",""), "name": r.get("longname") or r.get("shortname",""), "type": r.get("quoteType","")}
            for r in quotes
        ])
    except Exception:
        return jsonify([])


@app.route("/api/ai-summary", methods=["POST"])
def ai_summary():
    """Generate a natural language summary using OpenAI API based on the passed indicators."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "OPENAI_API_KEY environment variable is not set"}), 500

    client = OpenAI(api_key=api_key)

    symbol = data.get("symbol", "the stock")
    indicators = data.get("indicators", {})

    prompt = f"""
    You are an expert technical analyst. I will provide you with the current technical indicator readings for {symbol}. 
    Provide a concise, 2-3 sentence summary of the technical health of the stock.
    Be objective and highlight the most critical signals. Do NOT give financial advice.
    
    Data:
    {json.dumps(indicators, indent=2)}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Or whatever model you prefer
            messages=[
                {"role": "system", "content": "You are a helpful and expert technical stock market analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        summary = response.choices[0].message.content.strip()
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*50}")
    print(f"  US Stock Technical Analyzer")
    print(f"  Open your browser → http://localhost:{port}")
    print(f"{'='*50}\n")
    app.run(debug=False, port=port, threaded=True)
