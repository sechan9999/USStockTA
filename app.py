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

@app.route("/api/health")
def health_check():
    return jsonify({"status": "ok", "message": "Vercel backend is fully operational."})


@app.route("/api/stock")
def get_stock():
    symbol   = request.args.get("symbol", "AAPL").upper().strip()
    interval = request.args.get("interval", "1d")
    period   = request.args.get("period",   "6mo")

    try:
        # 1. Fetch data directly from Yahoo Finance API (Bypasses yfinance library issues on Vercel)
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={period}&includePrePost=false"
        
        # We need headers mimicking a real browser to prevent Yahoo from returning 403 Forbidden
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        res = requests.get(url, headers=headers, timeout=10)
        
        # Try a fallback url if query2 fails
        if not res.ok:
            url_fallback = url.replace("query2", "query1")
            res = requests.get(url_fallback, headers=headers, timeout=10)
            res.raise_for_status()
            
        data = res.json()
        if data.get("chart", {}).get("error"):
            return jsonify({"error": data["chart"]["error"]["description"]}), 404
            
        result = data["chart"]["result"][0]
        timestamps = result.get("timestamp", [])
        
        if not timestamps:
            return jsonify({"error": f"No data found for '{symbol}'. Check the ticker."}), 404

        quote = result["indicators"]["quote"][0]
        m = result["meta"]
        
        opens   = [safe(v) for v in quote.get("open", [])]
        highs   = [safe(v) for v in quote.get("high", [])]
        lows    = [safe(v) for v in quote.get("low", [])]
        closes  = [safe(v) for v in quote.get("close", [])]
        volumes = [safe_int(v) for v in quote.get("volume", [])]

        meta = {
            "symbol":              symbol,
            "longName":            m.get("longName") or m.get("shortName") or symbol,
            "regularMarketPrice":  m.get("regularMarketPrice") or closes[-1],
            "previousClose":       m.get("previousClose") or m.get("chartPreviousClose") or closes[-1],
            "marketCap":           None, # Fast API doesn't have marketCap unfortunately, avoiding extra calls to prevent IP ban
            "fiftyTwoWeekHigh":    None,
            "fiftyTwoWeekLow":     None,
            "trailingPE":          None,
            "beta":                None,
            "dividendYield":       None,
            "sector":              "",
            "industry":            ""
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
