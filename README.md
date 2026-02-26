# US Stock Technical Analyzer

A powerful, single-page web application and Flask backend for comprehensive US stock market technical analysis. 

This app fetches real-time and historical market data using `yfinance` to display beautiful, interactive technical charts (via Lightweight Charts) and calculates key technical indicators instantly.

## Features

- **Interactive Charts**: Candlestick, Line, and OHLC Bar charts tailored for financial analysts using TradingView's Lightweight Charts.
- **Technical Indicators**: 
  - Simple Moving Averages (SMA 20, SMA 50)
  - Exponential Moving Averages (EMA 20, EMA 50)
  - Bollinger Bands
  - Volume Weighted Average Price (VWAP)
  - Relative Strength Index (RSI)
  - MACD (Moving Average Convergence Divergence)
  - Stochastic Oscillator
  - Average True Range (ATR)
- **Algorithm-based Scoring**: Generates a unified Technical Score and signal (STRONG BUY, BUY, HOLD, SELL, STRONG SELL) based on composite indicator readings.
- **Pattern Recognition**: Automatically detects significant events like Golden/Death crosses, Bollinger Band breakouts, MACD crossovers, RSI extremes, inside bars, and volume spikes.
- **🤖 AI Technical Summary Feature**: Leverages OpenAI to generate a concise, natural language technical analysis report based on the current indicators. 

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sechan9999/USStockTA.git
   cd USStockTA
   ```

2. **Install the required dependencies:**
   Ensure you have Python 3 installed.
   ```bash
   pip install flask yfinance numpy pandas openai
   ```

3. **Set your OpenAI API Key (Required for the AI Summary feature):**
   ```bash
   # On Windows (PowerShell)
   $env:OPENAI_API_KEY="your-api-key-here"

   # On macOS/Linux
   export OPENAI_API_KEY="your-api-key-here"
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Open in browser:**
   Navigate to `http://localhost:5000`   https://sechan9999.github.io/USStockTA/

## Tech Stack
- **Frontend**: Vanilla JavaScript (ES6+), CSS3 Variables, HTML5, [Lightweight Charts](https://github.com/tradingview/lightweight-charts)
- **Backend**: Python, Flask, Pandas, Numpy, `yfinance`, OpenAI API

## Disclaimer
*For educational purposes only — not financial advice. All data and analysis provided by this application should be verified independently before making any investment decisions.*
