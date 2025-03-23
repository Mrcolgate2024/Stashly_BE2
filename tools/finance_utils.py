import yfinance as yf
import pandas as pd
import numpy as np

def fetch_yfinance_metrics(ticker: str, period: str = "1y") -> dict:
    """
    Fetch historical data from Yahoo Finance and compute:
    - Annualized return
    - Annualized volatility
    - Sharpe ratio (assumes 1% risk-free rate)
    """
    try:
        data = yf.download(ticker, period=period, interval="1d", progress=False)
        data = data["Adj Close"].dropna()
        daily_returns = data.pct_change().dropna()
        ann_return = (1 + daily_returns.mean()) ** 252 - 1
        ann_volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (ann_return - 0.01) / ann_volatility if ann_volatility > 0 else None

        return {
            "ticker": ticker,
            "annual_return": round(ann_return * 100, 2),
            "volatility": round(ann_volatility * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2) if sharpe_ratio else None,
            "success": True
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e), "success": False}

