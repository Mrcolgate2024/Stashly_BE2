import pandas as pd
import numpy as np
from langchain_core.messages import AIMessage
from tools.yfinance_utils import fetch_yfinance_metrics
from config.settings import Settings

def run_returns_agent(state, settings: Settings):
    try:
        df = pd.read_excel("market_data.xlsx", sheet_name=None)
        historical_df = df["historical_data"].dropna(how="all")

        if pd.api.types.is_datetime64_any_dtype(historical_df.iloc[:, 0]) or 'date' in str(historical_df.columns[0]).lower():
            historical_df.set_index(historical_df.columns[0], inplace=True)

        if (historical_df > 1.0).all().all():
            returns = historical_df.pct_change().dropna()
        else:
            returns = historical_df.copy()

        summary = []
        for asset in returns.columns:
            ret_series = returns[asset].dropna()
            if len(ret_series) < 2:
                continue
            total_return = (1 + ret_series).prod() - 1
            ann_return = (1 + total_return) ** (12 / len(ret_series)) - 1
            volatility = ret_series.std() * np.sqrt(12)
            excess_return = ann_return - 0.01
            sharpe_ratio = excess_return / volatility if volatility else np.nan
            summary.append({
                "Asset": asset,
                "Annual Return (%)": round(ann_return * 100, 2),
                "Volatility (%)": round(volatility * 100, 2),
                "Sharpe Ratio": round(sharpe_ratio, 2)
            })

        result_df = pd.DataFrame(summary)

        # fallback from yfinance if user asks for an external ticker
        user_msg = state.get("messages", [])[-1].content.lower()
        if "ticker:" in user_msg:
            ticker = user_msg.split("ticker:")[1].split()[0].upper()
            yf_data = fetch_yfinance_metrics(ticker)
            if yf_data["success"]:
                result_df = pd.concat([
                    result_df,
                    pd.DataFrame([{
                        "Asset": ticker,
                        "Annual Return (%)": yf_data["annual_return"],
                        "Volatility (%)": yf_data["volatility"],
                        "Sharpe Ratio": yf_data["sharpe_ratio"]
                    }])
                ])

        return {"messages": [AIMessage(content="Here is the return and risk analysis:\n\n" + result_df.to_markdown(index=False))]}

    except Exception as e:
        return {"messages": [AIMessage(content=f"Error in returns agent: {str(e)}")]}
