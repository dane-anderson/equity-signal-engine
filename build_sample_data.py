import os
import yfinance as yf

tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY", "AMZN"]
os.makedirs("data", exist_ok=True)

for ticker in tickers:
    df = yf.download(ticker, period="5y", auto_adjust=False, progress=False)
    if df is None or df.empty:
        print(f"Skipped {ticker}: no data")
        continue

    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    df.to_csv(f"data/{ticker}.csv")
    print(f"Saved data/{ticker}.csv")