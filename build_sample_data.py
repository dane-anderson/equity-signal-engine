import os
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

tickers = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "COST",
    "NFLX", "ADBE", "PEP", "CSCO", "TMUS", "AMD", "TXN", "INTU", "QCOM", "AMGN",
    "HON", "INTC", "AMAT", "BKNG", "ISRG", "CMCSA", "ADP", "GILD", "LRCX", "ADI",
    "SBUX", "VRTX", "MU", "PANW", "MELI", "MDLZ", "REGN", "SNPS", "KLAC", "CDNS",
    "MAR", "CRWD", "ASML", "ABNB", "FTNT", "ORLY", "CSX", "MNST", "PYPL", "MRVL",
    "ADSK", "KDP", "NXPI", "AEP", "CHTR", "WDAY", "PCAR", "ROST", "PAYX", "CTAS",
    "FAST", "ODFL", "EXC", "DDOG", "XEL", "EA", "CPRT", "FANG", "BKR", "TEAM",
    "GEHC", "CCEP", "DLTR", "KHC", "IDXX", "LULU", "MCHP", "CSGP", "ANSS", "TTWO",
    "MDB", "ZS", "BIIB", "WBD", "ILMN", "ARM", "ON", "DASH", "TTD", "ZS",
    "SPLK", "DXCM", "ENPH", "RIVN", "LCID", "JD", "PDD", "SHOP", "ZM", "DOCU"
]

tickers = list(dict.fromkeys(tickers))

os.makedirs("data", exist_ok=True)

saved = []
skipped = []
saved_signals = []

for ticker in tickers:
    try:
        print(f"Downloading {ticker}...")

        df = yf.download(
            ticker,
            period="5y",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            skipped.append(ticker)
            print(f"Skipped {ticker}: no data")
            continue

        if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)

        df.to_csv(f"data/{ticker}.csv")
        saved.append(ticker)
        print(f"Saved data/{ticker}.csv")

        # Build features for ranking
        work_df = df[["Close"]].copy()

        work_df["Return_1d"] = work_df["Close"].pct_change()
        work_df["MA_5"] = work_df["Close"].rolling(5).mean()
        work_df["MA_10"] = work_df["Close"].rolling(10).mean()
        work_df["Return_5d"] = work_df["Close"].pct_change(5)
        work_df["MA_ratio"] = work_df["MA_5"] / work_df["MA_10"]
        work_df["Volatility_5d"] = work_df["Return_1d"].rolling(5).std()
        work_df["Momentum_3d"] = work_df["Close"] - work_df["Close"].shift(3)

        work_df["Future_Close_5d"] = work_df["Close"].shift(-5)
        work_df["Future_Return_5d"] = (
            (work_df["Future_Close_5d"] - work_df["Close"]) / work_df["Close"]
        )
        work_df["Target"] = (work_df["Future_Return_5d"] > 0.01).astype(int)

        work_df = work_df.dropna()

        if len(work_df) < 50:
            print(f"Skipped signal build for {ticker}: not enough usable rows")
            continue

        features = [
            "Return_1d",
            "Return_5d",
            "MA_5",
            "MA_10",
            "MA_ratio",
            "Volatility_5d",
            "Momentum_3d",
        ]

        X = work_df[features]
        y = work_df["Target"]

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight="balanced",
        )
        model.fit(X, y)

        latest = X.iloc[[-1]]
        probability = model.predict_proba(latest)[0][1]

        saved_signals.append(
            {
                "Ticker": ticker,
                "Up Probability": probability,
            }
        )

    except Exception as e:
        skipped.append(ticker)
        print(f"Skipped {ticker}: {e}")

print("\nDone.")
print(f"Saved: {len(saved)}")
print(f"Skipped: {len(skipped)}")
if skipped:
    print("Skipped tickers:", skipped)

results_df = pd.DataFrame(saved_signals)
results_df = results_df.sort_values(by="Up Probability", ascending=False)
results_df.to_csv("data/top_signals.csv", index=False)
print("Saved top signals to data/top_signals.csv")