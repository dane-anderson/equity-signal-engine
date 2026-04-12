
import os
import yfinance as yf

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

# remove duplicates while preserving order
tickers = list(dict.fromkeys(tickers))

os.makedirs("data", exist_ok=True)

saved = []
skipped = []

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

    except Exception as e:
        skipped.append(ticker)
        print(f"Skipped {ticker}: {e}")

print("\nDone.")
print(f"Saved: {len(saved)}")
print(f"Skipped: {len(skipped)}")
if skipped:
    print("Skipped tickers:", skipped)