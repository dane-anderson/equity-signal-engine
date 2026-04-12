from pandas_datareader import data as pdr
import pandas as pd
import yfinance as yf
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Stock Direction ML Model", layout="centered")

st.title("📈 Stock Direction ML Model")
st.write("Predict whether a stock may rise more than 1% over the next 5 trading days.")

ticker = st.text_input("Enter a stock ticker", value="AAPL").strip().upper()



@st.cache_data(ttl=3600)
def load_stock_data(symbol: str) -> pd.DataFrame:
    symbol = symbol.strip().upper()

    # Try Yahoo download first
    try:
        df = yf.download(
            symbol,
            period="5y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    # Try Yahoo Ticker.history second
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="5y", interval="1d", auto_adjust=False)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    # Fallback to Stooq
    try:
        stooq_symbol = f"{symbol}.US"
        df = pdr.DataReader(stooq_symbol, "stooq")
        if df is not None and not df.empty:
            df = df.sort_index()
            return df
    except Exception:
        pass

    return pd.DataFrame()

if st.button("Run Prediction"):
    try:
        df = load_stock_data(ticker)

        if df.empty:
            st.error("Still no data returned. Try AAPL, MSFT, NVDA, TSLA, SPY, or AMZN.")
            st.stop()

        # Make chart work whether Close is Series or DataFrame
        close_for_chart = df["Close"]
        if isinstance(close_for_chart, pd.DataFrame):
            close_for_chart = close_for_chart.iloc[:, 0]
        st.line_chart(close_for_chart)

        # Flatten columns if needed
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Close"]].copy()

        # Feature engineering
        df["Return_1d"] = df["Close"].pct_change()
        df["MA_5"] = df["Close"].rolling(5).mean()
        df["MA_10"] = df["Close"].rolling(10).mean()
        df["Return_5d"] = df["Close"].pct_change(5)
        df["MA_ratio"] = df["MA_5"] / df["MA_10"]
        df["Volatility_5d"] = df["Return_1d"].rolling(5).std()
        df["Momentum_3d"] = df["Close"] - df["Close"].shift(3)

        # Target
        df["Future_Close_5d"] = df["Close"].shift(-5)
        df["Future_Return_5d"] = (df["Future_Close_5d"] - df["Close"]) / df["Close"]
        df["Target"] = (df["Future_Return_5d"] > 0.01).astype(int)

        # Clean data
        df = df.dropna()

        if df.empty or len(df) < 50:
            st.error("Not enough usable data after feature engineering.")
            st.stop()

        # Features and target
        features = [
            "Return_1d",
            "Return_5d",
            "MA_5",
            "MA_10",
            "MA_ratio",
            "Volatility_5d",
            "Momentum_3d",
        ]
        X = df[features]
        y = df["Target"]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        # Model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)

        # Latest row prediction
        latest_features = X.iloc[[-1]]
        prediction = model.predict(latest_features)[0]
        probability = model.predict_proba(latest_features)[0][1]

        # Display results
        st.subheader(f"Results for {ticker}")

        if prediction == 1:
            st.success(f"📈 Likely UP (>1%) with {probability:.2%} confidence")
        else:
            st.warning(f"📉 No strong upward move predicted ({probability:.2%} confidence)")

        st.subheader("Latest Feature Values")
        st.dataframe(latest_features)

    except Exception as e:
        st.error(f"Something went wrong: {e}")