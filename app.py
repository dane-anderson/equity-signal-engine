import os
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Nasdaq-100 Stock Direction ML Model", layout="centered")

st.title("📈 Nasdaq-100 Stock Direction ML Model")
st.write("Predict short-term price direction across a large-cap stock universe using historical data.")
st.caption("Public demo uses bundled historical data for reliability.")

available_files = os.listdir("data")
tickers = sorted([f.replace(".csv", "") for f in available_files if f.endswith(".csv")])

ticker = st.selectbox("Choose a stock ticker", tickers, index=tickers.index("AAPL") if "AAPL" in tickers else 0)


@st.cache_data
def load_stock_data(symbol: str) -> pd.DataFrame:
    path = f"data/{symbol}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


if st.button("Run Prediction"):
    try:
        df = load_stock_data(ticker)

        if df.empty:
            st.error("No bundled data found for that ticker.")
            st.stop()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if "Close" not in df.columns:
            st.error("Close column missing from data.")
            st.stop()

        st.line_chart(df["Close"])

        df = df[["Close"]].copy()

        df["Return_1d"] = df["Close"].pct_change()
        df["MA_5"] = df["Close"].rolling(5).mean()
        df["MA_10"] = df["Close"].rolling(10).mean()
        df["Return_5d"] = df["Close"].pct_change(5)
        df["MA_ratio"] = df["MA_5"] / df["MA_10"]
        df["Volatility_5d"] = df["Return_1d"].rolling(5).std()
        df["Momentum_3d"] = df["Close"] - df["Close"].shift(3)

        df["Future_Close_5d"] = df["Close"].shift(-5)
        df["Future_Return_5d"] = (df["Future_Close_5d"] - df["Close"]) / df["Close"]
        df["Target"] = (df["Future_Return_5d"] > 0.01).astype(int)

        df = df.dropna()

        if df.empty or len(df) < 50:
            st.error("Not enough usable data after feature engineering.")
            st.stop()

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

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)

        latest_features = X.iloc[[-1]]
        prediction = model.predict(latest_features)[0]
        probability = model.predict_proba(latest_features)[0][1]

        st.subheader(f"Results for {ticker}")

        if prediction == 1:
            st.success(f"📈 Likely UP (>1%) with {probability:.2%} confidence")
        else:
            st.warning(f"📉 No strong upward move predicted ({probability:.2%} confidence)")

        st.subheader("Latest Feature Values")
        st.dataframe(latest_features)

    except Exception as e:
        st.error(f"Something went wrong: {e}")