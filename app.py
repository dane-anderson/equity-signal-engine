import pandas as pd
import yfinance as yf
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Stock Direction ML Model", layout="centered")

st.title("📈 Stock Direction ML Model")
st.write("Predict whether a stock may rise more than 1% over the next 5 trading days.")

# User input
ticker = st.text_input("Enter a stock ticker", value="AAPL").upper()

if st.button("Run Prediction"):
    try:
        # Download stock data
        df = yf.download(ticker, start="2020-01-01", end="2026-01-01")

        if df.empty:
            st.error("No data found for that ticker.")
        else:
            # Flatten columns if needed
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
            df = df[["Close"]]

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
            st.write(f"**Prediction:** {'Up > 1% in next 5 days' if prediction == 1 else 'No strong upward move predicted'}")
            st.write(f"**Confidence (probability of upward move):** {probability:.2%}")

            st.subheader("Latest Feature Values")
            st.dataframe(latest_features)

    except Exception as e:
        st.error(f"Something went wrong: {e}")