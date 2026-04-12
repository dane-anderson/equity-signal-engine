import os
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Equity Signal Engine", layout="centered")

st.title("📈 AI Equity Signal Engine")

st.markdown("""
Predict short-term stock direction across a large-cap equity universe using a machine learning model.

Built with:
- 🧠 Random Forest ML model
- 📊 Financial feature engineering
- ⚡ Streamlit interactive app

⚠️ Educational project — not financial advice
""")

st.markdown("---")
st.subheader("📊 Select Stock")

DATA_DIR = "data"
files = [f.replace(".csv", "") for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
tickers = sorted(files)

ticker = st.selectbox(
    "Choose a stock ticker",
    tickers,
    index=tickers.index("AAPL") if "AAPL" in tickers else 0,
)


@st.cache_data
def load_stock_data(symbol: str) -> pd.DataFrame:
    path = f"{DATA_DIR}/{symbol}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


@st.cache_data
def load_top_signals() -> pd.DataFrame:
    path = f"{DATA_DIR}/top_signals.csv"
    df = pd.read_csv(path)
    return df


st.markdown("---")
st.subheader("🏆 Top 5 Stocks Right Now")
st.caption("Ranked by predicted probability of >1% return over the next 5 trading days")

top5 = load_top_signals().head(5).copy()

def label_signal(p):
    if p > 0.65:
        return "🔥 Strong"
    elif p > 0.55:
        return "📈 Bullish"
    elif p > 0.45:
        return "⚖️ Neutral"
    else:
        return "📉 Bearish"

top5["Signal"] = top5["Up Probability"].apply(label_signal)
top5["Up Probability"] = top5["Up Probability"].map(lambda x: f"{x:.2%}")

st.dataframe(top5, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("📈 Model Output")

if st.button("Run Prediction"):
    with st.spinner("Running model..."):
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

            accuracy = model.score(X_test, y_test)
            latest_features = X.iloc[[-1]]
            prediction = model.predict(latest_features)[0]
            probability = model.predict_proba(latest_features)[0][1]

            if probability > 0.65:
                signal = "🔥 Strong Buy Signal"
            elif probability > 0.55:
                signal = "📈 Mild Bullish Signal"
            elif probability > 0.45:
                signal = "⚖️ Neutral Signal"
            else:
                signal = "📉 Bearish Signal"

            st.subheader(f"{ticker} Prediction")
            st.markdown(f"### {signal}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Prediction", "UP 📈" if prediction == 1 else "NO STRONG MOVE 📉")

            with col2:
                st.metric("Confidence", f"{probability:.2%}")

            with col3:
                st.metric("Model Accuracy", f"{accuracy:.2%}")

            st.subheader("Latest Feature Values")
            st.dataframe(latest_features, use_container_width=True)

        except Exception as e:
            st.error(f"Something went wrong: {e}")