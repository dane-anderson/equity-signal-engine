# 📈 AI Equity Signal Engine

An ML-powered stock screening system that ranks large-cap equities by predicted short-term return probability.

## 🚀 Live App
👉 https://stock-direction-ml-model-cdrysyqq5qzoadhqry9e44.streamlit.app/

---

## 🧠 Overview

This project builds a machine learning pipeline that:

- Predicts whether a stock will rise >1% over the next 5 trading days
- Generates probability-based signals across a large-cap universe
- Ranks stocks to identify top opportunities

---

## ⚙️ Tech Stack

- Python
- Pandas
- Scikit-learn (Random Forest)
- Streamlit

---

## 📊 Features

- 📈 Stock-level prediction with probability output
- 🏆 Top 5 ranked stocks by predicted return probability
- 🧠 Feature engineering:
  - Momentum
  - Volatility
  - Moving averages
- ⚡ Fast UI using precomputed signals
- 📉 Historical price visualization

---

## 🏗 Architecture

- `build_sample_data.py`  
  → Data pipeline + model inference across all tickers  
  → Generates ranked signals (`top_signals.csv`)

- `app.py`  
  → Interactive Streamlit frontend  
  → Displays rankings + runs live prediction per stock

- `data/`  
  → Historical stock data + precomputed signals

---

## ⚡ Performance Design

To ensure fast load times:

- Signals are precomputed offline
- The app loads rankings instantly from disk
- Avoids recomputing models in real time

---

## 🧠 Key Insight

Separating computation from presentation dramatically improves performance and reliability in ML applications.

---

## ⚠️ Disclaimer

This project is for educational purposes only and does not constitute financial advice.

---

## 📸 Screenshot

<img width="847" height="651" alt="app-screenshot" src="https://github.com/user-attachments/assets/3816dbba-acb4-40df-91d5-00334010b871" />

Top-ranked large-cap stocks by predicted short-term return probability, with model-generated signals and historical price visualization.

## 🧠 Design Decisions

### Live vs Precomputed Data

Initially, this app fetched live market data and computed signals on each user request. While this approach ensured up-to-date results, it introduced noticeable latency due to API calls and feature computation.

To improve performance and create a better user experience, the system was redesigned to precompute signals and serve them instantly from stored results.

This tradeoff prioritizes speed and responsiveness over fully real-time computation, significantly improving user experience.

---

## ⚡ System Architecture

The application is structured in two main parts:

- **Data Pipeline:** Fetches market data, computes features, and generates model signals  
- **Application Layer:** Loads precomputed results and displays them instantly to the user  

This separation allows the app to remain fast while still incorporating complex data processing behind the scenes.
