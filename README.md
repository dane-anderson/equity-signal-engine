
# 📈 AI Equity Signal Engine

ML-powered stock screening system that ranks large-cap equities by predicted short-term return probability.

---

## 🚀 Live App

👉 https://stock-direction-ml-model-cdrysyqq5qzoadhqry9e44.streamlit.app/

---

## 🧠 What It Does

This project uses machine learning to predict whether a stock will increase by more than 1% over the next 5 trading days and ranks stocks based on predicted probability.

---

## 🧠 Overview

The system builds a complete pipeline that:

- Processes historical stock data  
- Engineers predictive features  
- Applies a machine learning model  
- Generates probability-based signals  
- Ranks stocks to identify top opportunities  

This project demonstrates how ML models can be integrated into real-world decision systems.

---

## ✨ Features

- 📈 Stock-level prediction with probability output  
- 🏆 Top 5 ranked stocks by predicted return probability  
- 🧠 Feature engineering:
  - Momentum  
  - Volatility  
  - Moving averages  
- ⚡ Fast UI powered by precomputed signals  
- 📉 Historical price visualization  

---

## 🏗 Architecture

**Data Pipeline (`build_sample_data.py`)**
- Fetches market data  
- Computes features  
- Runs model inference  
- Generates ranked signals (`top_signals.csv`)  

**Application Layer (`app.py`)**
- Streamlit frontend  
- Loads precomputed signals instantly  
- Displays rankings and stock-level predictions  

---

## ⚡ Performance Design

To ensure fast performance:

- Signals are precomputed offline  
- The app loads results instantly from disk  
- Avoids recomputing models on every request  

This separation of computation and presentation improves both speed and reliability.

---

## 🛠 Tech Stack

- Python  
- Pandas  
- Scikit-learn (Random Forest)  
- Streamlit  

---

## 📊 Example Output

Top-ranked large-cap stocks by predicted short-term return probability, with model-generated signals and historical price visualization.

---

## 💡 Key Insight

Separating heavy computation from the user interface significantly improves performance in real-world ML applications.

---

## ⚠️ Disclaimer

This project is for educational purposes only and does not constitute financial advice.

---

## 🚀 Future Improvements

- Expand model features and improve accuracy  
- Add real-time data integration  
- Introduce portfolio-level optimization  
- Build a more advanced interactive dashboard  

---

## 💡 Vision

This project represents a step toward building intelligent financial systems that combine data, machine learning, and usability to support better decision-making.
