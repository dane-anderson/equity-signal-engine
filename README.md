# 📈 Stock Direction ML Model

This project builds a machine learning model to predict whether a stock’s price will increase over the next 5 days.

---

## 🔍 Features Engineered

- 1-day return
- 5-day return
- Moving averages (5-day, 10-day)
- Moving average ratio (MA_5 / MA_10)
- 5-day volatility
- 3-day momentum

---

## ⚙️ Models Used

- Logistic Regression (baseline)
- Random Forest (improved performance)

---

## 📊 Results

- Random Forest improved recall on upward movements
- Class imbalance handled using `class_weight="balanced"`
- Demonstrates importance of feature engineering in financial ML

---

## 🧠 Key Takeaways

- Feature engineering is critical in financial prediction
- Class imbalance can distort model performance if ignored
- Tree-based models can outperform linear models on nonlinear signals

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- scikit-learn
- yfinance