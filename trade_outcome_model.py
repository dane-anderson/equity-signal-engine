import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

# Choose a stock to study
ticker = "AAPL"

# Download historical data
df = yf.download(ticker, start="2020-01-01", end="2026-01-01")

# Keep only the close price
df = df.copy()
df.columns = df.columns.get_level_values(0)
df = df[["Close"]]

# Create simple features
df["Return_1d"] = df["Close"].pct_change()
df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_10"] = df["Close"].rolling(10).mean()
df["Return_5d"] = df["Close"].pct_change(5)
df["MA_ratio"] = df["MA_5"] / df["MA_10"]
df["Volatility_5d"] = df["Return_1d"].rolling(5).std()
df["Momentum_3d"] = df["Close"] - df["Close"].shift(3)

# Create target: 1 if price is higher in 5 days, else 0
df["Future_Close_5d"] = df["Close"].shift(-5)
df["Future_Return_5d"] = (df["Future_Close_5d"] - df["Close"]) / df["Close"]
df["Target"] = (df["Future_Return_5d"] > 0.01).astype(int)
print("\nNew target counts:")
print(df["Target"].value_counts())
# Drop rows with missing values
df = df.dropna()

print(df.head())
print("\nShape:", df.shape)
print("\nTarget counts:")
print(df["Target"].value_counts())
# Select features and target
features = ["Return_1d", "Return_5d", "MA_5", "MA_10", "MA_ratio", "Volatility_5d", "Momentum_3d"]
X = df[features]
y = df["Target"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# Build and train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
# Get probabilities
y_prob = model.predict_proba(X_test)[:, 1]

print("\nSample probabilities:")
print(y_prob[:10])

# Evaluate model
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
