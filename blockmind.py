import argparse
from .fetcher import fetch_transactions
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd


def extract_features(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame([{
            "tx_count": 0,
            "avg_interval": 9999,
            "night_tx_ratio": 0,
            "out_tx_ratio": 0,
        }])

    df = df.sort_values("timeStamp")
    df["delta"] = df["timeStamp"].diff().dt.total_seconds().fillna(0)
    df["hour"] = df["timeStamp"].dt.hour
    df["is_night"] = df["hour"].apply(lambda x: int(x < 6 or x > 22))
    df["is_out"] = df["from"].str.lower() == df["from"].str.lower()

    features = {
        "tx_count": len(df),
        "avg_interval": df["delta"].mean(),
        "night_tx_ratio": df["is_night"].mean(),
        "out_tx_ratio": (df["from"] == df.iloc[0]["from"]).mean()
    }

    return pd.DataFrame([features])


def dummy_model():
    # Это лишь демонстрация, в реальности нужна обученная модель
    model = RandomForestClassifier()
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    return model


def main():
    parser = argparse.ArgumentParser(description="Predict wallet activity")
    parser.add_argument("address", help="Ethereum wallet address")
    parser.add_argument("--api-key", required=True, help="Etherscan API key")

    args = parser.parse_args()
    df = fetch_transactions(args.address, args.api_key)
    features = extract_features(df)

    model = dummy_model()
    pred = model.predict_proba(features)[0][1]

    print(f"🧠 Probability of activity in next 7 days: {pred * 100:.2f}%")

if __name__ == "__main__":
    main()
