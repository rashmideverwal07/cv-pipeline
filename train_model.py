import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

DATA_FILE  = "gesture_data.csv"
MODEL_FILE = "gesture_model.pkl"

def train():
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found. Run collect_data.py first!")
        return

    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    print(f"Total samples: {len(df)}")
    print(f"Gestures found: {df['label'].unique()}")
    print(f"Samples per gesture:\n{df['label'].value_counts()}\n")

    X = df.drop("label", axis=1).to_numpy()
    y = df["label"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1        # use all CPU cores
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc * 100:.2f}%")
    print("\nDetailed report:")
    print(classification_report(y_test, y_pred))

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    print(f"\nModel saved to {MODEL_FILE}")
    print("Now run: python main.py  (predictions will show live!)")


if __name__ == "__main__":
    train()