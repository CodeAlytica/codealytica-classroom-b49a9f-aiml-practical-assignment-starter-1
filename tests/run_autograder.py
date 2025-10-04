import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import json

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# 👇 1. Load student's predictions
def load_student_predictions():
    path = "predictions.csv"
    if not os.path.exists(path):
        print("❌ predictions.csv not found. Student must create it.")
        return None
    df = pd.read_csv(path)
    if 'id' not in df.columns or 'target_pred' not in df.columns:
        print("❌ predictions.csv must have 'id' and 'target_pred' columns.")
        return None
    return df

# 👇 2. Create a small hidden 'truth' dataset (for testing)
# In production, you'll replace this with a hidden labels file.
def load_hidden_labels():
    data = {
        'id': [1, 2, 3, 4, 5],
        'target': [10, 20, 30, 40, 50]
    }
    return pd.DataFrame(data)

# 👇 3. Score function (RMSE)
def evaluate(preds, labels):
    df = preds.merge(labels, on='id', how='inner')
    rmse = mean_squared_error(df['target'], df['target_pred'], squared=False)
    score = max(0, 100 - rmse * 5)  # simple scaled score
    return rmse, score

# 👇 4. Main grading logic
def main():
    preds = load_student_predictions()
    if preds is None:
        with open(os.path.join(RESULT_DIR, "score.json"), "w") as f:
            json.dump({"score": 0}, f)
        return

    labels = load_hidden_labels()
    rmse, score = evaluate(preds, labels)

    print(f"✅ RMSE: {rmse:.2f} | Score: {score:.2f}/100")

    result = {
        "rmse": float(rmse),
        "score": float(score)
    }

    with open(os.path.join(RESULT_DIR, "score.json"), "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
