import pickle
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_PATH = "best_model.pkl"

def main():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    data = load_wine()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    samples = X_test[:5]
    preds = model.predict(samples)

    print("Loaded model:", type(model))
    print("\nSample predictions (first 5 test rows):")
    for i, (x, p) in enumerate(zip(samples, preds)):
        true = y_test[i]
        print(f"#{i} true={true} pred={p}")

    # Try predict_proba if available
    try:
        probs = model.predict_proba(samples)
        print("\nPredicted probabilities:")
        for i, pr in enumerate(probs):
            print(f"#{i}: {np.round(pr,3)}")
    except Exception:
        print("\nModel does not support predict_proba()")

    # Full test set report
    try:
        full_preds = model.predict(X_test)
        print("\nClassification report on test set:")
        print(classification_report(y_test, full_preds, digits=4))
    except Exception as e:
        print("\nCould not run full test set evaluation:", e)

if __name__ == '__main__':
    main()
