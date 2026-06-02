from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import time
import pickle
import json

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# note: avoid importing loguniform for compatibility across sklearn versions


@dataclass(frozen=True)
class ExperimentResult:
    model_name: str
    accuracy: float
    cv_accuracy: float
    confusion: list[list[int]]
    report: str
    details: str


def build_models() -> dict[str, Pipeline]:
    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        max_depth=6,
                    ),
                )
            ]
        ),
    }


def evaluate_model(
    model_name: str,
    model: Pipeline,
    x_train,
    x_test,
    y_train,
    y_test,
    target_names,
    cv,
) -> ExperimentResult:
    cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=target_names)
    confusion = confusion_matrix(y_test, predictions).tolist()
    details = "\n".join(
        [
            f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})",
        ]
    )
    return ExperimentResult(
        model_name=model_name,
        accuracy=accuracy,
        cv_accuracy=float(cv_scores.mean()),
        confusion=confusion,
        report=report,
        details=details,
    )


def tune_random_forest(x_train, y_train, cv) -> GridSearchCV:
    search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid={
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [4, 6, 8, 10, None],
            "min_samples_split": [2, 4, 6, 8],
        },
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    return search


def randomized_tune_mlp(x_train, y_train, cv, n_iter=50):
    param_dist = {
        "hidden_layer_sizes": [
            (64,),
            (128,),
            (128, 64),
            (200, 200),
            (300, 200, 100),
            (512, 256),
            (512, 512, 256),
        ],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-5, 1e-4, 1e-3, 1e-2],
        "activation": ["relu", "tanh", "logistic"],
        "solver": ["adam", "sgd"],
    }
    rs = RandomizedSearchCV(
        estimator=MLPClassifier(max_iter=3000, random_state=42),
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        random_state=42,
    )
    rs.fit(x_train, y_train)
    return rs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heavy", action="store_true", help="Run extended heavy experiments (long runtime)")
    parser.add_argument(
        "--mlp-random-iter",
        type=int,
        default=10,
        help="Number of RandomizedSearch iterations for MLP in heavy mode (default 10)",
    )
    args = parser.parse_args()

    dataset = load_wine()
    target_names = [str(name) for name in dataset.target_names]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=42,
        stratify=dataset.target,
    )

    results = [
        evaluate_model(name, model, x_train, x_test, y_train, y_test, target_names, cv)
        for name, model in build_models().items()
    ]
    # Optional heavy experiments (long-running) -------------------------------------------------
    if args.heavy:
        start = time.time()
        print("Running heavy experiments: Randomized MLP search and RF GridSearch...")
        mlp_search = randomized_tune_mlp(x_train, y_train, cv, n_iter=args.mlp_random_iter)
        print(f"MLP best params: {mlp_search.best_params_}, CV best: {mlp_search.best_score_:.4f}")
        rf_search = tune_random_forest(x_train, y_train, cv)
        print(f"RF best params: {rf_search.best_params_}, CV best: {rf_search.best_score_:.4f}")

        # stacking ensemble combining best estimators
        estimators = [
            ("rf", rf_search.best_estimator_),
            ("mlp", mlp_search.best_estimator_),
        ]
        stack = StackingClassifier(
            estimators=estimators, final_estimator=LogisticRegression(), n_jobs=-1
        )
        stack.fit(x_train, y_train)
        stack_pred = stack.predict(x_test)
        stack_acc = accuracy_score(y_test, stack_pred)
        stack_report = classification_report(y_test, stack_pred, target_names=target_names)
        stack_conf = confusion_matrix(y_test, stack_pred).tolist()
        results.append(
            ExperimentResult(
                model_name="Stacking(RF+MLP)",
                accuracy=stack_acc,
                cv_accuracy=max(float(rf_search.best_score_), float(mlp_search.best_score_)),
                confusion=stack_conf,
                report=stack_report,
                details=f"RF_best={rf_search.best_params_}, MLP_best={mlp_search.best_params_}",
            )
        )

        # save best stacking model
        model_path = Path("best_model.pkl")
        with model_path.open("wb") as f:
            pickle.dump(stack, f)
        print(f"Saved stacked model to {model_path}")
        elapsed = time.time() - start
        print(f"Heavy experiments completed in {elapsed/60:.2f} minutes")

    else:
        # quick RF GridSearch for demonstration (kept small so default run is fast)
        tuned_search = tune_random_forest(x_train, y_train, cv)
        tuned_predictions = tuned_search.best_estimator_.predict(x_test)
        tuned_accuracy = accuracy_score(y_test, tuned_predictions)
        tuned_report = classification_report(y_test, tuned_predictions, target_names=target_names)
        tuned_confusion = confusion_matrix(y_test, tuned_predictions).tolist()
        tuned_result = ExperimentResult(
            model_name="RandomForest(GridSearchCV)",
            accuracy=tuned_accuracy,
            cv_accuracy=float(tuned_search.best_score_),
            confusion=tuned_confusion,
            report=tuned_report,
            details=f"Best params: {tuned_search.best_params_}",
        )
        results.append(tuned_result)

    best_result = max(results, key=lambda item: item.accuracy)

    print("=== Dataset Information ===")
    print(f"Dataset: Wine classification")
    print(f"Samples: {dataset.data.shape[0]}")
    print(f"Features: {dataset.data.shape[1]}")
    print(f"Classes: {target_names}")
    print()

    print("=== Train/Test Split ===")
    print(f"Train size: {len(x_train)}")
    print(f"Test size: {len(x_test)}")
    print()

    print("=== Model Comparison ===")
    for result in results:
        print(f"Model: {result.model_name}")
        print(f"Accuracy: {result.accuracy:.4f}")
        print(f"CV accuracy: {result.cv_accuracy:.4f}")
        print(f"Confusion matrix: {result.confusion}")
        print(result.details)
        print(result.report)
        print("-" * 60)

    print("=== Best Model ===")
    print(f"Selected model: {best_result.model_name}")
    print(f"Best accuracy: {best_result.accuracy:.4f}")

    output_path = Path("oss_ml_wine_results.txt")
    # write JSON results for programmatic consumption and a human-readable text file
    record = {
        "best_model": best_result.model_name,
        "accuracy": float(best_result.accuracy),
        "cv_accuracy": float(best_result.cv_accuracy),
        "report": best_result.report,
        "details": best_result.details,
    }
    output_json = Path("oss_ml_wine_results.json")
    output_json.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    output_path.write_text(
        "\n".join(["OSS ML Assignment Results", f"Best model: {best_result.model_name}", f"Accuracy: {best_result.accuracy:.4f}", f"CV accuracy: {best_result.cv_accuracy:.4f}", "", "Classification report:", best_result.report, "", "Details:", best_result.details]),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
