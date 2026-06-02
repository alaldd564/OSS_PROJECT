from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
            "n_estimators": [100, 200, 300],
            "max_depth": [4, 6, 8, None],
            "min_samples_split": [2, 4, 6],
        },
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    return search


def main() -> None:
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
    output_path.write_text(
        "\n".join(
            [
                "OSS ML Assignment Results",
                f"Best model: {best_result.model_name}",
                f"Accuracy: {best_result.accuracy:.4f}",
                f"CV accuracy: {best_result.cv_accuracy:.4f}",
                "",
                "Classification report:",
                best_result.report,
                "",
                "Details:",
                best_result.details,
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
