from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def _to_2d(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def _to_1d(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _standardize(train_features: np.ndarray, test_features: np.ndarray):
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (train_features - mean) / std, (test_features - mean) / std


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    x_train, x_test = _standardize(_to_2d(data.train_features), _to_2d(data.test_features))
    y_train = _to_1d(data.train_target)

    n_samples, n_features = x_train.shape
    weights = np.zeros(n_features, dtype=float)
    bias = 0.0
    loss_history: list[float] = []

    for _ in range(data.epochs):
        predictions = x_train @ weights + bias
        errors = predictions - y_train
        loss_history.append(float(np.mean(errors**2)))
        weights -= data.learning_rate * (2.0 / n_samples) * (x_train.T @ errors)
        bias -= data.learning_rate * (2.0 / n_samples) * float(errors.sum())

    test_predictions = x_test @ weights + bias
    return RegressionResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        predictions=test_predictions,
    )


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    x_train, x_test = _standardize(_to_2d(data.train_features), _to_2d(data.test_features))
    y_train = _to_1d(data.train_target)

    n_samples, n_features = x_train.shape
    weights = np.zeros(n_features, dtype=float)
    bias = 0.0
    loss_history: list[float] = []
    eps = 1e-12

    for _ in range(data.epochs):
        probabilities = _sigmoid(x_train @ weights + bias)
        loss = -np.mean(
            y_train * np.log(probabilities + eps)
            + (1.0 - y_train) * np.log(1.0 - probabilities + eps)
        )
        loss_history.append(float(loss))
        errors = probabilities - y_train
        weights -= data.learning_rate * (1.0 / n_samples) * (x_train.T @ errors)
        bias -= data.learning_rate * (1.0 / n_samples) * float(errors.sum())

    test_probabilities = _sigmoid(x_test @ weights + bias)
    test_predictions = (test_probabilities >= 0.5).astype(int)
    return ClassificationResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        probabilities=test_probabilities,
        predictions=test_predictions,
    )


def _classification_metrics(y_true, probabilities, predictions) -> ClassificationMetrics:
    y_true = _to_1d(y_true).astype(int)
    probabilities = _to_1d(probabilities)
    predictions = _to_1d(predictions).astype(int)
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        accuracy=float(accuracy_score(y_true, predictions)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
    )


def _metrics_dict(metrics: ClassificationMetrics) -> dict[str, float]:
    return {
        "roc_auc": metrics.roc_auc,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
    }


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните вашу логистическую регрессию, Decision Tree и KNN на одном split."""
    own = train_logistic_classifier(data)
    own_metrics = _classification_metrics(data.test_target, own.probabilities, own.predictions)

    x_train, x_test = _standardize(_to_2d(data.train_features), _to_2d(data.test_features))
    y_train = _to_1d(data.train_target).astype(int)
    y_test = _to_1d(data.test_target).astype(int)

    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(x_train, y_train)
    tree_proba = tree.predict_proba(x_test)[:, 1]
    tree_pred = tree.predict(x_test)

    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    knn_proba = knn.predict_proba(x_test)[:, 1]
    knn_pred = knn.predict(x_test)

    return BaselineComparisonResult(
        logistic_regression=own_metrics,
        decision_tree=_classification_metrics(y_test, tree_proba, tree_pred),
        knn=_classification_metrics(y_test, knn_proba, knn_pred),
    )


def _prepare_student_performance():
    frame = pd.read_csv(Path(__file__).with_name("Student_Performance.csv"))
    frame["Extracurricular Activities"] = (
        frame["Extracurricular Activities"].map({"Yes": 1, "No": 0}).astype(float)
    )
    features = frame.drop(columns=["Performance Index"])
    target = frame["Performance Index"]
    return train_test_split(features, target, test_size=0.2, random_state=42)


def _prepare_german_credit():
    frame = pd.read_csv(Path(__file__).with_name("german.csv"), sep=";")
    features = frame.drop(columns=["Creditability"])
    target = frame["Creditability"]
    return train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )


def run_experiments(output_path: str | Path = "experiment_results.json") -> dict:
    x_train, x_test, y_train, y_test = _prepare_student_performance()
    regression_data = RegressionInput(
        train_features=x_train,
        train_target=y_train,
        test_features=x_test,
        test_target=y_test,
    )
    regression = train_linear_regression(regression_data)
    test_mse = float(mean_squared_error(_to_1d(y_test), regression.predictions))

    gx_train, gx_test, gy_train, gy_test = _prepare_german_credit()
    classification_data = ClassificationInput(
        train_features=gx_train,
        train_target=gy_train,
        test_features=gx_test,
        test_target=gy_test,
    )
    comparison = compare_classical_models(classification_data)

    x_train_scaled, x_test_scaled = _standardize(_to_2d(gx_train), _to_2d(gx_test))
    sklearn_logreg = LogisticRegression(max_iter=1000, random_state=42)
    sklearn_logreg.fit(x_train_scaled, _to_1d(gy_train).astype(int))
    sklearn_proba = sklearn_logreg.predict_proba(x_test_scaled)[:, 1]
    sklearn_pred = sklearn_logreg.predict(x_test_scaled)
    sklearn_metrics = _classification_metrics(gy_test, sklearn_proba, sklearn_pred)

    results = {
        "linear_regression": {"test_mse": test_mse},
        "classification": {
            "own_logistic": _metrics_dict(comparison.logistic_regression),
            "logistic_regression": _metrics_dict(sklearn_metrics),
            "decision_tree": _metrics_dict(comparison.decision_tree),
            "knn": _metrics_dict(comparison.knn),
        },
    }

    path = Path(output_path)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


if __name__ == "__main__":
    print(json.dumps(run_experiments(), indent=2, ensure_ascii=False))
