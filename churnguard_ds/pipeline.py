from __future__ import annotations

import random
from dataclasses import dataclass

from .dataset import TARGET_FIELD, generate_customer_records
from .metrics import accuracy_score, confusion_matrix, precision_recall_f1, roc_auc_score
from .model import LogisticRegressionScratch
from .preprocessing import FeatureEncoder


@dataclass(slots=True)
class ChurnPrediction:
    probability: float
    risk_band: str
    top_risk_drivers: list[str]
    top_retention_drivers: list[str]


@dataclass(slots=True)
class ChurnReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: dict[str, int]
    top_features: list[tuple[str, float]]


def train_test_split(
    records: list[dict[str, object]],
    test_ratio: float = 0.25,
    seed: int = 21,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    split_index = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split_index], shuffled[split_index:]


def risk_band(probability: float) -> str:
    if probability >= 0.7:
        return "High"
    if probability >= 0.4:
        return "Medium"
    return "Low"


class ChurnTrainer:
    def __init__(self, learning_rate: float = 0.05, epochs: int = 900) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.encoder: FeatureEncoder | None = None
        self.model: LogisticRegressionScratch | None = None
        self.feature_names: list[str] = []
        self.decision_threshold: float = 0.5

    def fit(self, records: list[dict[str, object]]) -> "ChurnTrainer":
        self.encoder = FeatureEncoder.fit(records)
        self.feature_names = self.encoder.feature_names
        features = self.encoder.transform(records)
        labels = [int(record[TARGET_FIELD]) for record in records]
        self.model = LogisticRegressionScratch(
            learning_rate=self.learning_rate,
            epochs=self.epochs,
        ).fit(features, labels)
        self.decision_threshold = self._select_threshold(features, labels)
        return self

    def evaluate(self, records: list[dict[str, object]]) -> ChurnReport:
        features, labels = self._transform_with_labels(records)
        probabilities = self.model.predict_probabilities(features)
        predictions = self.model.predict(features, threshold=self.decision_threshold)
        precision, recall, f1 = precision_recall_f1(labels, predictions)

        ranked_features = sorted(
            zip(self.feature_names, self.model.weights),
            key=lambda item: abs(item[1]),
            reverse=True,
        )

        return ChurnReport(
            accuracy=round(accuracy_score(labels, predictions), 3),
            precision=round(precision, 3),
            recall=round(recall, 3),
            f1=round(f1, 3),
            roc_auc=round(roc_auc_score(labels, probabilities), 3),
            confusion=confusion_matrix(labels, predictions),
            top_features=[(name, round(weight, 3)) for name, weight in ranked_features[:8]],
        )

    def predict_customer(self, customer: dict[str, object]) -> ChurnPrediction:
        if not self.encoder or not self.model:
            raise ValueError("Model must be fitted before prediction.")

        encoded = self.encoder.transform_record(customer)
        probability = self.model.predict_probabilities([encoded])[0]
        contributions = [
            (name, value * weight)
            for name, value, weight in zip(self.feature_names, encoded, self.model.weights)
        ]
        readable_contributions = [
            (name, contribution)
            for name, contribution in contributions
            if "=" not in name
        ]
        top_risk_drivers = [
            f"{name} ({contribution:.2f})"
            for name, contribution in sorted(readable_contributions, key=lambda item: item[1], reverse=True)
            if contribution > 0
        ][:3]
        top_retention_drivers = [
            f"{name} ({contribution:.2f})"
            for name, contribution in sorted(readable_contributions, key=lambda item: item[1])
            if contribution < 0
        ][:3]

        return ChurnPrediction(
            probability=round(probability, 3),
            risk_band=risk_band(probability),
            top_risk_drivers=top_risk_drivers,
            top_retention_drivers=top_retention_drivers,
        )

    def _transform_with_labels(self, records: list[dict[str, object]]) -> tuple[list[list[float]], list[int]]:
        if not self.encoder or not self.model:
            raise ValueError("Model must be fitted before evaluation.")
        features = self.encoder.transform(records)
        labels = [int(record[TARGET_FIELD]) for record in records]
        return features, labels

    def _select_threshold(self, features: list[list[float]], labels: list[int]) -> float:
        probabilities = self.model.predict_probabilities(features)
        best_threshold = 0.5
        best_f1 = -1.0

        for candidate in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]:
            predictions = [1 if probability >= candidate else 0 for probability in probabilities]
            _, _, f1 = precision_recall_f1(labels, predictions)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = candidate

        return best_threshold


def build_default_report(seed: int = 7) -> tuple[ChurnTrainer, ChurnReport]:
    records = generate_customer_records(seed=seed)
    train_records, test_records = train_test_split(records)
    trainer = ChurnTrainer().fit(train_records)
    report = trainer.evaluate(test_records)
    return trainer, report
