from __future__ import annotations

import math


def sigmoid(value: float) -> float:
    bounded = max(-35.0, min(35.0, value))
    return 1.0 / (1.0 + math.exp(-bounded))


class LogisticRegressionScratch:
    def __init__(self, learning_rate: float = 0.05, epochs: int = 900, l2: float = 0.001) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.weights: list[float] = []
        self.bias: float = 0.0

    def fit(self, features: list[list[float]], labels: list[int]) -> "LogisticRegressionScratch":
        if not features:
            raise ValueError("Features cannot be empty.")

        feature_count = len(features[0])
        sample_count = len(features)
        self.weights = [0.0] * feature_count
        self.bias = 0.0

        for _ in range(self.epochs):
            gradient_weights = [0.0] * feature_count
            gradient_bias = 0.0

            for row, label in zip(features, labels):
                probability = sigmoid(self._linear_score(row))
                error = probability - label
                gradient_bias += error
                for index, value in enumerate(row):
                    gradient_weights[index] += error * value

            scale = 1.0 / sample_count
            for index in range(feature_count):
                regularization = self.l2 * self.weights[index]
                self.weights[index] -= self.learning_rate * ((gradient_weights[index] * scale) + regularization)
            self.bias -= self.learning_rate * gradient_bias * scale

        return self

    def _linear_score(self, row: list[float]) -> float:
        return sum(weight * value for weight, value in zip(self.weights, row)) + self.bias

    def predict_probabilities(self, features: list[list[float]]) -> list[float]:
        return [sigmoid(self._linear_score(row)) for row in features]

    def predict(self, features: list[list[float]], threshold: float = 0.5) -> list[int]:
        return [1 if probability >= threshold else 0 for probability in self.predict_probabilities(features)]

