from __future__ import annotations

from dataclasses import dataclass

from .dataset import CATEGORICAL_FIELDS, NUMERIC_FIELDS


@dataclass(slots=True)
class FeatureEncoder:
    numeric_fields: list[str]
    categorical_fields: list[str]
    means: dict[str, float]
    stds: dict[str, float]
    category_values: dict[str, list[str]]
    feature_names: list[str]

    @classmethod
    def fit(cls, records: list[dict[str, object]]) -> "FeatureEncoder":
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        category_values: dict[str, list[str]] = {}
        feature_names: list[str] = []

        for field in NUMERIC_FIELDS:
            values = [float(record[field]) for record in records]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            means[field] = mean
            stds[field] = variance ** 0.5 or 1.0
            feature_names.append(field)

        for field in CATEGORICAL_FIELDS:
            values = sorted({str(record[field]) for record in records})
            category_values[field] = values
            feature_names.extend(f"{field}={value}" for value in values[1:])

        return cls(
            numeric_fields=list(NUMERIC_FIELDS),
            categorical_fields=list(CATEGORICAL_FIELDS),
            means=means,
            stds=stds,
            category_values=category_values,
            feature_names=feature_names,
        )

    def transform_record(self, record: dict[str, object]) -> list[float]:
        vector: list[float] = []

        for field in self.numeric_fields:
            raw_value = float(record[field])
            scaled = (raw_value - self.means[field]) / self.stds[field]
            vector.append(scaled)

        for field in self.categorical_fields:
            raw_value = str(record[field])
            categories = self.category_values[field]
            vector.extend(1.0 if raw_value == category else 0.0 for category in categories[1:])

        return vector

    def transform(self, records: list[dict[str, object]]) -> list[list[float]]:
        return [self.transform_record(record) for record in records]
