from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path


NUMERIC_FIELDS = [
    "age",
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "payment_delay_days",
    "num_products",
    "satisfaction_score",
    "used_mobile_app",
    "is_active_recently",
]

CATEGORICAL_FIELDS = ["contract_type", "internet_type"]
TARGET_FIELD = "churned"
IDENTIFIER_FIELD = "customer_id"


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def generate_customer_records(size: int = 320, seed: int = 7) -> list[dict[str, object]]:
    rng = random.Random(seed)
    records: list[dict[str, object]] = []

    for index in range(1, size + 1):
        contract_type = rng.choices(
            population=["monthly", "quarterly", "annual"],
            weights=[0.58, 0.14, 0.28],
            k=1,
        )[0]
        internet_type = rng.choices(
            population=["fiber", "cable", "mobile"],
            weights=[0.52, 0.28, 0.20],
            k=1,
        )[0]

        tenure_months = rng.randint(1, 72)
        monthly_charges = round(
            rng.uniform(35.0, 120.0)
            + (2.5 if internet_type == "fiber" else 0.0)
            - (7.0 if contract_type == "annual" else 0.0),
            2,
        )
        support_tickets = min(8, max(0, int(rng.gauss(2.2, 1.6))))
        payment_delay_days = min(35, max(0, int(rng.gauss(7.0, 6.0))))
        num_products = rng.randint(1, 4)
        satisfaction_score = min(5, max(1, int(round(rng.gauss(3.1, 1.1)))))
        used_mobile_app = 1 if rng.random() < 0.72 else 0
        is_active_recently = 1 if rng.random() < 0.68 else 0
        age = rng.randint(21, 67)

        logit = -0.15
        logit += 0.03 * monthly_charges
        logit -= 0.04 * tenure_months
        logit += 0.58 * support_tickets
        logit += 0.09 * payment_delay_days
        logit -= 0.35 * num_products
        logit -= 0.85 * satisfaction_score
        logit -= 0.9 * used_mobile_app
        logit -= 1.15 * is_active_recently
        logit += 1.1 if contract_type == "monthly" else 0.2 if contract_type == "quarterly" else -0.65
        logit += 0.28 if internet_type == "fiber" else 0.1 if internet_type == "mobile" else -0.18
        logit += rng.gauss(0.0, 0.3)

        churn_probability = sigmoid(logit)
        decision_margin = churn_probability + rng.uniform(-0.03, 0.03)
        churned = 1 if decision_margin >= 0.5 else 0

        records.append(
            {
                "customer_id": f"CUST-{index:04d}",
                "age": age,
                "tenure_months": tenure_months,
                "monthly_charges": monthly_charges,
                "support_tickets": support_tickets,
                "payment_delay_days": payment_delay_days,
                "num_products": num_products,
                "satisfaction_score": satisfaction_score,
                "contract_type": contract_type,
                "internet_type": internet_type,
                "used_mobile_app": used_mobile_app,
                "is_active_recently": is_active_recently,
                "churned": churned,
            }
        )

    return records


def export_records_to_csv(records: list[dict[str, object]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [IDENTIFIER_FIELD] + NUMERIC_FIELDS[:-2] + CATEGORICAL_FIELDS + NUMERIC_FIELDS[-2:] + [TARGET_FIELD]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return path


def load_customer_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
