from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from churnguard_ds.dataset import export_records_to_csv, generate_customer_records
from churnguard_ds.pipeline import ChurnTrainer, train_test_split


class ChurnGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = generate_customer_records(seed=7)
        cls.train_records, cls.test_records = train_test_split(cls.records)
        cls.trainer = ChurnTrainer().fit(cls.train_records)

    def test_model_reaches_reasonable_quality(self) -> None:
        report = self.trainer.evaluate(self.test_records)

        self.assertGreaterEqual(report.accuracy, 0.8)
        self.assertGreaterEqual(report.f1, 0.75)
        self.assertGreaterEqual(report.roc_auc, 0.85)

    def test_high_risk_customer_scores_above_low_risk_customer(self) -> None:
        high_risk_customer = {
            "age": 28,
            "tenure_months": 3,
            "monthly_charges": 102.0,
            "support_tickets": 6,
            "payment_delay_days": 22,
            "num_products": 1,
            "satisfaction_score": 1,
            "contract_type": "monthly",
            "internet_type": "fiber",
            "used_mobile_app": 0,
            "is_active_recently": 0,
        }
        low_risk_customer = {
            "age": 45,
            "tenure_months": 48,
            "monthly_charges": 58.0,
            "support_tickets": 0,
            "payment_delay_days": 0,
            "num_products": 3,
            "satisfaction_score": 5,
            "contract_type": "annual",
            "internet_type": "cable",
            "used_mobile_app": 1,
            "is_active_recently": 1,
        }

        high_risk_prediction = self.trainer.predict_customer(high_risk_customer)
        low_risk_prediction = self.trainer.predict_customer(low_risk_customer)

        self.assertGreater(high_risk_prediction.probability, low_risk_prediction.probability)
        self.assertEqual(high_risk_prediction.risk_band, "High")
        self.assertEqual(low_risk_prediction.risk_band, "Low")

    def test_dataset_export_writes_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "customers.csv"
            export_records_to_csv(self.records[:10], output_path)

            self.assertTrue(output_path.exists())
            content = output_path.read_text(encoding="utf-8")
            self.assertIn("customer_id", content)
            self.assertIn("churned", content)


if __name__ == "__main__":
    unittest.main()
