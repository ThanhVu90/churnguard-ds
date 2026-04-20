from __future__ import annotations

import argparse

from .dataset import export_records_to_csv, generate_customer_records, load_customer_json
from .pipeline import ChurnTrainer, build_default_report, train_test_split


def handle_demo(_: argparse.Namespace) -> None:
    trainer, report = build_default_report()

    print("ChurnGuard DS")
    print("Model: Logistic Regression (from scratch)")
    print()
    print(f"Accuracy : {report.accuracy}")
    print(f"Precision: {report.precision}")
    print(f"Recall   : {report.recall}")
    print(f"F1 Score : {report.f1}")
    print(f"ROC AUC  : {report.roc_auc}")
    print()
    print("Most influential features")
    for name, weight in report.top_features[:5]:
        print(f"- {name}: {weight}")

    sample_customer = {
        "age": 29,
        "tenure_months": 5,
        "monthly_charges": 99.0,
        "support_tickets": 4,
        "payment_delay_days": 16,
        "num_products": 1,
        "satisfaction_score": 2,
        "contract_type": "monthly",
        "internet_type": "fiber",
        "used_mobile_app": 0,
        "is_active_recently": 0,
    }
    prediction = trainer.predict_customer(sample_customer)
    print()
    print("Sample customer risk")
    print(f"- Probability: {prediction.probability}")
    print(f"- Risk band  : {prediction.risk_band}")
    print(f"- Risk drivers: {', '.join(prediction.top_risk_drivers) or '-'}")
    print(f"- Retention drivers: {', '.join(prediction.top_retention_drivers) or '-'}")


def handle_evaluate(_: argparse.Namespace) -> None:
    _, report = build_default_report()
    print("ChurnGuard DS Evaluation")
    print(f"Accuracy : {report.accuracy}")
    print(f"Precision: {report.precision}")
    print(f"Recall   : {report.recall}")
    print(f"F1 Score : {report.f1}")
    print(f"ROC AUC  : {report.roc_auc}")
    print(f"Confusion: {report.confusion}")


def handle_predict(args: argparse.Namespace) -> None:
    records = generate_customer_records(seed=args.seed)
    train_records, _ = train_test_split(records)
    trainer = ChurnTrainer().fit(train_records)

    customer = load_customer_json(args.json_file)
    prediction = trainer.predict_customer(customer)

    print("ChurnGuard DS Prediction")
    print(f"Input file : {args.json_file}")
    print(f"Probability: {prediction.probability}")
    print(f"Risk band  : {prediction.risk_band}")
    print(f"Risk drivers: {', '.join(prediction.top_risk_drivers) or '-'}")
    print(f"Retention drivers: {', '.join(prediction.top_retention_drivers) or '-'}")


def handle_export_data(args: argparse.Namespace) -> None:
    records = generate_customer_records(size=args.size, seed=args.seed)
    path = export_records_to_csv(records, args.output)
    print(f"Exported {len(records)} records to {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Customer churn prediction project for portfolio usage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="Run the churn modeling demo.")
    demo_parser.set_defaults(func=handle_demo)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the churn model.")
    evaluate_parser.set_defaults(func=handle_evaluate)

    predict_parser = subparsers.add_parser("predict", help="Predict churn risk for a JSON customer record.")
    predict_parser.add_argument("--json-file", required=True, help="Path to a JSON file with customer fields.")
    predict_parser.add_argument("--seed", type=int, default=7, help="Seed used for training data generation.")
    predict_parser.set_defaults(func=handle_predict)

    export_parser = subparsers.add_parser("export-data", help="Export the synthetic dataset to CSV.")
    export_parser.add_argument("--output", required=True, help="Output CSV path.")
    export_parser.add_argument("--size", type=int, default=320, help="Number of records to generate.")
    export_parser.add_argument("--seed", type=int, default=7, help="Seed used for data generation.")
    export_parser.set_defaults(func=handle_export_data)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

