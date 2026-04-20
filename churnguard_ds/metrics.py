from __future__ import annotations


def accuracy_score(actual: list[int], predicted: list[int]) -> float:
    correct = sum(1 for truth, guess in zip(actual, predicted) if truth == guess)
    return correct / len(actual) if actual else 0.0


def precision_recall_f1(actual: list[int], predicted: list[int]) -> tuple[float, float, float]:
    true_positive = sum(1 for truth, guess in zip(actual, predicted) if truth == 1 and guess == 1)
    false_positive = sum(1 for truth, guess in zip(actual, predicted) if truth == 0 and guess == 1)
    false_negative = sum(1 for truth, guess in zip(actual, predicted) if truth == 1 and guess == 0)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def confusion_matrix(actual: list[int], predicted: list[int]) -> dict[str, int]:
    return {
        "tn": sum(1 for truth, guess in zip(actual, predicted) if truth == 0 and guess == 0),
        "fp": sum(1 for truth, guess in zip(actual, predicted) if truth == 0 and guess == 1),
        "fn": sum(1 for truth, guess in zip(actual, predicted) if truth == 1 and guess == 0),
        "tp": sum(1 for truth, guess in zip(actual, predicted) if truth == 1 and guess == 1),
    }


def roc_auc_score(actual: list[int], probabilities: list[float]) -> float:
    positives = [score for truth, score in zip(actual, probabilities) if truth == 1]
    negatives = [score for truth, score in zip(actual, probabilities) if truth == 0]
    if not positives or not negatives:
        return 0.0

    wins = 0.0
    comparisons = 0
    for positive_score in positives:
        for negative_score in negatives:
            comparisons += 1
            if positive_score > negative_score:
                wins += 1.0
            elif positive_score == negative_score:
                wins += 0.5
    return wins / comparisons

