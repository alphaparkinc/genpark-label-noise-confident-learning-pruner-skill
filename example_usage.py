import json
from client import LabelNoiseConfidentLearningPruner

def main():
    pruner = LabelNoiseConfidentLearningPruner()
    classes = ["positive", "negative"]
    samples = [
        {"id": "doc_1", "given_label": "positive", "predicted_probs": [0.92, 0.08]},
        {"id": "doc_2", "given_label": "positive", "predicted_probs": [0.88, 0.12]},
        {"id": "doc_3", "given_label": "positive", "predicted_probs": [0.04, 0.96]}, # Mislabeled negative!
        {"id": "doc_4", "given_label": "negative", "predicted_probs": [0.10, 0.90]}
    ]
    result = pruner.identify_label_issues(samples, classes)
    print("Label Noise Triage:")
    print(json.dumps(result, indent=2))
    assert result["label_issues_found"] == 1
    assert result["flagged_issues"][0]["id"] == "doc_3"
    assert result["flagged_issues"][0]["recommended_true_label"] == "negative"
    print("Label noise pruner verification: PASS")

if __name__ == "__main__":
    main()
