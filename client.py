from typing import Dict, Any, List, Optional

class LabelNoiseConfidentLearningPruner:
    """
    Identifies mislabeled examples in training datasets by comparing noisy given labels
    against model out-of-fold predicted probability vectors using confident thresholds.
    """
    def identify_label_issues(
        self,
        samples: List[Dict[str, Any]],
        class_names: List[str]
    ) -> Dict[str, Any]:
        # Step 1: Compute confident threshold per class (average predicted prob for assigned class)
        thresholds = {}
        for c_idx, c_name in enumerate(class_names):
            class_probs = [s["predicted_probs"][c_idx] for s in samples if s.get("given_label") == c_name]
            thresholds[c_name] = sum(class_probs) / max(1, len(class_probs)) if class_probs else 0.5

        flagged_samples = []
        clean_samples = []

        # Step 2: Identify label errors where predicted probability for another class exceeds threshold
        for sample in samples:
            given = sample.get("given_label")
            probs = sample.get("predicted_probs", [])
            max_prob = max(probs)
            pred_idx = probs.index(max_prob)
            pred_class = class_names[pred_idx]

            is_issue = (pred_class != given) and (max_prob >= thresholds.get(pred_class, 0.5))

            if is_issue:
                flagged_samples.append({
                    "id": sample.get("id"),
                    "given_label": given,
                    "recommended_true_label": pred_class,
                    "confidence_prob": max_prob,
                    "reason": f"Predicted '{pred_class}' ({max_prob:.2f}) exceeds class threshold ({thresholds.get(pred_class):.2f})"
                })
            else:
                clean_samples.append(sample.get("id"))

        return {
            "total_samples": len(samples),
            "clean_samples_count": len(clean_samples),
            "label_issues_found": len(flagged_samples),
            "contamination_rate_pct": round((len(flagged_samples) / max(1, len(samples))) * 100, 2),
            "class_thresholds": {k: round(v, 3) for k, v in thresholds.items()},
            "flagged_issues": flagged_samples
        }
