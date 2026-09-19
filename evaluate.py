"""Evaluate a fine-tuned mT5-small checkpoint on slot extraction."""

import json
import torch
from transformers import MT5ForConditionalGeneration, AutoTokenizer

from train import format_input, get_device


def evaluate(model_dir, test_data_path, prompt_mode, max_examples=None):
    """Run inference and compute exact-match + field-level accuracy.

    Returns:
        metrics  – dict with exact_match, field_accuracy, counts
        predictions – list of per-example dicts for qualitative review
    """
    device = get_device()

    dtype = torch.bfloat16 if device.type in ("mps", "cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = MT5ForConditionalGeneration.from_pretrained(model_dir, torch_dtype=dtype)
    model.to(device)
    model.eval()

    with open(test_data_path) as f:
        test_data = json.load(f)
    if max_examples:
        test_data = test_data[:max_examples]

    exact_matches = 0
    field_correct = 0
    field_total = 0
    predictions = []

    for item in test_data:
        input_text = format_input(item["input"], prompt_mode)
        inputs = tokenizer(
            input_text, return_tensors="pt", max_length=64, truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=64)

        pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        target_json = item["target"]
        target_text = json.dumps(target_json, sort_keys=True)

        # Try to parse prediction as JSON
        try:
            pred_json = json.loads(pred_text)
            pred_normalized = json.dumps(pred_json, sort_keys=True)
        except (json.JSONDecodeError, TypeError):
            pred_json = {}
            pred_normalized = pred_text

        # Exact match (normalized JSON string comparison)
        is_exact = pred_normalized == target_text
        if is_exact:
            exact_matches += 1

        # Field-level accuracy (recall over target fields)
        for key, val in target_json.items():
            field_total += 1
            if pred_json.get(key) == val:
                field_correct += 1

        predictions.append({
            "input": item["input"],
            "target": target_text,
            "prediction": pred_text,
            "exact_match": is_exact,
        })

    n = len(test_data)
    metrics = {
        "exact_match": exact_matches / n if n else 0,
        "field_accuracy": field_correct / field_total if field_total else 0,
        "exact_match_count": exact_matches,
        "total": n,
    }
    return metrics, predictions


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to checkpoint directory")
    p.add_argument("--data", required=True, help="Path to test JSON file")
    p.add_argument("--mode", choices=["bare", "schema"], required=True)
    args = p.parse_args()

    metrics, _ = evaluate(args.model, args.data, args.mode)
    print(json.dumps(metrics, indent=2))
