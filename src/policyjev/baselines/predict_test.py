"""Final-model inference on the labeled test rows: writes probabilities only.

The test labels file is read only to find which test rows were scored (rows labeled -1 are
excluded); label values are never used here. No metrics and no thresholds are computed.
"""

import argparse
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from policyjev.baselines.distilbert import TextDataset, collate, predict, tokenize
from policyjev.config import LABELS, REPO_ROOT, load_config
from policyjev.evaluation.splits import load_labeled_frame, require_split_access
from policyjev.freeze_manifest import verify_entries, verify_model_folder

OUTPUT_NAMES = {"logreg": "logreg_word", "distilbert": "distilbert"}
LOGREG_FILE = "logreg_word.joblib"


def predict_logreg(model_path: Path, texts: list[str]) -> np.ndarray:
    """Probabilities of the saved TF-IDF + logistic regression model, shape (n, 6)."""
    saved = joblib.load(model_path)
    if list(saved["labels"]) != LABELS:
        raise ValueError(f"{model_path.name} was trained with a different label order")
    return np.asarray(saved["classifier"].predict_proba(saved["vectorizer"].transform(texts)))


def resolve_device(device: str) -> torch.device:
    """Map 'auto' to cuda when available, else cpu."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def predict_distilbert(
    model_dir: Path,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: str = "auto",
) -> np.ndarray:
    """Sigmoid probabilities of a saved DistilBERT model, shape (n, 6), in the order of texts.

    Texts are tokenized exactly as in training (truncation to max_length) and batched by
    length for speed; the original order is restored before returning.
    """
    torch_device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(torch_device)
    if model.config.num_labels != len(LABELS):
        raise ValueError(f"model has {model.config.num_labels} outputs, expected {len(LABELS)}")

    token_ids = tokenize(texts, tokenizer, max_length)
    order = np.argsort([len(ids) for ids in token_ids], kind="stable")
    dataset = TextDataset(
        [token_ids[i] for i in order], np.zeros((len(order), len(LABELS)), dtype=np.float32)
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate, pad_id=tokenizer.pad_token_id),
    )
    sorted_probs = predict(model, loader, torch_device)
    probs = np.empty_like(sorted_probs)
    probs[order] = sorted_probs
    return probs


def main() -> None:
    """Write artifacts/predictions/<system>_test.csv for the chosen baseline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=sorted(OUTPUT_NAMES), required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="logreg: folder holding logreg_word.joblib; distilbert: the saved model folder",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    require_split_access("test")

    config = load_config()
    test = load_labeled_frame("test")[["id", "comment_text"]]
    ids = test["id"].tolist()
    texts = test["comment_text"].tolist()

    if args.system == "logreg":
        model_dir = args.model_dir or REPO_ROOT / "artifacts" / "models"
        verify_entries([(f"artifacts/models/{LOGREG_FILE}", model_dir / LOGREG_FILE)])
        probs = predict_logreg(model_dir / LOGREG_FILE, texts)
    else:
        model_dir = args.model_dir or REPO_ROOT / "artifacts" / "models" / "distilbert"
        verify_model_folder("artifacts/models/distilbert", model_dir)
        cfg = config["distilbert"]
        probs = predict_distilbert(
            model_dir, texts, cfg["max_length"], cfg["eval_batch_size"], args.device
        )

    predictions = pd.DataFrame(probs, columns=LABELS)
    predictions.insert(0, "id", ids)
    out_dir = REPO_ROOT / "artifacts" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{OUTPUT_NAMES[args.system]}_test.csv"
    predictions.to_csv(out_path, index=False, float_format="%.8g")
    print(f"wrote {len(predictions)} test rows to {out_path}")


if __name__ == "__main__":
    main()
