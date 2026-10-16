"""Latency: local inference for the baselines, recorded API calls for JEV.

Baselines (A, B): p50 and p95 per-comment latency at batch size 1 and throughput at batch size
32, on a fixed random sample of validation comments, after warm-up, on CPU and on GPU when
available. JEV (C): p50 and p95 request latency and the number of successful requests from the recorded
calls. JEV latency is a network round trip including queueing under concurrent requests, so it
is reported but not ranked against local inference. Comment text is never written to disk.
"""

import argparse
import platform
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from policyjev.config import REPO_ROOT, load_config
from policyjev.evaluation.splits import SPLITS, load_labeled_frame, require_split_access
from policyjev.jev.cache import JevCache

SAMPLE_SIZE = 500
WARMUP_CALLS = 10
THROUGHPUT_BATCH = 32


def sample_texts(texts: Sequence[str], n: int, seed: int) -> list[str]:
    """A fixed seeded random sample of n texts (all of them if fewer are available)."""
    chosen = np.random.default_rng(seed).choice(len(texts), size=min(n, len(texts)), replace=False)
    return [texts[i] for i in chosen]


def single_latency_ms(
    predict_one: Callable[[str], object],
    texts: Sequence[str],
    warmup: int = WARMUP_CALLS,
    clock: Callable[[], float] = time.perf_counter,
    sync: Callable[[], None] = lambda: None,
) -> tuple[float, float]:
    """p50 and p95 per-comment latency in milliseconds at batch size 1, after warm-up."""
    for text in list(texts)[:warmup]:
        predict_one(text)
    sync()
    durations = []
    for text in texts:
        started = clock()
        predict_one(text)
        sync()
        durations.append((clock() - started) * 1000.0)
    return float(np.percentile(durations, 50)), float(np.percentile(durations, 95))


def throughput_per_second(
    predict_batch: Callable[[list[str]], object],
    texts: Sequence[str],
    batch_size: int = THROUGHPUT_BATCH,
    clock: Callable[[], float] = time.perf_counter,
    sync: Callable[[], None] = lambda: None,
) -> float:
    """Comments per second when scoring the texts in batches of batch_size."""
    texts = list(texts)
    predict_batch(texts[:batch_size])
    sync()
    started = clock()
    for start in range(0, len(texts), batch_size):
        predict_batch(texts[start : start + batch_size])
    sync()
    elapsed = clock() - started
    return len(texts) / elapsed if elapsed > 0 else float("nan")


def jev_latency_cost(cache: JevCache) -> dict:
    """Request latency percentiles and the number of successful requests from the recorded calls."""
    n_ok = cache.count_ok_calls()
    p50, p95 = cache.latency_percentiles()
    return {
        "n_requests": n_ok,
        "latency_p50_ms": p50 * 1000.0,
        "latency_p95_ms": p95 * 1000.0,
    }


def machine_info(device: str) -> dict:
    """Describe the machine and library versions used for a measurement."""
    import sklearn
    import torch

    if device == "cuda":
        name = torch.cuda.get_device_name(0)
    else:
        name = platform.processor() or platform.machine()
    return {
        "device": device,
        "device_name": name,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
    }


def measure_logreg(model_path: Path, texts: list[str]) -> dict:
    """Latency and throughput of the saved logistic regression on CPU."""
    import joblib

    saved = joblib.load(model_path)
    vectorizer, classifier = saved["vectorizer"], saved["classifier"]

    def predict_batch(batch: list[str]):
        return classifier.predict_proba(vectorizer.transform(batch))

    p50, p95 = single_latency_ms(lambda t: predict_batch([t]), texts)
    return {
        "system": "logreg_word",
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "throughput_per_s_batch32": throughput_per_second(predict_batch, texts),
        **machine_info("cpu"),
    }


def measure_distilbert(model_dir: Path, texts: list[str], max_length: int, device: str) -> dict:
    """Latency and throughput of the saved DistilBERT model on one device."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch_device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(torch_device).eval()

    @torch.no_grad()
    def predict_batch(batch: list[str]):
        encoded = tokenizer(
            batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt"
        ).to(torch_device)
        return torch.sigmoid(model(**encoded).logits).cpu()

    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    p50, p95 = single_latency_ms(lambda t: predict_batch([t]), texts, sync=sync)
    return {
        "system": "distilbert",
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "throughput_per_s_batch32": throughput_per_second(predict_batch, texts, sync=sync),
        **machine_info(device),
    }


def main() -> None:
    """Write artifacts/metrics/latency_cost.csv."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS, default="validation")
    args = parser.parse_args()
    require_split_access(args.split)

    config = load_config()
    frame = load_labeled_frame(args.split)
    texts = sample_texts(frame["comment_text"].tolist(), SAMPLE_SIZE, config["seed"])
    models_dir = REPO_ROOT / "artifacts" / "models"
    rows = []

    logreg_path = models_dir / "logreg_word.joblib"
    if logreg_path.exists():
        rows.append(measure_logreg(logreg_path, texts))
    else:
        print(f"skipped logistic regression: {logreg_path} not found")

    distilbert_dir = models_dir / "distilbert"
    if distilbert_dir.is_dir():
        import torch

        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            rows.append(
                measure_distilbert(distilbert_dir, texts, config["distilbert"]["max_length"], device)
            )
    else:
        print(f"skipped DistilBERT: {distilbert_dir} not found; copy the saved model folder here")

    with JevCache(REPO_ROOT / config["jev"]["cache_path"]) as cache:
        rows.append({"system": "jev", **jev_latency_cost(cache), "device": "api"})

    table = pd.DataFrame(rows)
    table["n_comments"] = np.where(table["system"] == "jev", np.nan, len(texts))
    metrics_dir = REPO_ROOT / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(metrics_dir / "latency_cost.csv", index=False)

    print(
        f"{len(texts)} {args.split} comments (seed {config['seed']}); baselines have no API cost, "
        "so only compute time is reported for them."
    )
    print("JEV latency is a network round trip under concurrent requests and is not ranked against local inference.")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
