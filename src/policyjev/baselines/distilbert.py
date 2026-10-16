"""DistilBERT multi-label fine-tuning baseline (sigmoid outputs, per-label BCE)."""

import argparse
import json
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from policyjev.baselines.logreg import per_label_average_precision
from policyjev.config import LABELS, REPO_ROOT, load_config
from policyjev.data.load import RAW_DIR, load_train
from policyjev.data.split import load_split_ids


def tokenize(texts, tokenizer, max_length: int) -> list[list[int]]:
    """Tokenize texts with truncation to max_length, special tokens included."""
    texts = list(texts)
    input_ids: list[list[int]] = []
    for start in range(0, len(texts), 10000):
        batch = texts[start : start + 10000]
        encoded = tokenizer(
            batch, add_special_tokens=True, truncation=True, max_length=max_length
        )
        input_ids.extend(encoded["input_ids"])
    return input_ids


class TextDataset(Dataset):
    """Pairs of token id lists and label vectors."""

    def __init__(self, input_ids: list[list[int]], labels: np.ndarray) -> None:
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> tuple[list[int], np.ndarray]:
        return self.input_ids[index], self.labels[index]


def collate(
    batch: list[tuple[list[int], np.ndarray]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch to its longest item; return ids, attention mask and float labels."""
    longest = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), longest), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), longest), dtype=torch.long)
    for row, (ids, _) in enumerate(batch):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, : len(ids)] = 1
    labels = torch.tensor(np.stack([label for _, label in batch]), dtype=torch.float32)
    return input_ids, attention_mask, labels


def build_optimizer(
    model: nn.Module, lr: float, weight_decay: float
) -> torch.optim.AdamW:
    """AdamW with no weight decay on bias and LayerNorm parameters."""
    layer_norm_params = {
        id(p)
        for module in model.modules()
        if isinstance(module, nn.LayerNorm)
        for p in module.parameters(recurse=False)
    }
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias") or id(param) in layer_norm_params:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device,
    max_grad_norm: float,
    scaler=None,
) -> float:
    """Train for one epoch with BCE-with-logits and return the mean batch loss."""
    device = torch.device(device)
    use_amp = device.type == "cuda" and scaler is not None
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()

    total_loss = 0.0
    n_batches = 0
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = loss_fn(logits.float(), labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device) -> np.ndarray:
    """Return sigmoid probabilities of shape (n, 6) in loader order."""
    device = torch.device(device)
    model.eval()
    outputs = []
    for input_ids, attention_mask, _ in loader:
        logits = model(
            input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)
        ).logits
        outputs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _select_rows(df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    """Return df's rows for ids, in the given id order."""
    return df.set_index("id").loc[ids].reset_index()


def main() -> None:
    """Fine-tune DistilBERT, keep the best epoch by validation mean average precision."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    seed = config["seed"]
    cfg = config["distilbert"]
    suffix = "_smoke" if args.limit else ""
    name = f"distilbert{suffix}"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_train = load_train(args.raw_dir)
    train_ids, validation_ids = load_split_ids(REPO_ROOT / "data" / "processed")
    if args.limit:
        train_ids = train_ids[: args.limit]
        validation_ids = validation_ids[: args.limit]
    train_df = _select_rows(full_train, train_ids)
    validation_df = _select_rows(full_train, validation_ids)

    tokenizer = AutoTokenizer.from_pretrained(cfg["checkpoint"])
    train_tokens = tokenize(train_df["comment_text"], tokenizer, cfg["max_length"])
    val_tokens = tokenize(validation_df["comment_text"], tokenizer, cfg["max_length"])
    y_train = train_df[LABELS].to_numpy(dtype=np.float32)
    y_val = validation_df[LABELS].to_numpy(dtype=int)

    collate_fn = partial(collate, pad_id=tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TextDataset(train_tokens, y_train),
        batch_size=cfg["batch_size"],
        shuffle=True,
        generator=generator,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        TextDataset(val_tokens, y_val.astype(np.float32)),
        batch_size=cfg["eval_batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["checkpoint"],
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
    ).to(device)
    optimizer = build_optimizer(model, cfg["learning_rate"], cfg["weight_decay"])
    total_steps = len(train_loader) * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg["warmup_ratio"] * total_steps), total_steps
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    models_dir = args.out_dir / "models" / name
    predictions_dir = args.out_dir / "predictions"
    metrics_dir = args.out_dir / "metrics"
    for folder in (models_dir, predictions_dir, metrics_dir):
        folder.mkdir(parents=True, exist_ok=True)

    epoch_rows = []
    best_mean = -np.inf
    best_epoch = 0
    best_proba = None
    started = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            cfg["max_grad_norm"], scaler,
        )
        proba = predict(model, val_loader, device)
        scores = per_label_average_precision(y_val, proba)
        mean_ap = float(np.mean(scores))
        print(f"epoch {epoch}: train_loss={train_loss:.4f} mean_ap={mean_ap:.4f}")
        epoch_rows.append(
            {"epoch": epoch, **dict(zip(LABELS, scores)), "mean_ap": mean_ap,
             "train_loss": train_loss}
        )
        if mean_ap > best_mean:
            best_mean = mean_ap
            best_epoch = epoch
            best_proba = proba
            model.save_pretrained(models_dir)
            tokenizer.save_pretrained(models_dir)
    training_seconds = time.time() - started

    predictions = pd.DataFrame(best_proba, columns=LABELS)
    predictions.insert(0, "id", validation_df["id"].to_numpy())
    predictions.to_csv(
        predictions_dir / f"{name}_validation.csv", index=False, float_format="%.8g"
    )

    pd.DataFrame(epoch_rows).to_csv(metrics_dir / f"{name}_epochs.csv", index=False)

    best_scores = per_label_average_precision(y_val, best_proba)
    per_label = pd.DataFrame(
        {
            "label": LABELS,
            "positives": y_val.sum(axis=0),
            "average_precision": best_scores,
        }
    )
    mean_row = pd.DataFrame(
        [{"label": "mean", "positives": np.nan,
          "average_precision": float(np.mean(best_scores))}]
    )
    pd.concat([per_label, mean_row], ignore_index=True).to_csv(
        metrics_dir / f"{name}_validation.csv", index=False
    )

    run_info = {
        "seed": seed,
        "distilbert": cfg,
        "best_epoch": best_epoch,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "training_seconds": training_seconds,
    }
    with (metrics_dir / f"{name}_run.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)


if __name__ == "__main__":
    main()
