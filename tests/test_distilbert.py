import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertConfig, DistilBertForSequenceClassification

from policyjev.baselines.distilbert import (
    TextDataset,
    build_optimizer,
    collate,
    predict,
    train_one_epoch,
)


def make_model() -> DistilBertForSequenceClassification:
    """Build a tiny randomly initialised DistilBERT classifier."""
    config = DistilBertConfig(
        vocab_size=100,
        max_position_embeddings=32,
        n_layers=1,
        n_heads=2,
        dim=16,
        hidden_dim=32,
        num_labels=6,
        problem_type="multi_label_classification",
    )
    return DistilBertForSequenceClassification(config)


def make_data() -> tuple[list[list[int]], np.ndarray]:
    """Random 64 x 16 token ids; labels are all ones when token 7 appears."""
    rng = np.random.default_rng(0)
    ids = rng.integers(1, 100, size=(64, 16))
    labels = np.zeros((64, 6), dtype=np.float32)
    labels[(ids == 7).any(axis=1)] = 1.0
    return ids.tolist(), labels


def test_collate_pads_to_batch_maximum():
    batch = [
        ([5, 6, 7], np.zeros(6, dtype=np.float32)),
        ([8], np.ones(6, dtype=np.float32)),
    ]
    input_ids, attention_mask, labels = collate(batch, pad_id=0)

    assert input_ids.shape == (2, 3)
    assert input_ids.tolist() == [[5, 6, 7], [8, 0, 0]]
    assert attention_mask.tolist() == [[1, 1, 1], [1, 0, 0]]
    assert labels.dtype == torch.float32
    assert labels.shape == (2, 6)


def test_training_reduces_loss_and_predict_shape():
    torch.manual_seed(0)
    model = make_model()
    ids, labels = make_data()
    loader = DataLoader(
        TextDataset(ids, labels),
        batch_size=16,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, pad_id=0),
    )
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.01)

    losses = [
        train_one_epoch(model, loader, optimizer, None, "cpu", 1.0, None)
        for _ in range(30)
    ]
    assert losses[-1] < losses[0]

    proba = predict(model, loader, "cpu")
    assert proba.shape == (64, 6)
    assert (proba >= 0).all()
    assert (proba <= 1).all()


def test_build_optimizer_excludes_bias_and_layernorm_from_decay():
    model = make_model()
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.01)

    decay_group, no_decay_group = optimizer.param_groups
    assert decay_group["weight_decay"] == 0.01
    assert no_decay_group["weight_decay"] == 0.0

    no_decay_ids = {id(p) for p in no_decay_group["params"]}
    for name, param in model.named_parameters():
        is_bias = name.endswith("bias")
        is_layer_norm = "layer_norm" in name.lower() or "LayerNorm" in name
        if is_bias or is_layer_norm:
            assert id(param) in no_decay_ids, name
        else:
            assert id(param) not in no_decay_ids, name
