import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizerFast

from policyjev.baselines import predict_test as pt
from policyjev.baselines.logreg import build_vectorizer, fit_logreg
from policyjev.config import LABELS
from policyjev.evaluation import splits

TEXTS = [
    "alpha beta gamma",
    "gamma delta",
    "alpha alpha alpha beta beta",
    "delta epsilon zeta eta",
    "beta",
    "gamma gamma delta epsilon zeta eta theta iota kappa",
] * 10


def save_logreg(path, labels=None):
    rng = np.random.default_rng(0)
    y = (rng.random((len(TEXTS), 6)) < 0.5).astype(int)
    vectorizer = build_vectorizer((1, 1), 1, 1000, False)
    X = vectorizer.fit_transform(TEXTS)
    classifier = fit_logreg(X, y, 1.0, 0, 1)
    joblib.dump(
        {"vectorizer": vectorizer, "classifier": classifier, "c": 1.0, "labels": labels or LABELS},
        path,
    )
    return vectorizer, classifier


def test_logreg_prediction_matches_the_saved_model(tmp_path):
    path = tmp_path / "logreg_word.joblib"
    vectorizer, classifier = save_logreg(path)

    probs = pt.predict_logreg(path, TEXTS[:7])

    assert probs.shape == (7, 6)
    assert np.allclose(probs, classifier.predict_proba(vectorizer.transform(TEXTS[:7])))
    assert (probs >= 0).all() and (probs <= 1).all()


def test_logreg_label_order_mismatch_is_rejected(tmp_path):
    path = tmp_path / "logreg_word.joblib"
    save_logreg(path, labels=list(reversed(LABELS)))
    with pytest.raises(ValueError):
        pt.predict_logreg(path, TEXTS[:3])


VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
]


def save_tiny_distilbert(model_dir):
    """A randomly initialised tiny model plus a local word-level tokenizer; no downloads."""
    vocab_file = model_dir.parent / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
    tokenizer = DistilBertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True)
    config = DistilBertConfig(
        vocab_size=len(VOCAB), max_position_embeddings=32, n_layers=1, n_heads=2,
        dim=16, hidden_dim=32, num_labels=6, problem_type="multi_label_classification",
    )
    model_dir.mkdir()
    DistilBertForSequenceClassification(config).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def test_distilbert_prediction_is_batch_independent_and_order_preserving(tmp_path):
    model_dir = tmp_path / "model"
    save_tiny_distilbert(model_dir)

    batched = pt.predict_distilbert(model_dir, TEXTS, max_length=8, batch_size=4, device="cpu")
    single = pt.predict_distilbert(model_dir, TEXTS, max_length=8, batch_size=1, device="cpu")

    assert batched.shape == (len(TEXTS), 6)
    assert (batched >= 0).all() and (batched <= 1).all()
    assert np.allclose(batched, single, atol=1e-5)

    one_by_one = np.vstack(
        [pt.predict_distilbert(model_dir, [t], max_length=8, batch_size=1, device="cpu") for t in TEXTS[:6]]
    )
    assert np.allclose(batched[:6], one_by_one, atol=1e-5)


def test_distilbert_truncates_to_max_length(tmp_path):
    model_dir = tmp_path / "model"
    save_tiny_distilbert(model_dir)
    long_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    cut_text = "alpha beta gamma delta epsilon zeta"

    long = pt.predict_distilbert(model_dir, [long_text], max_length=8, batch_size=1, device="cpu")
    cut = pt.predict_distilbert(model_dir, [cut_text], max_length=8, batch_size=1, device="cpu")

    assert np.allclose(long, cut, atol=1e-5)


def test_auto_device_resolves_to_a_valid_device():
    assert pt.resolve_device("auto").type in ("cpu", "cuda")
    assert pt.resolve_device("cpu").type == "cpu"


def test_cli_refuses_to_run_without_the_tag_and_reads_nothing(monkeypatch):
    monkeypatch.setattr(splits, "tag_exists", lambda tag: False)

    def forbidden(*args, **kwargs):
        raise AssertionError("test data must not be read")

    monkeypatch.setattr(pt, "load_labeled_frame", forbidden)
    monkeypatch.setattr(pt, "predict_logreg", forbidden)
    monkeypatch.setattr(sys, "argv", ["predict_test", "--system", "logreg"])

    with pytest.raises(SystemExit) as info:
        pt.main()
    assert "protocol-frozen" in str(info.value)


def test_cli_writes_ids_in_order_and_probabilities_only(monkeypatch, tmp_path):
    monkeypatch.setattr(splits, "tag_exists", lambda tag: True)
    frame = pd.DataFrame(
        {"id": ["t3", "t1", "t2"], "comment_text": ["a", "b", "c"], **{l: [0, 1, 0] for l in LABELS}}
    )
    monkeypatch.setattr(pt, "load_labeled_frame", lambda split: frame)
    monkeypatch.setattr(
        pt, "predict_logreg", lambda path, texts: np.full((len(texts), 6), 0.25)
    )
    monkeypatch.setattr(pt, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(pt, "verify_entries", lambda entries: None)
    monkeypatch.setattr(sys, "argv", ["predict_test", "--system", "logreg"])

    pt.main()

    written = pd.read_csv(tmp_path / "artifacts" / "predictions" / "logreg_word_test.csv", dtype={"id": str})
    assert written.columns.tolist() == ["id"] + LABELS
    assert written["id"].tolist() == ["t3", "t1", "t2"]
    assert (written[LABELS] == 0.25).all().all()


def test_length_sorted_batching_matches_the_unsorted_path(tmp_path, monkeypatch):
    from functools import partial

    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from policyjev.baselines import distilbert as db

    model_dir = tmp_path / "model"
    save_tiny_distilbert(model_dir)
    max_length = 8
    texts = TEXTS + ["alpha beta gamma delta epsilon zeta eta theta iota kappa alpha beta"]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    token_ids = db.tokenize(texts, tokenizer, max_length)
    unsorted_loader = DataLoader(
        db.TextDataset(token_ids, np.zeros((len(texts), 6), dtype=np.float32)),
        batch_size=len(texts),
        shuffle=False,
        collate_fn=partial(db.collate, pad_id=tokenizer.pad_token_id),
    )
    reference = db.predict(model, unsorted_loader, "cpu")

    widths = []
    real_collate = pt.collate

    def recording_collate(batch, pad_id):
        result = real_collate(batch, pad_id)
        widths.append(result[0].shape[1])
        return result

    monkeypatch.setattr(pt, "collate", recording_collate)
    sorted_path = pt.predict_distilbert(model_dir, texts, max_length=max_length, batch_size=8, device="cpu")

    assert sorted_path.shape == reference.shape
    assert np.allclose(sorted_path, reference, atol=1e-5)
    assert widths == sorted(widths)
    assert widths[0] < widths[-1] <= max_length
