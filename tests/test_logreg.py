import numpy as np

from scipy import sparse

from policyjev.baselines.logreg import fit_logreg
from policyjev.baselines.logreg import run_search
from policyjev.config import LABELS

VOCAB = [f"w{i}" for i in range(30)]
TRIGGERS = {label: [f"w{2 * j}", f"w{2 * j + 1}"] for j, label in enumerate(LABELS)}
CFG = {
    "c_grid": [1.0, 2.0],
    "word_ngrams": [1, 2],
    "min_df": 1,
    "max_features": 5000,
    "char_ngrams": False,
    "n_jobs": 1,
}
UNSEEN_WORD = "zzzunique"


def build_corpus() -> tuple[list[str], np.ndarray]:
    """Build 400 synthetic comments whose labels depend on trigger-word presence."""
    rng = np.random.default_rng(0)
    texts = []
    labels = []
    for _ in range(400):
        words = rng.choice(VOCAB, size=8).tolist()
        texts.append(" ".join(words))
        labels.append(
            [int(any(t in words for t in TRIGGERS[label])) for label in LABELS]
        )
    return texts, np.array(labels, dtype=int)


def run_on_corpus(add_unseen_word: bool = False):
    """Run the search on a 300/100 split of the synthetic corpus."""
    texts, y = build_corpus()
    train_texts, val_texts = texts[:300], texts[300:]
    if add_unseen_word:
        val_texts = [f"{t} {UNSEEN_WORD}" if i % 2 == 0 else t for i, t in enumerate(val_texts)]
    y_train, y_val = y[:300], y[300:]
    result = run_search(train_texts, y_train, val_texts, y_val, CFG, seed=0)
    return result, val_texts


def test_search_returns_grid_value_and_expected_table():
    (_vectorizer, _classifier, search, best_c), _ = run_on_corpus()

    assert best_c in CFG["c_grid"]
    assert len(search) == len(CFG["c_grid"])
    assert list(search.columns) == ["c"] + LABELS + ["mean_ap"]
    assert search["c"].tolist() == CFG["c_grid"]


def test_validation_probabilities_have_valid_shape_and_range():
    (vectorizer, classifier, _search, _best_c), val_texts = run_on_corpus()
    proba = classifier.predict_proba(vectorizer.transform(val_texts))

    assert proba.shape == (100, 6)
    assert (proba >= 0).all()
    assert (proba <= 1).all()


def test_vectorizer_is_fit_on_train_only():
    (vectorizer, _classifier, _search, _best_c), val_texts = run_on_corpus(
        add_unseen_word=True
    )

    assert any(UNSEEN_WORD in t.split() for t in val_texts)
    assert UNSEEN_WORD not in vectorizer.vocabulary_


def test_synthetic_labels_are_learnable():
    (_vectorizer, _classifier, search, _best_c), _ = run_on_corpus()
    assert search["mean_ap"].max() > 0.5


def test_fit_logreg_canonicalizes_training_matrix():
    """fit_logreg leaves the training matrix in canonical form so parallel workers can read it."""
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([2, 0, 1, 0])
    indptr = np.array([0, 2, 4])
    X = sparse.csr_matrix((data, indices, indptr), shape=(2, 3))
    y = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]])
    assert not X.has_sorted_indices
    fit_logreg(X, y, 1.0, 0, 1)
    assert X.has_canonical_format