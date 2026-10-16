"""TF-IDF + one-vs-rest logistic regression baseline."""

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion

from policyjev.config import LABELS, REPO_ROOT, load_config
from policyjev.data.load import load_train
from policyjev.data.split import load_split_ids


def build_vectorizer(
    word_ngrams: tuple[int, int],
    min_df: int,
    max_features: int,
    char_ngrams: bool,
) -> TfidfVectorizer | FeatureUnion:
    """Return an unfitted word (and optionally char) TF-IDF transformer."""
    word = TfidfVectorizer(
        ngram_range=word_ngrams,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    if not char_ngrams:
        return word
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    return FeatureUnion([("word", word), ("char", char)])


def fit_logreg(X, y: np.ndarray, c: float, seed: int, n_jobs: int):
    """Fit a one-vs-rest logistic regression without class weights."""
    classifier = OneVsRestClassifier(
        LogisticRegression(C=c, solver="liblinear", random_state=seed),
        n_jobs=n_jobs,
    )
    X.sum_duplicates()
    return classifier.fit(X, y)


def per_label_average_precision(y_true: np.ndarray, proba: np.ndarray) -> list[float]:
    """Average precision for each label column."""
    return [
        float(average_precision_score(y_true[:, j], proba[:, j]))
        for j in range(y_true.shape[1])
    ]


def run_search(
    train_texts,
    y_train: np.ndarray,
    val_texts,
    y_val: np.ndarray,
    cfg: dict,
    seed: int,
) -> tuple[object, object, pd.DataFrame, float]:
    """Fit the vectorizer on train, pick C by mean validation average precision."""
    vectorizer = build_vectorizer(
        tuple(cfg["word_ngrams"]),
        cfg["min_df"],
        cfg["max_features"],
        cfg["char_ngrams"],
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    rows = []
    best_classifier = None
    best_c = None
    best_score = -np.inf
    for c in sorted(cfg["c_grid"]):
        classifier = fit_logreg(X_train, y_train, c, seed, cfg["n_jobs"])
        scores = per_label_average_precision(y_val, classifier.predict_proba(X_val))
        mean_ap = float(np.mean(scores))
        rows.append({"c": c, **dict(zip(LABELS, scores)), "mean_ap": mean_ap})
        if mean_ap > best_score:
            best_score = mean_ap
            best_classifier = classifier
            best_c = c

    search = pd.DataFrame(rows, columns=["c"] + LABELS + ["mean_ap"])
    return vectorizer, best_classifier, search, float(best_c)


def main() -> None:
    """Run the C search, save the model, validation predictions and metrics."""
    config = load_config()
    cfg = config["baselines"]["logreg"]
    seed = config["seed"]

    full_train = load_train()
    train_ids, validation_ids = load_split_ids(REPO_ROOT / "data" / "processed")
    train_df = full_train[full_train["id"].isin(train_ids)].reset_index(drop=True)
    validation_df = full_train[full_train["id"].isin(validation_ids)].reset_index(
        drop=True
    )

    y_train = train_df[LABELS].to_numpy(dtype=int)
    y_val = validation_df[LABELS].to_numpy(dtype=int)

    vectorizer, classifier, search, best_c = run_search(
        train_df["comment_text"], y_train, validation_df["comment_text"], y_val,
        cfg, seed,
    )

    tag = "word_char" if cfg["char_ngrams"] else "word"
    models_dir = REPO_ROOT / "artifacts" / "models"
    predictions_dir = REPO_ROOT / "artifacts" / "predictions"
    metrics_dir = REPO_ROOT / "artifacts" / "metrics"
    for folder in (models_dir, predictions_dir, metrics_dir):
        folder.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "c": best_c,
            "labels": LABELS,
        },
        models_dir / f"logreg_{tag}.joblib",
    )

    proba = classifier.predict_proba(vectorizer.transform(validation_df["comment_text"]))
    predictions = pd.DataFrame(proba, columns=LABELS)
    predictions.insert(0, "id", validation_df["id"].to_numpy())
    predictions.to_csv(
        predictions_dir / f"logreg_{tag}_validation.csv",
        index=False,
        float_format="%.8g",
    )

    search.to_csv(metrics_dir / f"logreg_{tag}_search.csv", index=False)

    scores = per_label_average_precision(y_val, proba)
    per_label = pd.DataFrame(
        {
            "label": LABELS,
            "positives": y_val.sum(axis=0),
            "average_precision": scores,
        }
    )
    mean_row = pd.DataFrame(
        [{"label": "mean", "positives": np.nan, "average_precision": np.mean(scores)}]
    )
    per_label = pd.concat([per_label, mean_row], ignore_index=True)
    per_label.to_csv(metrics_dir / f"logreg_{tag}_validation.csv", index=False)

    print(search.to_string(index=False))
    print(f"chosen C: {best_c}")
    print(per_label.to_string(index=False))
    grid = sorted(cfg["c_grid"])
    if best_c in (grid[0], grid[-1]):
        print("WARNING: best C is at the edge of the grid")


if __name__ == "__main__":
    main()
