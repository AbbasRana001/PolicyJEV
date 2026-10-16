import numpy as np
import pandas as pd

from policyjev.baselines.token_lengths import count_tokens, length_report
from policyjev.config import LABELS


class FakeTokenizer:
    """Tokenizer stub: one token per word plus two special tokens."""

    def __call__(self, texts, add_special_tokens, truncation):
        return {"input_ids": [[0] * (len(t.split()) + 2) for t in texts]}


def test_count_tokens_is_word_count_plus_two():
    texts = ["a", "a b", "a b c", "", "a b c d e"]
    counts = count_tokens(texts, FakeTokenizer(), batch_size=2)

    assert counts.tolist() == [3, 4, 5, 2, 7]


def test_length_report_percentile_and_share():
    rows = []
    positives = [None, "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    for i, positive in enumerate(positives):
        record = {"id": str(i), "comment_text": "x"}
        for label in LABELS:
            record[label] = int(label == positive)
        rows.append(record)
    df = pd.DataFrame(rows)
    lengths = np.array([10, 20, 30, 40, 50, 60, 200])

    report = length_report(df, lengths).set_index("group")

    assert report.loc["all", "comments"] == 7
    assert report.loc["all", "tokens_p50"] == 40.0
    assert report.loc["all", "tokens_max"] == 200
    assert abs(report.loc["all", "share_over_128"] - round(1 / 7, 4)) < 1e-9
    assert report.loc["no_label", "comments"] == 1
    assert report.loc["identity_hate", "share_over_128"] == 1.0
    assert report.loc["threat", "share_over_64"] == 0.0
