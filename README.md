# PolicyJEV

A comparison, run under a frozen protocol, of a zero-shot classifier (TypeSafe JEV) against two trained baselines on toxic-comment classification, followed by a test of what happens when the scores are turned into moderation decisions (allow, human review, action).

Author: Muhammad Abbas Rana. Licence: MIT.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23407397.svg)](https://doi.org/10.5281/zenodo.23407397)

## The question

JEV needs no training data: you describe each label as a set of natural-language questions and it returns probabilities. Two things were tested:

1. **Ranking quality.** Does zero-shot JEV rank toxic comments as well as a TF-IDF logistic regression and a fine-tuned DistilBERT?
2. **Decisions.** If the probabilities drive a three-way policy with thresholds fitted on validation data, how do the three systems behave on unseen data?

## Result in short

On the held-out test set (63,978 comments, 6,243 with at least one positive label), JEV ranks worse than both trained baselines. Its probabilities behave differently once they drive decisions, but every system missed some of the validation targets on test.

Mean average precision (AP) over the six labels, with 95% paired-bootstrap intervals (1,000 resamples):

| System | Mean AP | 95% interval |
|---|---|---|
| TF-IDF + logistic regression | 0.583 | [0.566, 0.601] |
| DistilBERT (fine-tuned) | 0.677 | [0.662, 0.693] |
| JEV (zero-shot, question set v2) | 0.527 | [0.509, 0.544] |

JEV minus logistic regression: -0.056 [-0.075, -0.038]. JEV minus DistilBERT: -0.150 [-0.167, -0.134].

A secondary supervised model fitted on JEV's six probabilities (the "JEV stack") scored 0.536 [0.520, 0.555]. It did not help on test (+0.009, interval -0.004 to 0.023), is excluded from the policy analysis, and is never reported as JEV.

Policy outcome on the test set (calibrated probabilities, thresholds fitted on validation only):

| System | Allow | Review | Action |
|---|---|---|---|
| Logistic regression | 48,798 | 8,703 | 6,477 |
| DistilBERT | 51,450 | 4,358 | 8,170 |
| JEV | 49,927 | 11,060 | 2,991 |

JEV sends the most comments to human review and takes automatic action on the fewest. The five policy measures are in [`docs/results_policy.md`](docs/results_policy.md). Paired contrasts (JEV minus each baseline) exist for four of them: M1 review share, M2 ACTION precision, M3 miss rate and M4 rare miss rate. ACTION coverage has no contrast (`artifacts/metrics/policy_test_contrasts.csv`).

Rare-label recall at a fixed review budget (comments positive for threat, severe_toxic or identity_hate; the top share of traffic by maximum calibrated probability goes to review). The three recall values come from [`artifacts/metrics/review_budget_test.csv`](artifacts/metrics/review_budget_test.csv), where their `status` is `descriptive` at every budget. The last two columns give each JEV-minus-baseline contrast with its 95% paired-bootstrap interval and the `status` recorded for it in that file.

| Budget | Logistic regression | DistilBERT | JEV | JEV minus logistic regression [95% interval], status | JEV minus DistilBERT [95% interval], status |
|---|---|---|---|---|---|
| 1% | 0.195 | 0.125 | 0.307 | +0.112 [+0.087, +0.136], descriptive | +0.182 [+0.155, +0.207], hypothesis generated on validation |
| 2% | 0.386 | 0.243 | 0.462 | +0.076 [+0.047, +0.103], descriptive | +0.219 [+0.196, +0.250], hypothesis generated on validation |
| 5% | 0.688 | 0.603 | 0.674 | −0.014 [−0.038, +0.016], pre-specified | +0.071 [+0.046, +0.101], pre-specified |
| 10% | 0.855 | 0.863 | 0.817 | −0.038 [−0.061, −0.011], pre-specified | −0.046 [−0.069, −0.022], pre-specified |
| 20% | 0.957 | 0.982 | 0.934 | −0.023 [−0.040, −0.005], descriptive | −0.048 [−0.062, −0.033], descriptive |

- At 1% and 2% JEV has the highest rare-label recall. Neither comparison was a pre-specified contrast: the comparison with logistic regression is descriptive, and the comparison with DistilBERT is a hypothesis generated on validation (it held on test). On validation, JEV and logistic regression had been level.
- Only the 5% and 10% contrasts were pre-specified. At 5%, JEV is above DistilBERT, and its interval against logistic regression includes zero. At 10%, JEV is below both baselines (both intervals are below zero).
- At 20% (descriptive) JEV is also below both baselines.
- No multiplicity adjustment is made across these intervals.

### Where the validation targets were missed

Thresholds were chosen on validation so that recall is at least 0.90 at the review threshold and precision is 0.90 at the action threshold (with a 30-comment support floor). On test:

- Precision at the action threshold fell on the toxic and obscene labels for all three systems, from 0.90-0.91 on validation to 0.62-0.82.
- Recall at the review threshold was below 0.90 in 8 of 18 system-label pairs (logistic regression 1, DistilBERT 2, JEV 5).

The cause was not established. Both results are reported as found.

## How the study was run

- **Data.** Jigsaw Toxic Comment Classification Challenge: six labels (toxic, severe toxic, obscene, threat, insult, identity hate). The labelled training file was split 85/15 into train and validation (135,635 and 23,936 comments, seed 42). The released test set with labels is the test set.
- **Logistic regression.** TF-IDF word 1-2-grams, min_df 3, up to 200,000 features, C chosen from [1, 2, 4, 8, 16] on validation.
- **DistilBERT.** `distilbert-base-uncased`, max length 256, 3 epochs, batch 32, learning rate 2e-5, weight decay 0.01, warm-up 0.06, gradient clipping 1.0, one run on a Colab T4.
- **JEV.** Model `jev-1.13.0`. Three question sets were compared on validation by AP (v2 0.5659, v1 0.5409, v3 0.5377); v2 was chosen. JEV needs no labels to run but does need question writing and selection, so it is label-light, not label-free.
- **Calibration.** Fitted on validation (isotonic for logistic regression and JEV, Platt for DistilBERT), with 5-fold out-of-fold Brier scores.
- **Policy.** A comment's decision is the most severe across the six labels: below the low threshold, allow; between the thresholds, human review; at or above the high threshold, action. Thresholds come from out-of-fold calibrated validation probabilities. A label with no valid high threshold is review-only and can never trigger action.
- **Freeze and test.** The order of events, from the commit history and git tags of the private development repository: the protocol, question set v2, the saved calibrators, the classification thresholds fitted on validation and a hash manifest ([`docs/frozen_manifest.csv`](docs/frozen_manifest.csv)) were frozen before any test scoring (tags `questions-frozen`, `questions-alt-frozen` and `protocol-frozen`). Each system's test predictions were then produced once; for JEV, a second pass re-queried a fixed 5,000-comment test subset (see [`docs/results.md`](docs/results.md), section 7). Amendment 3, which defines the moderation policy and its thresholds, was written afterwards: the test predictions existed, the test results for average precision, F1, calibration and the review-budget analysis had been seen, and a results report existed. Its thresholds were fitted on validation data only, and it was frozen (tag `policy-frozen`) before the policy was evaluated on the test set. The protocol has three recorded amendments ([`docs/experiment_protocol.md`](docs/experiment_protocol.md)). The protocol file uses the word "pre-registered" for the original protocol; it means committed to git before test scoring, not entered in a public registry. This public repository is a snapshot of the final files, so the tags are not part of it; commands that read the test split check for those tags and refuse to run here.
- **Cost.** The protocol specified a cost-per-1,000-comments measure. Pricing details are not published here, so that measure is not reported. A pre-set spending cap applied to the batch scoring runs (validation, test and second pass); live console scoring has no cap. JEV scoring made roughly 141,000 successful calls (the exact count comes from a call log that is not published). Latency is reported instead.
- **Latency (one machine).** Local timings are on a single Intel CPU: logistic regression 1.8 ms p50, DistilBERT 40.8 ms p50. JEV is a network round trip: 363 ms p50 and 686 ms p95, and depends on network and provider load.

Full write-up: [`docs/results.md`](docs/results.md). Decision log: [`docs/decisions.md`](docs/decisions.md). Generated tables: [`docs/results_tables.md`](docs/results_tables.md).

## Limitations

- One dataset, one JEV version and one question set.
- Intervals cover test-set sampling only; the baselines are single-seed runs.
- No multiplicity adjustment across labels and contrasts. F1 is reported without intervals.
- Labels are noisy, as in any crowd-annotated toxicity set.
- The drop from validation to test is unexplained.
- Latency was measured on one machine.
- Comments are sent to a third-party API when JEV scores them.

## The console

A React, TypeScript and Tailwind front end with a FastAPI back end, in [`console/`](console/). It has three views:

- **Score.** Type a comment and see each system's probabilities, decision and latency side by side. JEV runs only when you press Score (or Ctrl+Enter); "Score as I type" calls the two local systems only.
- **Policy.** The thresholds per system, a probability probe that shows which decision a given probability would get, the five policy measures, and validation-versus-test comparisons.
- **Results.** AP by label and the review-budget curves.

Notes:

- The key status chip reading "ready" means a key is configured, not that it has been verified. A rejected or missing key shows "Could not score" or "Unavailable" when JEV is used.
- Scoring a comment with JEV sends its text to the vendor's service. This project does not log or store comments, and cannot state what the vendor retains.
- Local scoring (logistic regression, DistilBERT) needs model weights that are not in this repository. Setup and run instructions are in [`console/README.md`](console/README.md).

## Setup

Developed on Windows with Python 3.13 and Node 22. The Python test suite also passed in a clone of this repository without the ignored model weights. A run of everything from a bare checkout, including the front end, has not been verified.

```
python -m pip install -r requirements.txt
python -m pip install -e .
cd console
npm install
```

What works with only this repository:

- Read all results, tables and figures.
- Regenerate `docs/results_tables.md` from the tracked metrics files.
- Use the Policy and Results views, which read tracked CSVs.

What needs more:

- Retraining, EDA and test inference need the Kaggle files at `data/raw/train.csv`, `test.csv` and `test_labels.csv` (accept the competition rules to download them).
- Scoring with JEV needs your own `TYPESAFE_API_KEY` in `.env` (see `.env.example`).
- The batch JEV runners (`policyjev.jev.runner` and `policyjev.jev.second_pass`) also read `TYPESAFE_COST_RATE` and `TYPESAFE_HARD_STOP`, and need both for a run that is not a dry run (see `.env.example`).
- The freeze check and local scoring need the model weights, which are ignored by git.

## Data and licensing

- **Comment text is not in this repository.** Tracked data are comment ids, split lists, probabilities and metrics.
- **Jigsaw data.** From the [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). The underlying comments come from Wikipedia talk pages. Check the competition page for the current terms before reusing the data.
- **DistilBERT.** Sanh, Debut, Chaumond and Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter" (2019, arXiv:1910.01108); `distilbert-base-uncased` is released under Apache-2.0 according to its model card.
- **TypeSafe JEV.** A commercial API used under the vendor's terms; this project is independent and is not affiliated with or endorsed by the vendor. The JEV probabilities in `artifacts/predictions` are outputs of that service.
- **Code.** MIT, see [`LICENSE`](LICENSE). Dependency licences are their own: mostly MIT, BSD and Apache-2.0, with some MPL-2.0 components (Python: `certifi` and `tqdm`; front end: `lightningcss` and its platform packages, pulled in by the build tooling) and two OFL-1.1 web fonts (Bricolage Grotesque and Instrument Sans, from the `@fontsource` packages).

## Layout

```
src/policyjev/   package: data, baselines, jev client, evaluation, policy, api
tests/           Python tests
console/         front end, end-to-end checks and scripts
artifacts/       calibrators, metrics, predictions, policy thresholds
configs/         project config and the three JEV question sets
docs/            protocol, decisions, results, policy results, manifest
reports/figures/ figures
data/processed/  split id lists and the question pilot sample
```

[`docs/charter.md`](docs/charter.md) is the original planning document and is kept as history; where it differs from the protocol, the protocol governs.
