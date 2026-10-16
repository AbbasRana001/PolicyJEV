# PolicyJEV

A pre-registered comparison of a zero-shot classifier (TypeSafe JEV) against two trained baselines on toxic-comment classification, followed by a test of what happens when the scores are turned into moderation decisions (allow, human review, action).

Author: Muhammad Abbas Rana. Licence: MIT.

## The question

JEV needs no training data: you describe each label as a set of natural-language questions and it returns probabilities. Two things were tested:

1. **Ranking quality.** Does zero-shot JEV rank toxic comments as well as a TF-IDF logistic regression and a fine-tuned DistilBERT?
2. **Decisions.** If the probabilities drive a three-way policy with thresholds fixed in advance, how do the three systems behave on unseen data?

## Result in short

On the held-out test set (63,978 comments, 6,243 with at least one positive label), JEV ranks worse than both trained baselines. Its probabilities behave differently once they drive decisions, but the validation targets did not hold on test for any system.

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

JEV sends the most comments to human review and takes automatic action on the fewest. The five policy measures and their paired contrasts are in [`docs/results_policy.md`](docs/results_policy.md).

Rare-label recall at a fixed review budget is where JEV does best: it leads at 1% and 2% budgets (recall 0.307 and 0.462, against 0.195 and 0.386 for logistic regression), is level with logistic regression at 5%, and trails at 10% and 20%.

### Where the validation targets did not hold

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
- **Freeze and test.** The protocol, question set, calibrators, thresholds and a hash manifest ([`docs/frozen_manifest.csv`](docs/frozen_manifest.csv)) were frozen before any test scoring, in a private development repository. The test set was then run once. The protocol has three recorded amendments ([`docs/experiment_protocol.md`](docs/experiment_protocol.md)). This public repository is a single clean commit of the final files, so the freeze tags from the development repository are not part of it; the manifest is what records what was frozen.
- **Cost.** The protocol specified a cost-per-1,000-comments measure. Pricing details are not published here, so that measure is not reported. JEV scoring made 141,183 successful calls in total, and a pre-set spending cap applied to every run. Latency is reported instead.
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

Developed and tested on Windows with Python 3.13 and Node 22, where the Python and front-end test suites pass with the full local setup. A run from a bare checkout without the ignored model weights has not been verified.

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
- The freeze check and local scoring need the model weights, which are ignored by git.

## Data and licensing

- **Comment text is not in this repository.** Tracked data are comment ids, split lists, probabilities and metrics.
- **Jigsaw data.** From the [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge), built on the annotations described in Wulczyn, Thain and Dixon, "Ex Machina: Personal Attacks Seen at Scale", WWW 2017 (doi:10.1145/3038912.3052591). The underlying comments come from Wikipedia talk pages. Check the competition page for the current terms before reusing the data.
- **DistilBERT.** Sanh, Debut, Chaumond and Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter" (2019, arXiv:1910.01108); `distilbert-base-uncased` is released under Apache-2.0.
- **TypeSafe JEV.** A commercial API used under the vendor's terms; this project is independent and is not affiliated with or endorsed by the vendor. The JEV probabilities in `artifacts/predictions` are outputs of that service.
- **Code.** MIT, see [`LICENSE`](LICENSE). Dependency licences are their own (mostly MIT, BSD and Apache-2.0).

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
