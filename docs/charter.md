# PolicyJEV: Project Charter

> Historical planning document. docs/experiment_protocol.md governs the experiment. In particular, the Val-J and Test-J subset design, sample weights and weighted-versus-full sanity check described below were replaced by full validation and test splits, and "training-free" wording for JEV is superseded by the protocol (JEV is zero-shot but uses validation labels for question-set selection, calibration and thresholds).

---

## 1. What this project is

PolicyJEV is a **research-backed NLP/ML capstone and technical case study**, not a publication project.

It evaluates **TypeSafe JEV** (typed probabilistic judgments from six Noul questions) for multi-label toxic-content moderation, compares it with two conventional supervised models, and builds a small moderation console in which model probabilities drive configurable human-review decisions.

It must be strong enough to hold up to public review: GitHub, portfolio and live demos.

**Honest framing (used everywhere, including the README):** the two baselines are *trained* on ~160k labeled comments; JEV is *not trained at all* (it is prompted with question definitions). The case study asks how far a training-free typed-probability approach gets, and whether its probabilities are usable for routing. It is not framed as "JEV vs the best possible classifier".

## 2. Core research question

How effectively can JEV's typed probabilistic judgments perform multi-label toxic-content moderation compared with conventional supervised NLP models, and can those probabilities support configurable human-review moderation policies?

We care about: (1) classification performance, (2) probability/calibration quality, (3) rare harmful categories (threat, severe_toxic, identity_hate), (4) configurable thresholds, (5) ALLOW / HUMAN REVIEW / ACTION routing.

## 3. Dataset

**Jigsaw Toxic Comment Classification** only. Labels: `toxic, severe_toxic, obscene, threat, insult, identity_hate`.

Practical notes (verify in Module 1, do not assume):
- `train.csv` is ~160k labeled rows. The labeled test set is `test.csv` + `test_labels.csv`; rows with label `-1` were never scored and must be dropped (leaves roughly 64k).
- Downloading requires accepting the Kaggle competition rules. **Raw data is never committed to git** (gitignored). We commit code, configs and aggregate results only.
- The data contains abusive text. The README/UI show a content warning, and public demo examples are **self-written examples**, not copied dataset comments.
- Jigsaw labels are noisy (annotator disagreement), especially threat and identity_hate. Some "model errors" will really be label errors. The error analysis must say so honestly.

## 4. Data design (the main methodological fix)

### 4.1 Three-way split
| Split | Source | Used for |
|---|---|---|
| **train** (~85% of train.csv) | train.csv | fitting baselines |
| **validation** (~15% of train.csv) | train.csv | early stopping, hyperparameters, **threshold selection**, any calibration fitting |
| **test** | official labeled test | **final evaluation only, one time** |

- The split is created once, stratified so that rare labels appear in both train and validation, with a fixed seed. Row IDs are saved to disk and never regenerated.
- The test set is not opened (no EDA, no predictions, no thresholds) until the experiment protocol is frozen and all models and thresholds are final.

### 4.2 Why JEV gets its own benchmark subsets
Running JEV on every comment may be too expensive or slow, and the rare classes are tiny (threat is roughly 0.3% of comments). Random subsampling would leave almost no threat positives. So JEV runs on two fixed, **stratified** subsets:

- **Val-J** (from validation): used to choose thresholds. Target ~1.5k comments.
- **Test-J** (from test): used for the final three-way comparison. Target ~3–4k comments.

Construction (identical for both):
- **Rare stratum:** every comment positive for threat, severe_toxic or identity_hate (all of them, or a fixed random cap if the budget demands).
- **Common stratum:** a fixed random sample of all other comments.
- Each row stores a **sampling weight** = (stratum size in the full split) ÷ (rows sampled from it). Plain meaning: the row "stands in for" that many real comments.

Consequences, stated so nobody forgets them:
- **All three systems are compared on exactly the same Test-J rows.** Never JEV-on-subset vs baselines-on-full-test.
- **Headline metrics are weighted** (scikit-learn accepts `sample_weight`), so precision, PR-AUC, Brier score and workload percentages reflect real-world prevalence rather than the enriched sample.
- Every rare-class metric is reported **next to its raw positive count** (e.g., "threat: n = 211").
- **Sanity check (mandatory):** for the baselines, compare weighted-Test-J metrics with metrics on the full test set. They should be close. If not, the sampling/weighting code has a bug.
- Baselines also report full-test numbers as a secondary result.
- **Budget gate:** target sizes are targets. The cost pilot (Module 3) sets the real N. We shrink N before we ever drop a system or change the design. Minimum viable: Val-J ~700, Test-J ~2k.

### 4.3 Thresholds and leakage rules
- Any threshold (F1-optimal per-label thresholds for all three systems, and the LOW/HIGH policy thresholds) is **chosen on Val-J only** and then **applied once to Test-J**.
- Default classification threshold is also reported at a fixed 0.5 for transparency.
- The Policy Simulator defaults to Val-J. A Test-J view exists as "held-out confirmation", labelled as such. Official policy numbers in the README are computed from validation-frozen thresholds applied to Test-J once.
- JEV outputs are never used to train, tune or select anything for the other models.
- The **question-design pilot uses train-split comments only** (~100–150 comments, disjoint from Val-J and Test-J). Questions are frozen and git-tagged **before** any Val-J or Test-J run.

## 5. Systems

**System A: TF-IDF + One-vs-Rest Logistic Regression.** Word (and optionally char) n-grams. Regularization strength chosen on validation.

**System B: DistilBERT multi-label.** text → DistilBERT → 6 logits → sigmoid → 6 probabilities. Binary cross-entropy per label (`BCEWithLogitsLoss`), **not softmax**. Sensible defaults: max length chosen from the EDA length distribution (likely 128), 2–3 epochs, one seed (reported honestly as a single run), trained on Colab/Kaggle GPU. Checkpoint chosen on validation.

**System C: TypeSafe JEV.** One shared comment state, six carefully defined Noul questions mapped to the Jigsaw labels, six probabilities out. One well-designed question set, frozen after the pilot. No formulation study.

## 6. Evaluation (compact)

Classification: micro F1, macro F1, per-label precision / recall / F1 / PR-AUC.
Probability quality: Brier score (per label and mean), reliability plots (with few bins and counts shown, since rare labels are sparse).
Uncertainty: **bootstrap 95% intervals** (resampling within strata, ~1,000 resamples) for headline metrics and rare-class metrics. This is not hypothesis testing; it just keeps claims like "JEV beats DistilBERT on threats" honest.
Special attention: threat, severe_toxic, identity_hate.
Error analysis: a small number of meaningful false positives and false negatives per system, checked against the possibility of label noise.
Practical: latency (p50/p95) for each system, and the number of API calls for JEV.

Optional stretch (only if everything else is done): one extra row "JEV + Platt calibration" (a simple logistic rescaling of JEV probabilities, fit on Val-J, evaluated on Test-J). Clearly labelled as post-hoc. Skip it without guilt if time is short.

## 7. Moderation policy

A separate deterministic engine, never mixed with model code or UI judgment display:

- p < LOW → **ALLOW**
- LOW ≤ p < HIGH → **HUMAN REVIEW**
- p ≥ HIGH → **ACTION**

Thresholds may be per-category. A comment's decision is the most severe decision across its six labels (ACTION > REVIEW > ALLOW). Compare threshold settings using (all **weighted** to natural prevalence): auto-allow %, auto-action %, human-review %, false positives, false negatives, and threat recall. Plain-language definition for the UI: *false positive = harmless comment sent to ACTION; false negative = harmful comment auto-ALLOWed.*

Threshold changes never call JEV again; they operate on cached predictions.

## 8. JEV engineering requirements

API wrapper; model/version configuration; caching; token-usage logging; latency logging; error handling (retries with backoff, clear failures, no silent drops); probability persistence.

- **Cache key** = hash of (normalized comment text, question-set version, JEV model/version, relevant config). Any change creates a new key; the same input is never paid for twice.
- Cache and logs live in **SQLite** (one file, no server).
- Calls are resumable: a crashed batch run restarts without repeating completed rows.
- If repeated calls are non-deterministic, the first cached result is canonical and this is documented.
- **Unknown facts to verify at the Module 3 Gate (we have no confirmed numbers):** cost per call, whether six questions cost one call or six, rate limits, exact probability output format, determinism, model-version pinning. The Gate converts these into the final N for Val-J and Test-J.

## 9. Final application: PolicyJEV Moderation Console

React + TypeScript + Tailwind + FastAPI. Exactly three views:

1. **Live Moderation:** enter text → six JEV probabilities → thresholds → policy decision.
2. **Policy Simulator:** adjust thresholds → instant effect on automation, review workload and errors from cached predictions (no JEV calls).
3. **Benchmark Case Study:** JEV vs DistilBERT vs Logistic Regression; compact charts and metrics; rare classes; calibration; latency/cost.

Least-resistance rules:
- **Contract first:** Module 5 exports JSON artifacts (metrics, predictions, policy curves) and fixes the API schema. The frontend is built against those files, so the React work doesn't wait on live model runs.
- The Simulator and Benchmark views read **real exported experiment values**, never hardcoded numbers.
- **Demo mode:** Live Moderation works locally with a JEV key supplied by the user. The public/recorded demo uses cached example results so no key is exposed and no money is spent.
- Out of scope for the app: auth, user accounts, extra databases, mobile layout, deployment pipelines. Result: README screenshots plus a short screen recording. Hosting is optional later.
- Model judgment and policy decision remain separate in code (separate modules) and UI (separate panels).

## 10. Roadmap

Effort numbers are rough guesses for part-time work, to be revised as we go.

| Module | What | Done when | Rough effort |
|---|---|---|---|
| **0 Setup** | Repo, environment, project charter, experiment protocol | Fresh clone installs and runs a smoke test; protocol doc written | 1–2 sessions |
| **1 Dataset + EDA** | Load Jigsaw, imbalance analysis, train/val split saved by ID, Val-J / Test-J construction rules and weights written (test untouched except for row IDs/labels needed to build Test-J, see note) | Split files and subset files exist, reproducible from seed; EDA findings recorded | 2–3 sessions |
| **2 Baselines** | TF-IDF + LR, DistilBERT; validation predictions saved; thresholds from Val-J | Both models produce 6 probabilities per comment with saved artifacts | 3–4 sessions |
| **3 JEV integration** | Client, SQLite cache, logging; **cost/feasibility Gate**; question pilot on train-only; freeze + tag questions; generate Val-J then Test-J predictions | All Val-J and Test-J probabilities cached; cost/latency logs complete | 4–6 sessions |
| **4 Comparative case study** | Three systems on identical Test-J rows: metrics, CIs, calibration, rare classes, errors, latency/cost, sanity check | Results tables/figures saved as artifacts; negative results reported as found | 2–3 sessions |
| **5 Policy experiment** | Policy engine, threshold trade-offs, exported JSON artifacts, API contract | Weighted trade-off tables from validation-frozen thresholds applied to Test-J | 2 sessions |
| **6 Console** | FastAPI + React, three views | All three views work from real artifacts; demo mode works | 6–9 sessions |
| **7 Final case study + portfolio** | README, architecture diagram, results, screenshots, case-study write-up, summary bullets (no fabricated metrics) | A stranger can understand and run the project from the README | 2–3 sessions |

Note on Test-J construction: building Test-J needs the test labels (to find the rare stratum). That is allowed because it only selects rows. No model predictions or metrics are looked at, and the selection rule is fixed and written down before any test evaluation.

**Cut line if time runs short** (drop in this order): (1) optional JEV calibration row, (2) latency/cost chart, (3) Test-J toggle in the Simulator, (4) polish on the Benchmark view (a static results page is acceptable). **Never cut:** the leakage rules, identical-row comparison, caching, weighted reporting, or the policy/judgment separation.

## 11. Explicitly out of scope

Research-paper submission; second dataset (Civil Comments etc.); extensive hypothesis testing; multiple transformer architectures; large JEV formulation ablations; extensive repeatability studies or multi-seed sweeps; systematic literature review; publication-grade expansion. These may be future extensions only after the core project is finished.

## 12. Repository layout

```
policyjev/
  README.md
  docs/            charter.md, experiment_protocol.md, decisions.md
  configs/         project.yaml, policy.yaml (jev questions added in Module 3)
  data/            raw/ (gitignored), processed/ (split IDs, subsets, weights)
  src/policyjev/   data/, baselines/, jev/, evaluation/, policy/
  notebooks/       EDA and exploration only
  artifacts/       predictions/, metrics/, exports/
  backend/         FastAPI app (imports src/policyjev)
  web/             React + TypeScript + Tailwind (Vite)
  tests/
```

## 13. Methodological rules

- No data leakage; test set untouched until the protocol is frozen.
- Thresholds and calibration chosen on validation (Val-J) only.
- Correct multi-label setup (sigmoid + per-label BCE).
- Reproducibility: fixed seeds, saved split IDs, pinned dependencies, config-driven runs.
- JEV outputs cached; never pay twice; JEV never trains the other models.
- Same Test-J rows for all three systems; weighted headline metrics; raw positive counts shown.
- Real experiment values in the final UI.
- Report disappointing or negative results as they are; never alter the method to rescue a result.
- Claims in the README and write-up use only measured numbers.
