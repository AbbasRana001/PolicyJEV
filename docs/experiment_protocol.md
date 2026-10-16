# Experiment protocol

Status: ready to freeze. The freeze is the git tag protocol-frozen on the commit that contains this file and docs/frozen_manifest.csv. The tag is created before the test set is scored and before any system produces test predictions. After the tag, methodology changes require a dated amendment; the only changes allowed without one are the corrections described in section 10.

## 1. Data
- Dataset: Jigsaw Toxic Comment Classification. Labels: toxic, severe_toxic, obscene, threat, insult, identity_hate.
- Raw data is never committed. Only code, configs, split IDs and aggregate results are.
- Test rows labeled -1 were never scored and are dropped.

## 2. Splits
- train.csv is split about 85% train / 15% validation, stratified so rare labels appear in both. The seed is in configs/project.yaml. Row IDs are saved to disk and never regenerated.
- Validation is the whole 23,936-row validation split. It is the only place where thresholds, calibration, the stacked secondary system or any other tuning happens.
- Test is the whole official labeled test set (-1 rows dropped). It is not opened (no EDA, no predictions, no thresholds) until this protocol is frozen and tagged and all models, calibrators and thresholds are final. It is then evaluated once. Code that reads test predictions, labels or text refuses to run unless the protocol-frozen tag exists.
- At the freeze, docs/frozen_manifest.csv records SHA-256 hashes of the files that define the methodology: the question files, the saved calibrators and stack, the validation thresholds table, the logistic regression model and the DistilBERT weights. Code that produces or evaluates test predictions verifies the entries it uses and refuses to run on a mismatch.

## 3. JEV scoring on full splits
- JEV scores every validation row and, after the freeze, every test row. There are no JEV-specific subsets and no sample weights, and every system is evaluated on the same full rows.
- Why: an earlier design scored stratified subsets (Val-J and Test-J) with sampling weights and a weighted-versus-full sanity check. The measured JEV cost made full splits affordable, and scoring everything removes the sampling error and the weighting machinery. The subset outputs are superseded.
- The subset files and code were removed from the repository (history retains them) and are recorded in decisions.md.
- Every scored row is stored in the local cache keyed by normalized text, question set version and model; a row is never paid for twice.
- The question pilot uses train-split comments only, so it is disjoint from validation and test.
- Only question set v2, selected by the rule in Amendment 1, is scored on test.
- The JEV model identifier is pinned to jev-1.13.0. If the vendor retires it before the test run, the test run stops and an amendment decides what happens next.
- If any test row cannot be scored by JEV after the configured retries, the failure count is reported, the row is not dropped silently, and all systems are evaluated on the rows common to all of them, with the count stated.

## 4. Systems
Primary systems:
- A: TF-IDF with one-vs-rest logistic regression.
- B: DistilBERT multi-label (sigmoid outputs, per-label binary cross-entropy).
- C: TypeSafe JEV (model jev-1.13.0, pinned), six questions, question set v2 (see the amendments). JEV is zero-shot: it is not trained on Jigsaw. It does use validation labels (23,936 comments) in three places: the question set was selected among three pre-registered candidates by validation average precision, its probabilities are calibrated on validation, and its decision thresholds are fit on validation.

Secondary system:
- D: jev_stack, a supervised system. For each label, a logistic regression on JEV's six v2 probabilities (standardized features), fit on all validation rows only and applied once to test. It uses validation labels, so it is always labeled supervised and secondary and is never reported as JEV. Its probabilities are the outputs of its logistic regression and receive no further calibration; its F1 thresholds are fit on its 5-fold out-of-fold validation predictions (same grid step and tie rule as the other systems) and applied to its test probabilities.

## 5. Rules
1. Thresholds and calibration are fit on validation only. Per system, the calibration method (Platt or isotonic) is the one with the lower mean 5-fold out-of-fold Brier score on validation; the chosen calibrator is then fit on all validation rows and applied to test once. Per-label F1 thresholds are fit on out-of-fold calibrated validation scores (grid step 0.01, lower median of ties) and applied to test once. A fixed 0.5 threshold is also reported.
2. All systems are compared on identical rows.
3. Every rare-label metric is shown with its raw positive count.
4. Differences between systems are reported as paired bootstrap intervals: each resample draws rows once and applies them to every system.
5. JEV outputs are never used to train or tune the three primary systems. The only system fit on JEV outputs is the secondary system D, which is labeled supervised.
6. JEV is not deterministic: in the pilot, repeated calls on the same comment differed by up to 0.03 at a resolution of 0.01. The first cached result is canonical. A second JEV pass on a fixed random subset of 5,000 test rows (seed = project seed + 11) is reported as run-to-run variation and is not used in any headline metric.
7. Question wording is frozen: v1 under the questions-frozen tag, and v2 and v3 under the questions-alt-frozen tag. No set is revised after it is scored. Only v2 is scored on test.
8. Results are reported as found. The method is not changed to improve a result.
9. Claims in the README, CV or portfolio use measured numbers only.

## 6. Metrics
- Micro and macro F1; per-label precision, recall, F1 and average precision.
- Average precision is computed on raw scores and uses the same implementation for every system. JEV probabilities take 101 discrete values, so average precision involves more ties than for the trained models; a tie control rounds the baselines' probabilities to two decimals.
- Brier score on calibrated probabilities (per label and mean), with the raw Brier score alongside. Reliability tables use five equal-count bins, per system and per label. Reliability plots use fixed probability bins (edges 0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1), because equal-count bins put almost all positives of these imbalanced labels in one bin; they are shown per system and per label, with an additional pooled panel.
- Rare labels are threat, severe_toxic and identity_hate.
- Paired bootstrap 95% percentile intervals (1,000 resamples, seeded) for average precision and for the differences between systems, per label and for the mean. F1 at thresholds fitted on validation is labeled in-sample when reported on validation.
- Latency and cost. Baselines (A, B): p50 and p95 per-comment latency at batch size 1 and throughput at batch size 32, on a fixed random sample of 500 validation comments (project seed), after warm-up, on CPU and on GPU when available, with the machine described in the output. JEV (C): p50 and p95 request latency and cost per 1,000 comments from the recorded calls (tokens times the listed price). The baselines have no API cost, so only compute time is reported for them. JEV latency is a network round trip including queueing under concurrent requests, so it is reported but not ranked against local inference.
- Error analysis: qualitative, on validation only, comparative rather than per system. Per label, seeded samples of up to 25 comments from four categories: (a) JEV false negatives that both baselines catch, (b) JEV hits that both baselines miss, (c) false positives of JEV only, and (d) possible label noise, meaning all three systems agree with each other against the gold label (all three above their fitted thresholds with gold 0, or all three below with gold 1). Counts per category are reported. Comment text is shown in the terminal only and never written to files or committed.

Secondary analyses (pre-registered):
- (a) Sensitivity to question wording: question set v3 (reworded instructions only) is scored on validation only and compared with v1 and v2.
- (b) Review-budget analysis for the policy module. Each comment is scored by the maximum calibrated probability across the six labels. Budgets are 1, 2, 5, 10 and 20 percent of all comments. Metrics: recall_any (share of comments with any positive label that land in the reviewed set), precision of the reviewed set, and recall_rare (the same recall restricted to comments positive for threat, severe_toxic or identity_hate). The pre-specified contrasts are JEV against each primary baseline on recall_any and recall_rare at the 5 and 10 percent budgets. The hypothesis generated on validation is that JEV has higher recall_rare than DistilBERT at the 1 and 2 percent budgets. All other cells are descriptive. The report states that many intervals are shown and none is adjusted for multiplicity. Ties in the review score are broken with one seeded random vector shared across systems and bootstrap resamples.
- (c) Secondary system D (jev_stack) is compared with the three primary systems with paired intervals and is labeled supervised.
- (d) Run-to-run variation of JEV, from the second pass in rule 6: mean absolute probability change per label, and the change in average precision and in F1 at the validation thresholds.
- (e) Disagreement and error analysis between JEV and DistilBERT stays on validation only; test rows are not inspected.

## 7. Moderation policy
- p < LOW: ALLOW. LOW <= p < HIGH: HUMAN REVIEW. p >= HIGH: ACTION.
- Thresholds may differ per category. A comment gets its most severe decision across the six labels.
- Thresholds are chosen on validation and applied to test once. Policy numbers use natural prevalence, with no weights.
- How LOW and HIGH are chosen is fixed in a dated amendment written during the policy module from validation data only, before any policy number is computed on test. The policy is then evaluated on test once.

## 8. Out of scope
Paper submission, other datasets, multiple transformer architectures, multi-seed sweeps, formal significance tests (paired bootstrap intervals are reported instead).

## 9. Limitations
- The bootstrap treats comments as independent, so correlation between comments from the same thread is not modeled.
- Many intervals are shown, and only the pre-specified contrasts are interpreted.
- Jigsaw labels are noisy.
- One dataset is used.
- DistilBERT was trained in one run, so seed variance is not measured.
- The baselines received a modest hyperparameter search.
- JEV is non-deterministic: the first cached result is canonical, and the 5,000-row second pass quantifies the variation.
- No zero-shot general LLM prompting comparison is included.
- Comments were sent to a third-party API.
- The data is English-language.

## 10. Post-freeze corrections
The freeze fixes the methodology: splits, question set v2, calibration method per system, threshold rule, metrics, bootstrap settings and the pre-specified contrasts. Bug fixes to code that reads test predictions or applies saved calibrators are allowed after test predictions exist, without re-scoring JEV and without changing any method choice. Each such fix is logged in docs/decisions.md with the date, the reason and the effect on any reported number.

## 11. Amendments

### Amendment 1 (2026-10-12): Additional JEV question sets and a selection rule
- Trigger: question set v1 was scored on the full validation set, with mean average precision 0.541 against 0.644 for logistic regression and 0.724 for DistilBERT. After that, the vendor documentation was re-read, and the optional criteria field, which the documentation recommends for domain rules and boundary cases, was found unused in v1.
- Two further sets are pre-registered: v2 (short instructions plus criteria with a true and a false description per question) and v3 (reworded instructions only, no criteria, to measure sensitivity to wording).
- v2 and v3 were written once, from the label names and the vendor guidance, without inspecting validation errors. They will not be revised after scoring. They are frozen under the questions-alt-frozen tag before they are scored.
- Selection rule: the question set with the highest mean validation average precision is used for the test run, except that v1 is kept if the best alternative beats it by less than 0.005 mean average precision.
- The validation results of all three sets are reported. Test is scored with the selected set only.
- Disclosure: this amendment was made after seeing the v1 validation results, so three question sets are tried in total. The baselines also had hyperparameters selected on validation (the logistic regression C and the DistilBERT epoch), so selecting the wording on validation is the symmetric treatment.
- As a control for ties, baseline average precision is also computed after rounding probabilities to two decimals, because JEV probabilities take only 101 distinct values.
- Outcome: v2 was selected. Mean validation average precision was 0.5659 for v2, 0.5409 for v1 and 0.5377 for v3; v2 beat v1 by 0.0250 (95% paired-bootstrap interval 0.0144 to 0.0361), above the 0.005 margin.

### Amendment 2 (2026-10-12): Final protocol, secondary system, second pass
Made after the validation analyses (calibration, review budget and the exploratory JEV headroom analysis) and before any test prediction exists.
- (a) Secondary system D (jev_stack) is added: a logistic regression on JEV's six v2 probabilities, fit on all validation rows only and applied once to test. It is secondary and labeled supervised, and it is never reported as JEV. Rule 5 is amended to say that JEV outputs do not train or tune the three primary systems.
- (b) The ensemble of question sets is dropped: on validation it showed no gain over v2 (mean average precision difference -0.0004, interval -0.0085 to 0.0088). Only v2 is scored on test.
- (c) The review-budget analysis is pre-registered as a secondary analysis with budgets 1, 2, 5, 10 and 20 percent, with the pre-specified contrasts, the validation-generated hypothesis and the multiplicity statement given in section 6.
- (d) Run-to-run variation: a second JEV pass is run on a fixed random 5,000 test rows (seed = project seed + 11) and reported as mean absolute probability change per label and the change in average precision and in F1 at the validation thresholds. The first pass stays canonical.
- (e) Disagreement and error analysis stays on validation only.
- (f) The validation findings behind these decisions are recorded as measured in decisions.md.

### Amendment 3 (2026-10-13): Moderation policy rule, thresholds and test metrics
Written after the test run and the results report, using validation data only. No policy number has been computed on test. This amendment replaces the open choice of LOW and HIGH in section 7.
- Systems: logistic regression (logreg_word), DistilBERT and JEV (question set v2). The supervised stack is excluded.
- Inputs: calibrated probabilities from the frozen calibrators in artifacts/calibrators/ (applied to test). Thresholds are fit on the out-of-fold calibrated validation probabilities, the same source used for the frozen F1 thresholds (src/policyjev/evaluation/frozen_thresholds.py). These probabilities are not stored in a file: they are computed in memory by the shared validation loader from the raw validation predictions (artifacts/predictions/logreg_word_validation.csv, distilbert_validation.csv and jev_validation_v2.csv), the calibration method per system recorded under calibration.method in configs/project.yaml, and the project seed, using 5-fold out-of-fold calibration.
- Decision rule: for each label, p < LOW gives ALLOW; LOW <= p < HIGH gives HUMAN REVIEW; p >= HIGH gives ACTION. The comment's decision is the most severe outcome across the six labels (ACTION over HUMAN REVIEW over ALLOW). The decision records which labels triggered it: the labels whose own outcome equals the comment's decision.
- Threshold selection, per system and per label, from validation data only. Candidate thresholds are the distinct calibrated probability values in the validation data. HIGH is the lowest candidate with validation precision at or above the precision target and at least 30 validation comments at or above it; if none qualifies, ACTION is disabled for that label (HIGH = infinity). LOW is the highest candidate with validation recall at or above the recall target for that label. If LOW is above HIGH, LOW is set to HIGH. Ties use the rule p >= threshold. The selected thresholds are written to artifacts/policy/policy_thresholds.csv, which is added to the freeze manifest.
- Primary targets are recall 0.90 and precision 0.90. A pre-specified sensitivity grid of recall in {0.80, 0.90, 0.95} by precision in {0.80, 0.90, 0.95} (nine points, one of which is the primary point) is secondary and descriptive.
- Natural prevalence is used, with no sample weights.
- Primary test metrics, to be computed once for each system at the primary targets: (M1) review share, the share of all comments with decision HUMAN REVIEW; (M2) ACTION precision, the share of ACTION comments in which at least one label that reached its HIGH is truly positive (undefined if a system takes no ACTION); (M3) miss rate, the share of comments positive for at least one label that receive ALLOW; (M4) rare miss rate, the share of comments positive for at least one of threat, severe_toxic or identity_hate that receive ALLOW. Descriptive metrics: ACTION share, ACTION coverage (the share of any-positive comments that receive ACTION), and per-label versions of these.
- Pre-specified contrasts: JEV minus logistic regression and JEV minus DistilBERT on M1, M2, M3 and M4, using the same paired bootstrap as the rest of the protocol (1,000 resamples, seeded, 95% percentile intervals). A bootstrap resample in which a quantity is undefined for either system is left out of that quantity's interval, and the number left out is reported.
- Rules against changes: the targets, the support floor and the grid are fixed by this amendment. No threshold is refit and no target is changed after test results are computed. Failed or undefined quantities are reported as such.
- Disclosure: when this amendment was written, the test results for average precision, F1, calibration and the review-budget analysis had already been seen. The rule and the targets were chosen from validation data and conventional round numbers, not from those results.
- Freeze: the policy freeze is the git tag policy-frozen on the commit that contains this amendment, artifacts/policy/policy_thresholds.csv and the matching row in docs/frozen_manifest.csv. Code that computes policy numbers on test refuses to run unless that tag exists and the manifest entries match.
