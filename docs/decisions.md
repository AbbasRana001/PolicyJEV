# Decision log

Each entry records what was decided, why, and when. Entries are appended; earlier entries were corrected in places before publication.

## 2026-10-10: Python 3.13 for the project environment
Python 3.13 and 3.14 were both installed locally. PyTorch's install page lists Windows support for Python 3.10 to 3.14, so either should work; 3.13 was chosen as the older, more settled release. The requirement is declared in pyproject.toml (requires-python >= 3.13).

## 2026-10-10: Dependencies pinned in requirements.txt
Exact versions are recorded with pip freeze, and packages are added module by module. After the editable install exists, regenerate the file with `python -m pip freeze --exclude-editable` so the project does not list itself.

## 2026-10-10: SQLite files are gitignored
The JEV cache stores a hash of the normalized text, the comment id, the answers and token counts, not the comment text (see src/policyjev/jev/cache.py). Dataset text is not committed (charter section 3). So *.sqlite and *.db are ignored along with data/raw/.

## 2026-10-11: Line endings normalized to LF
.gitattributes sets LF for text files, so history is identical whether it is written on Windows or elsewhere.

## 2026-10-11: Train/validation split created
Seed 42, 15% validation, stratified on each comment's rarest label (order: threat, identity_hate, severe_toxic, insult, obscene, toxic). Sizes: 135,635 train / 23,936 validation. Per-label rates match within chance (artifacts/metrics/split_summary.csv). Validation holds only 72 threat positives, so threat thresholds fit on Val-J will be noisy; they are reported with intervals and raw counts.

## 2026-10-11: Val-J and Test-J built with nested sampling
All rare-stratum rows are kept (weight 1). Common-stratum rows are taken in a fixed seeded order, so a smaller subset is always a subset of a larger one with the same seed. If the cost pilot shrinks N, the same code is re-run with a smaller size. Initial targets: Val-J 1,500, Test-J 3,500 (the subset design was later dropped; see the entries below).

## 2026-10-11: No deduplication
Exact-text duplicates are rare (375 train rows; 56 validation rows share text with train, 0.23% of validation). Rows are kept as they are: removing them would change a split that is already frozen and committed, and at most 0.23% of validation rows are affected.

## 2026-10-11: Baseline training rules
The TF-IDF vectorizer is fit on the train partition only. Logistic regression uses no class weights, so probabilities stay interpretable; thresholds handle imbalance. C is chosen by mean per-label average precision on the full validation set. There is no refit on train plus validation, so validation stays held out for thresholds and calibration. Model files are written to artifacts/models/, which is gitignored.

## 2026-10-11: Training matrix canonicalized before parallel logistic regression fits
TfidfVectorizer output has unsorted column indices when max_features limits the vocabulary. scikit-learn's LogisticRegression sorts them in place, which fails inside joblib workers because large arrays are shared read-only. fit_logreg now calls sum_duplicates() on the training matrix first. On synthetic data the fitted coefficients were identical to a single-process fit.

## 2026-10-11: Logistic regression baseline (word 1 to 2-grams)
C was chosen by mean per-label average precision on the full validation set: C=4 gave 0.644 (C=2: 0.638, C=8: 0.643), so the choice is not sharp. Per-label results are in artifacts/metrics/logreg_word_validation.csv. Character n-grams (config switch char_ngrams) have not been run.

## 2026-10-11: DistilBERT fine-tuning settings
Checkpoint distilbert-base-uncased, initialised from the generic pretrained weights (not a checkpoint already fine-tuned on Jigsaw data). Per-label sigmoid with BCEWithLogitsLoss, AdamW (learning rate 2e-5, weight decay 0.01 except bias and LayerNorm), linear warmup of 6% then linear decay, batch size 32, up to 3 epochs, gradient clipping at 1.0, fp16 on GPU. One seed (42), reported as a single run. The epoch with the highest mean validation average precision is kept. GPU training is not bit-for-bit reproducible. The maximum sequence length is 256, chosen from artifacts/metrics/token_lengths.csv:
- Token counts on the train partition (artifacts/metrics/token_lengths.csv): 128 tokens would truncate 19.3% of comments (13.3% of severe_toxic); 256 truncates 6.9% (9.4% of severe_toxic); 512 truncates 2.2%.
- With random batches nearly every batch pads to max_length, so cost scales with it; 256 is about twice the compute of 128.
- 256 was chosen so that truncation does not handicap the supervised baseline relative to the logistic regression, which sees the whole text.

## 2026-10-12: Threshold selection
Per label, the threshold maximizes weighted F1 on Val-J over a 0.01 grid, with the lower median taken among tied grid points. Thresholds are fit on Val-J only and applied once to Test-J. Threat has 72 positives in Val-J, so its threshold is noisy.

## 2026-10-12: DistilBERT run
Trained on a Colab T4 (torch 2.11.0+cu130, transformers 5.18.0; the local environment has torch 2.14.1 and transformers 5.19.0), 48.6 minutes, max_length 256, 3 epochs. Validation mean AP 0.724 against 0.644 for logistic regression. The best epoch was the last, so the run may not be converged; no further tuning was done because the baseline settings were fixed in advance.

## 2026-10-12: JEV access facts
Verified from the vendor docs on 2026-10-12: the model is pinned to jev-1.13.0, because jev-latest is an alias that can change answers when a new release ships. The docs do not state whether answers are deterministic.

## 2026-10-12: JEV spending controls
Spend was capped by a hard stop that the client enforces across all runs; the worst-case overshoot is a few in-flight requests. The limit and the rate used to track spend are not committed: they are read from TYPESAFE_HARD_STOP and TYPESAFE_COST_RATE in the environment or the local .env file (see .env.example). The cache is the source of truth for spend and latency. It is gitignored, so it is backed up outside the repository.

## 2026-10-12: Question pilot design and wording rules
The pilot is 150 train-split comments (seed 45, unweighted, stored as ids only in data/processed/pilot.csv): 20 each for threat, severe_toxic and identity_hate (a row counts toward the first of those labels it is positive for, in that order), 40 positive for toxic, obscene or insult with no rare label, and 50 clean. It is disjoint from Val-J and Test-J, which are drawn from validation and test. Its purpose is limited to cost, sanity, determinism and spotting gross wording failures (for example a question that returns the same probability for every comment). It is not used to estimate performance or prevalence, and not to tune wording for a higher score.

Wording rules: question wording may change only on pilot evidence, with at most two revisions after v1. Every version stays in git (configs/questions_v1.yaml, then v2 and v3 as separate files). Val-J and Test-J are never used to choose wording. The final version is frozen with the questions-frozen tag before any Val-J run.

## 2026-10-12: Question set v1 frozen after the pilot
Question set v1 is the final wording; no revisions were made. Reasons:
- The pilot showed no gross failure: probabilities vary across comments and the questions separate positives from negatives.
- JEV is not deterministic. Repeated calls on the same comment differed by up to 0.03, at a resolution of 0.01. Single-call probabilities therefore carry a small amount of noise, and the first cached result is canonical.
- severe_toxic reached a pilot average precision of 0.38 against a 0.17 base rate in the pilot. That is weak but well above chance. (The pilot report was regenerated after this entry was written; artifacts/metrics/jev_pilot_report.csv now shows 0.328 for severe_toxic.)
- Wording was not revised, because the pilot has only 20 to 26 positives per rare label, so any change driven by those numbers would mostly tune to noise.

Measured: 535 input tokens per request on average. At that size, scoring the whole validation split and the whole labeled test set is affordable under the spending cap, so full-split runs are being considered in addition to Val-J and Test-J. The runner supports them as val_full and test_full with the same git-tag guards as val_j and test_j, and rows already scored for the pilot or a subset are reused from the cache. The hard stop is unchanged for now.

## 2026-10-12: Full splits replace the Val-J and Test-J subsets
The experiment now scores the whole validation split and, after the protocol freeze, the whole labeled test set. The full validation run scored 23,936 rows at 519 input tokens per request, with 0 failures. The low cost made the subsets unnecessary: scoring everything removes sampling error and the need for sample weights and the weighted-versus-full sanity check. The protocol (sections 2, 3, 5, 6, and the stale references in 4 and 7) is rewritten accordingly. The Val-J and Test-J files and the Val-J outputs were superseded and are not in the published repository (see the entry below that records their removal); thresholds are now fit on all validation rows (artifacts/metrics/thresholds_full_validation.csv). The cumulative hard stop was raised; it is set outside the repository.

## 2026-10-12: Question sets v2 and v3 added after seeing v1 validation results
Question set v1 scored a mean validation average precision of 0.541, against 0.644 for logistic regression and 0.724 for DistilBERT. Re-reading the vendor documentation showed that the optional criteria field, recommended for domain rules and boundary cases, was unused in v1. Two sets are therefore added: v2 (short instructions plus true/false criteria per question) and v3 (reworded instructions, no criteria, to measure wording sensitivity). Both were written once without inspecting validation errors and will not be revised after scoring. Selection rule: the highest mean validation AP is used for the test run, and v1 is kept if the best alternative beats it by less than 0.005. All three sets are reported, and three question sets are tried in total, which is disclosed in the protocol amendments. Support in the code: criteria are sent as a mapping with the string keys "true" and "false" (YAML keys must be quoted, and unquoted ones are rejected); the cache key already includes criteria. The runner requires the questions-frozen tag for v1 and the questions-alt-frozen tag for v2 and v3, and JEV scores are exported per version (artifacts/predictions/jev_validation_<version>.csv). jev.primary_question_version in configs/project.yaml chooses the version used by the thresholds and the validation report.

## 2026-10-12: Question set v2 selected by the pre-specified rule
All three sets were scored on the full validation split. Mean validation average precision: v2 0.5659, v1 0.5409, v3 0.5377. v2 beats v1 by +0.0250 (95% paired-bootstrap interval 0.0144 to 0.0361), which is above the 0.005 margin, so v2 is selected and is used for the test run (jev.primary_question_version is v2). v3 differs from v1 by -0.0031 (interval -0.0121 to +0.0058).

Wording sensitivity (v3 against v1, per label; v3 changes only the instructions): toxic -0.034 (-0.040 to -0.030), severe_toxic -0.016 (-0.026 to -0.005), obscene -0.008 (-0.017 to +0.002), threat +0.059 (+0.014 to +0.107), insult -0.034 (-0.046 to -0.023), identity_hate +0.014 (-0.006 to +0.033). Rewording alone therefore moves single-label AP by up to about 0.06 in either direction, even though the mean change is close to zero. For comparison, adding criteria (v2 against v1) changed toxic by +0.033, severe_toxic by +0.049, threat by +0.049 (interval -0.001 to +0.098), identity_hate by +0.022 and obscene and insult by about zero. Per-label rankings among the three sets are not stable enough to pick wording label by label, and the sets are not mixed.

Tie control: JEV probabilities take 101 distinct values, so ties could lower its AP relative to the trained models. Rounding the baselines' probabilities to two decimals lowers mean validation AP by 0.009 for logistic regression (0.644 to 0.635) and 0.007 for DistilBERT (0.724 to 0.717). Ties therefore explain a small part of the gap, not most of it.

Measured: question set v2 averages 891 input tokens per request and v3 averages 500 (v1 averaged 519 on the full validation run). The cumulative hard stop was raised again; it is set outside the repository.

## 2026-10-12: Calibration, review-budget, disagreement and headroom analyses
Added, working on validation now and reusable for the test split later; anything that reads test predictions or labels refuses to run unless the protocol-frozen git tag exists. Calibration (Platt and isotonic, seeded 5-fold out-of-fold on validation) is chosen per system by lower mean out-of-fold Brier and recorded in configs/project.yaml under calibration.method; the chosen calibrator is fit on all validation rows and saved as JSON in artifacts/calibrators/. Calibrated probabilities feed the Brier column, the review-budget analysis (score = maximum calibrated probability across labels) and the disagreement analysis; thresholds for F1 and disagreement are fit on out-of-fold calibrated validation scores. Average precision stays on raw scores because it depends only on ranking. The JEV headroom analysis (ensemble of v1, v2 and v3, and cross-validated stacking) is exploratory, uses validation only, is not a headline result, and the stacked variants use validation labels through cross-validation.

## 2026-10-12: Validation findings behind Amendment 2
Calibration (5-fold out-of-fold on validation). Chosen methods, recorded in configs/project.yaml: logistic regression isotonic, DistilBERT Platt, JEV (v2) isotonic. Mean Brier, raw to chosen out-of-fold: logistic regression 0.01403 to 0.01345, DistilBERT 0.01112 to 0.01072 (isotonic 0.01076), JEV 0.05846 to 0.01621 (Platt 0.01630). JEV's raw probabilities are badly calibrated and calibration removes most of that; after calibration JEV's Brier is still above both baselines.

Review budget (validation, calibrated max-probability score, 95% paired-bootstrap intervals). Pre-specified contrasts, JEV minus logistic regression: recall_any at 5% -0.017 (-0.023 to -0.009), at 10% -0.029 (-0.047 to -0.015); recall_rare at 5% +0.016 (-0.028 to +0.050), at 10% -0.031 (-0.060 to +0.002). JEV minus DistilBERT: recall_any at 5% -0.028 (-0.035 to -0.021), at 10% -0.092 (-0.106 to -0.079); recall_rare at 5% -0.018 (-0.054 to +0.018), at 10% -0.054 (-0.078 to -0.024). Hypothesis generated on validation, JEV minus DistilBERT recall_rare: +0.062 (+0.026 to +0.101) at the 1% budget and +0.096 (+0.051 to +0.141) at 2%; it is to be tested on test, not treated as established. At the 20% budget DistilBERT has the highest recall_any (0.975 against 0.940 for JEV and 0.935 for logistic regression). Many intervals are shown and none is adjusted for multiplicity.

JEV headroom (exploratory, validation only; mean AP, difference to v2 in brackets). Ensemble mean of v1, v2 and v3: 0.5655 (-0.0004, interval -0.0085 to +0.0088), so the ensemble is dropped. Stacked logistic regression on the six v2 probabilities: 0.6047 (+0.0389, interval +0.0204 to +0.0582); on all 18 probabilities: 0.6145 (+0.0486, +0.0303 to +0.0695). The stack still trails the baselines: stack_v2 minus logistic regression -0.0396 (-0.0670 to -0.0113) and minus DistilBERT -0.1189 (-0.1447 to -0.0927); stack_all minus logistic regression -0.0299 (-0.0572 to -0.0006) and minus DistilBERT -0.1092 (-0.1350 to -0.0813). The stack uses validation labels, so it is a supervised result. Only the six-feature stack is adopted, as secondary system D.

## 2026-10-12: Amendment 2 to the experiment protocol
The protocol is finalized and ready to freeze. It adds the secondary supervised system jev_stack, drops the question-set ensemble, pre-specifies the review-budget contrasts and the validation-generated hypothesis, fixes the second JEV pass at 5,000 random test rows with seed = project seed + 11 (reported as run-to-run variation, first pass canonical), keeps disagreement analysis on validation, and amends the leakage rule to say that JEV outputs do not train or tune the three primary systems. Code that touches test predictions, labels or text refuses to run without the protocol-frozen tag. Reasons are in the preceding entry.

## 2026-10-12: Baseline models available locally
The logistic regression is saved at artifacts/models/logreg_word.joblib (vectorizer and classifier, C=4). The DistilBERT weights are not on this machine: that model was trained on Colab and its folder (artifacts/models/distilbert, written by the training script) must be copied back or run on Colab for test inference. Test predictions are produced after the freeze with python -m policyjev.baselines.predict_test.

## 2026-10-12: Superseded Val-J and Test-J work removed
Nothing current depended on the subset design after the switch to full splits, so it was removed (git history retains it). Removed: src/policyjev/data/subsets.py and tests/test_subsets.py; the val_j and test_j runner sets and their guards; the Val-J threshold mode of src/policyjev/evaluation/thresholds.py (its CLI now only has --split full_validation); the jev_subsets block of configs/project.yaml; data/processed/val_j.csv and test_j.csv; artifacts/metrics/thresholds.csv and subset_summary.csv. The pilot script kept its never-overwrite save as save_pilot in src/policyjev/jev/pilot.py and now checks disjointness from the validation split instead of the subset files. Kept on purpose: the weighted precision/recall/F1 helpers and fit_thresholds in thresholds.py (used with unit weights by the full-validation code), data/processed/pilot.csv, and the historical entries in this log. docs/charter.md is a historical planning document and carries a note that the protocol governs; its Val-J and Test-J passages were not rewritten.

## 2026-10-12: JEV described as zero-shot, not training-free
JEV is not trained on Jigsaw, but it uses validation labels in three places: the question set was selected among three candidate question sets, each frozen before it was scored, by validation average precision, its probabilities are calibrated on validation, and its thresholds are fit on validation. The protocol (section 4), README and charter note now say so. The word "training-free" remains only where it accurately means "no model fitting", for the exploratory ensemble mean in the headroom analysis.

## 2026-10-12: Freeze tooling
Added: a freeze manifest (docs/frozen_manifest.csv, written with python -m policyjev.freeze_manifest write) with SHA-256 hashes of the question files, saved calibrators and stack, the validation thresholds table, the logistic regression model and the DistilBERT weights; a frozen validation thresholds table (artifacts/metrics/validation_thresholds.csv, written with python -m policyjev.evaluation.frozen_thresholds) that the test code now reads instead of recomputing thresholds; and verification of the entries each consumer uses (baseline test inference, the JEV runner and second pass for test_full, and every analysis run with --split test through the shared loader). New validation-only tools: reliability and summary figures (reliability_plots), latency and cost (latency), and the qualitative error review (error_review).

## 2026-10-12: Aggregate counts seen during the Val-J and Test-J cleanup
During the Val-J/Test-J cleanup, a repository-wide search printed two aggregate counts from the deleted subset_summary.csv: the test row count (63,978) and the Test-J rare-row count (1,080). No test labels, predictions or comment text were read, and no method choice depends on these counts.

## 2026-10-12: Reliability plots use fixed probability bins
The reliability diagrams now use fixed probability bins with edges 0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1 instead of five equal-count bins, because equal-count bins put almost all positives of these imbalanced labels in one bin and the other bins show nothing. Only bins with at least 20 comments are plotted, plotted bins with fewer than 20 positives are drawn with open markers and named low support in the legend, and a lower strip shows comments per bin on a log scale. Axes are labeled (mean predicted probability, observed rate) with equal limits and a y = x reference line. The five-equal-count-bin reliability table (calibration_reliability_validation.csv) is unchanged. Protocol section 6 was updated to match.

## 2026-10-12: Manifest hashes normalize line endings in text files
Hashing previously used the raw bytes of every file, so a CRLF and an LF checkout of the same text file would differ. Text files (.csv, .json, .yaml, .yml, .md, .txt) are now hashed after replacing CRLF with LF; binary files (.safetensors, .joblib and anything else) are hashed unchanged. No manifest had been written before this change.

## 2026-10-12: Baseline test inference sorts by length
predict_test batches comments sorted by tokenized length (maximum 256 tokens), pads each batch to its longest item and restores the original row order for the output. This was already how predict_distilbert worked; a test now checks that its outputs match the unsorted path within 1e-5. Nothing else in the test path changed.

## 2026-10-12: Reliability plot layout and wording
Layout and wording only; no metric, binning, threshold or number changed. In the reliability figures the legend row now sits above the caption with reserved margins so no text overlaps, the strip x-labels are shortened (<.01, .01-.05, .05-.1, .1-.2, .2-.4, .4-.6, .6-.8, >.8), the repeated "probability bin" label under each strip was removed and "bins by predicted probability" appears once in the caption, spacing between panel rows and between each panel and its strip was reduced (figure size 14 x 8.5 inches), and markers at the axes edge are no longer clipped. Protocol section 6 now says that reliability tables use five equal-count bins per system and per label, and that reliability plots use fixed probability bins per system and per label with an additional pooled panel. This replaces the earlier wording that described both as shown pooled, which the table does not do.

## 2026-10-13: Post-freeze correction (protocol section 10): comparison report console header
Reason: with --split test, src/policyjev/evaluation/comparison_report.py printed the header "Full-validation comparison (all validation rows, unweighted)", which names the wrong split. The header is now built from the split name ("Full-test comparison (all test rows, unweighted)" for test; the validation text is unchanged). Effect: console text only. No metric, CSV or figure changed, and no test scoring was re-run. docs/experiment_protocol.md is not edited.

## 2026-10-13: Policy thresholds fixed from validation data (protocol Amendment 3)
The moderation policy rule and its thresholds were specified in Amendment 3 and tagged policy-frozen before the policy was evaluated on the test set, because choosing them after seeing the policy's test results would be tuning on the evaluation set. Amendment 3 was written after the test predictions existed, after the test results for average precision, F1, calibration and the review-budget analysis had been seen, and after a results report existed; its disclosure records this. Rule: per label, p < LOW allows, LOW <= p < HIGH sends to human review, p >= HIGH takes action, and a comment gets its most severe outcome. HIGH is the lowest validation probability with precision >= 0.90 and at least 30 comments at or above it (ACTION is disabled for a label if none qualifies); LOW is the highest with recall >= 0.90, clamped to HIGH. The targets (0.90 recall, 0.90 precision) are assumptions about what a moderation team might want, not findings. A sensitivity grid of recall and precision in {0.80, 0.90, 0.95} is computed alongside as a descriptive check. Systems: logistic regression, DistilBERT and JEV (v2); the supervised stack is excluded. The thresholds are in artifacts/policy/policy_thresholds.csv (validation inputs only) and were added to the freeze manifest as one new entry; the 13 existing entries are unchanged. At the primary targets, ACTION is disabled for severe_toxic and threat in all three systems, for identity_hate in logistic regression and JEV, and for insult in JEV, because no validation threshold reaches 0.90 precision with 30 flagged comments.

## Open items (closed)
All of the items below were open on 2026-10-12 and have since been completed or superseded; the later entries above record the results. They are kept as history.
- Closed: Install matplotlib in the environment and add it to requirements.txt (figures could not be run here).
- Closed: Fit the stack and write the frozen thresholds table on validation: python -m policyjev.evaluation.stack, then python -m policyjev.evaluation.frozen_thresholds.
- Closed: Copy the DistilBERT model folder from Colab to artifacts/models/distilbert, then write the manifest (python -m policyjev.freeze_manifest write), commit, and create the protocol-frozen git tag on that commit.
- Closed: Produce baseline test predictions after the freeze: logistic regression locally, DistilBERT on Colab or after copying the model folder (python -m policyjev.baselines.predict_test --system logreg|distilbert).
- Closed: Score the test set with question set v2 (runner --set test_full) and export the scores (python -m policyjev.jev.scores --set test_full).
- Closed: Run the comparison and review-budget analyses (with and without --with-stack) on test with --split test. Disagreement and error review stay on validation.
- Closed: Run the second JEV pass (python -m policyjev.jev.second_pass --questions configs/questions_v2.yaml) and the second-pass report.
- Closed: Decide how LOW and HIGH are chosen for the policy module (a dated amendment from validation data) before any policy number is computed on test.
