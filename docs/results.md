# Results

This document reports what happened when the frozen protocol was applied to the official labeled Jigsaw test set (63,978 comments; 9.76% contain at least one toxic label). The tables in `docs/results_tables.md` are generated from the committed files in `artifacts/metrics/`; except where a section says otherwise (for example the call counts in section 8), the numbers below are taken from those tables and files. No model, calibrator, question set or threshold from the frozen protocol was changed after the test run. Amendment 3 (the moderation policy) was written after the test run, using validation data only, and is reported in `docs/results_policy.md`. JEV was queried a second time on a 5,000-comment test subset (section 7). Where a result is negative, it is reported as found.

Terms used throughout:

- **Average precision (AP)** summarises how well a system ranks true positives above negatives for one label, from 0 to 1. Higher is better. Mean AP is the average over the six labels.
- **Brier score** is the mean squared error between predicted probability and the true 0/1 label. Lower is better. It measures whether probabilities can be trusted as probabilities, not only whether the ranking is good.
- **Paired bootstrap interval** is a 95% interval for the difference between two systems, obtained by resampling test comments 1,000 times and scoring both systems on the same resample. An interval that excludes zero means the difference is unlikely to be sampling noise on this test set. These intervals reflect test-set sampling only; they do not capture variation from retraining the baselines.

## Summary

1. **Ranking quality: JEV did not match either trained baseline.** Mean AP was 0.527 for JEV, 0.583 for logistic regression and 0.677 for DistilBERT. On toxic, obscene and insult, the paired 95% intervals for JEV minus each baseline exclude zero and lie below it.
2. **On the two scarcest labels, the test set cannot separate JEV from the baselines.** For severe_toxic and threat, the paired intervals against both baselines include zero, and so does the interval against logistic regression on identity_hate. No significance test was run; the paired bootstrap intervals are the evidence.
3. **Under a tight human-review budget, JEV finds more rare-label comments.** At 1% and 2% of traffic reviewed, JEV recovers more rare-label positives than either baseline. Against DistilBERT this was the pattern seen on validation, and it held on test (status in `review_budget_test.csv`: hypothesis generated on validation); against logistic regression JEV was level on validation, so the advantage over it is a new, descriptive observation from test. At a 5% budget (pre-specified contrasts) JEV is ahead of DistilBERT (+0.071 [0.046, 0.101]) and its interval against logistic regression includes zero (−0.014 [−0.038, 0.016]). At a 10% budget (pre-specified) both baselines are ahead of JEV, and at 20% (descriptive) both baselines are also ahead.
4. **Probabilities are usable after calibration, but not better than the baselines'.** Raw JEV probabilities are poorly calibrated (mean Brier 0.066). After calibration fitted on validation data they are close to the baselines (0.023 against 0.020 and 0.019).
5. **A supervised stack on JEV's six outputs did not help.** The +0.04 mean AP gain seen on validation did not transfer; on test the gain is +0.009 and the interval includes zero.
6. **JEV is stable between runs and moderately fast, and DistilBERT is slow on CPU.** The model returns slightly different probabilities on repeat calls, but this changes no conclusion.

## 1. Ranking quality

| System | Mean AP [95% interval] |
| --- | --- |
| Logistic regression | 0.583 [0.566, 0.601] |
| DistilBERT | 0.677 [0.662, 0.693] |
| JEV | 0.527 [0.509, 0.544] |
| JEV stack (supervised) | 0.536 [0.520, 0.555] |

Paired mean-AP differences: JEV minus logistic regression −0.056 [−0.075, −0.038]; JEV minus DistilBERT −0.150 [−0.167, −0.134].

Validation mean AP was 0.644 (logistic regression), 0.724 (DistilBERT) and 0.566 (JEV). All three systems scored lower on test than on validation, including on the toxic label, where prevalence is nearly identical in the two sets. The mean drop was smallest for JEV (−0.039), then DistilBERT (−0.046), then logistic regression (−0.061), so JEV's gap to the baselines is slightly narrower on test than it was on validation. I do not have an explanation for the drop itself. Test comments were not inspected, in line with the protocol.

## 2. Per-label ranking

| Label | Positives | Logistic regression | DistilBERT | JEV |
| --- | --- | --- | --- | --- |
| toxic | 6,090 | 0.759 | 0.803 | 0.624 |
| severe_toxic | 367 | 0.308 | 0.376 | 0.357 |
| obscene | 3,691 | 0.779 | 0.803 | 0.619 |
| threat | 211 | 0.452 | 0.586 | 0.533 |
| insult | 3,427 | 0.705 | 0.794 | 0.548 |
| identity_hate | 712 | 0.494 | 0.703 | 0.481 |

Paired differences for JEV (95% interval):

| Label | JEV minus logistic regression | JEV minus DistilBERT |
| --- | --- | --- |
| toxic | −0.136 [−0.147, −0.125] | −0.179 [−0.190, −0.168] |
| severe_toxic | +0.048 [−0.001, 0.094] | −0.020 [−0.070, 0.023] |
| obscene | −0.160 [−0.172, −0.146] | −0.184 [−0.195, −0.171] |
| threat | +0.081 [−0.004, 0.153] | −0.053 [−0.123, 0.016] |
| insult | −0.157 [−0.173, −0.141] | −0.245 [−0.259, −0.232] |
| identity_hate | −0.012 [−0.060, 0.038] | −0.222 [−0.262, −0.180] |

JEV is clearly behind both baselines on the three most frequent labels (toxic, obscene, insult). On the rare labels the picture differs by label: on severe_toxic and threat the intervals against both baselines include zero; on identity_hate the interval against logistic regression includes zero (−0.012 [−0.060, 0.038]), but against DistilBERT JEV is clearly behind (−0.222 [−0.262, −0.180]). Because severe_toxic and threat have only 367 and 211 positives, their intervals are wide; an interval that includes zero means the test set cannot separate the systems, not that they are equal.

## 3. Decisions at a threshold

Thresholds were fitted on validation data and applied unchanged to test. The table gives F1 (the balance of precision and recall at the chosen threshold). F1 values are point estimates without intervals.

| System | Macro F1 | Micro F1 |
| --- | --- | --- |
| Logistic regression | 0.549 | 0.637 |
| DistilBERT | 0.617 | 0.680 |
| JEV | 0.529 | 0.555 |
| JEV stack (supervised) | 0.542 | 0.593 |

The macro gap between JEV and logistic regression is small (0.529 against 0.549); the micro gap is larger because micro F1 is dominated by the frequent labels. On threat, JEV has the highest F1 of the four systems (0.584 against 0.457 for logistic regression and 0.550 for DistilBERT).

With a fixed 0.5 threshold instead, JEV's macro F1 falls to 0.405 and its severe_toxic F1 is 0.000, because its calibrated probabilities for that label never reach 0.5 (the calibrator's largest output for severe_toxic is 0.417), even though raw JEV probabilities for that label are at or above 0.5 for 3,791 of the 63,978 test comments. This is why validation-fitted thresholds are part of the protocol.

## 4. Calibration

Mean Brier score:

| System | Raw | Calibrated |
| --- | --- | --- |
| Logistic regression | 0.019 | 0.020 |
| DistilBERT | 0.022 | 0.019 |
| JEV | 0.066 | 0.023 |
| JEV stack | 0.022 | 0.022 |

Calibration helped JEV and DistilBERT and very slightly hurt logistic regression on test, which was already well calibrated. On severe_toxic and threat JEV's calibrated Brier score is equal to or marginally below the baselines' at the three-decimal resolution of the table (severe_toxic 0.004 against 0.005). On identity_hate it equals logistic regression (0.007) and is marginally above DistilBERT (0.006). On toxic, obscene and insult it is higher (toxic 0.054, obscene 0.036, insult 0.032).

## 5. Human-review budget

The policy question is how many harmful comments a fixed review capacity catches. Each comment is scored by its highest calibrated probability across labels, the top share of traffic is sent to review, and recall is measured on the comments that are truly positive. `recall_any` is the share of comments with at least one positive label that land in the reviewed set; `recall_rare` is the same share restricted to comments positive for at least one of the rare labels (threat, severe_toxic, identity_hate), as defined in the experiment protocol.

Recall for rare-label positives:

| Budget | Logistic regression | DistilBERT | JEV |
| --- | --- | --- | --- |
| 1% | 0.195 | 0.125 | 0.307 |
| 2% | 0.386 | 0.243 | 0.462 |
| 5% | 0.688 | 0.603 | 0.674 |
| 10% | 0.855 | 0.863 | 0.817 |
| 20% | 0.957 | 0.982 | 0.934 |

Contrasts that were pre-specified before the test run (JEV minus baseline, 95% interval):

| Contrast | Metric | Budget | Difference |
| --- | --- | --- | --- |
| vs logistic regression | recall_any | 5% | −0.047 [−0.055, −0.038] |
| vs logistic regression | recall_any | 10% | −0.062 [−0.072, −0.052] |
| vs logistic regression | recall_rare | 5% | −0.014 [−0.038, 0.016] |
| vs logistic regression | recall_rare | 10% | −0.038 [−0.061, −0.011] |
| vs DistilBERT | recall_any | 5% | −0.052 [−0.060, −0.044] |
| vs DistilBERT | recall_any | 10% | −0.104 [−0.114, −0.094] |
| vs DistilBERT | recall_rare | 5% | +0.071 [0.046, 0.101] |
| vs DistilBERT | recall_rare | 10% | −0.046 [−0.069, −0.022] |

The 1% and 2% budgets were not among the pre-specified contrasts. The hypothesis generated on validation was that JEV has higher rare-label recall than DistilBERT at those budgets, and it held on test: the differences are +0.182 [0.155, 0.207] at 1% and +0.219 [0.196, 0.250] at 2%. Against logistic regression JEV is ahead by +0.112 [0.087, 0.136] at 1% and +0.076 [0.047, 0.103] at 2% (descriptive only; on validation the two were level).

Three points need care when reading this:

- **All comments:** at 1% and 2% budgets, `recall_any` is capped by capacity. Reviewing 1% of 63,978 comments (about 640) can catch at most about 10% of the 6,243 positives, and 2% at most about 20%. All systems sit near that ceiling (0.096–0.101 at 1%), so those rows do not separate them.
- **Rare labels at larger budgets:** the advantage disappears by 10%, where both baselines are ahead.
- **DistilBERT's weak rare-label recall at tight budgets** (0.125 at 1%, below logistic regression) is surprising for the system with the best AP. A likely cause is that ordinary toxic comments, which it scores very confidently, fill its review set when the score is the maximum across labels. I have not tested this; it is a hypothesis.

## 6. Supervised stack on JEV outputs

The secondary system fits a per-label logistic regression on JEV's six probabilities, using validation labels. It lifted validation mean AP from 0.566 to 0.605 (out-of-fold). On test the change is +0.009 [−0.004, 0.023]. It helps toxic, obscene, insult and identity_hate (+0.029 to +0.058) and hurts severe_toxic (−0.113 [−0.153, −0.072]). It also loses JEV's rare-label advantage at a 1% budget (0.206 against 0.307). Adding supervision on top of JEV's outputs is not a productive direction on this evidence.

## 7. Run-to-run variation

JEV was queried a second time on a fixed random sample of 5,000 test comments. The first run is the canonical one; the second only measures stability.

- Mean absolute change in probability: 0.004 (0.003 to 0.007 by label). About 30% of probabilities changed at all.
- Mean AP changed by +0.001 and mean F1 by 0.000. The largest per-label AP change was +0.011 (severe_toxic) and the largest F1 change −0.013 (identity_hate).

Per-label changes on this sample are noisy because the sample holds only 25 threat and 38 severe_toxic positives. AP values in this section are computed on the 5,000-row sample and are not comparable to the full-test figures above.

## 8. Latency and API calls

| System | Latency p50 | Latency p95 | Throughput |
| --- | --- | --- | --- |
| Logistic regression (CPU) | 1.8 ms | 2.3 ms | 5,310 comments/s (batch 32) |
| DistilBERT (CPU) | 40.8 ms | 147.6 ms | 7.1 comments/s (batch 32) |
| JEV (API) | 363 ms | 686 ms | not measured |

All local timings were taken on a single Intel CPU (as recorded in `latency_cost.csv`), 500 comments per system. The JEV latency row was computed from the call log before the test run: it pools the 72,246 successful requests recorded up to then (the pilot and the validation runs of all three question sets, including calls made during question-set selection) and excludes the test run and the second pass. JEV latency depends on the network, the provider's load and concurrent requests at the time. One request scores one comment, and the test run scored each of the 63,978 comments (the cache is keyed by normalized text, so identical texts are not queried twice). Across all runs (pilot, validation, test and second pass) JEV scoring made roughly 141,000 successful calls; the exact count comes from a call log that is not published. The batch runs were made under a pre-set hard spending cap, and no price or dollar figure is published.

## 9. Where JEV fails and where it helps (validation, qualitative)

This review used validation comments only. It describes patterns and is not a measurement.

- **Misses:** JEV most often misses short, profanity-heavy comments that Jigsaw labels as insult, obscene or toxic. JEV answers the questions literally; when a comment is crude but not an insult in the question's sense, it scores low while the annotators labeled it positive. This is a mismatch between the label definition and the question wording.
- **Catches the baselines miss:** hostile comments with few vocabulary cues, where the baselines have little lexical signal.
- **Apparent false positives:** many comments counted as JEV-only false positives look like real positives with doubtful gold labels, including at least one clearly threatening comment labeled as no threat. Some of the measured gap to the baselines is therefore label noise, but this review cannot say how much.
- **Shared errors:** where all three systems disagree with the gold label, the group mixes genuine label errors with comments that are hard for every system.

The question wording was not edited after this review.

## 10. Limitations

- One dataset, one JEV version (`jev-1.13.0`) and one question set, chosen among three candidates on validation.
- JEV is zero-shot in its judgments, but its calibration and thresholds, and the choice of question set, use labels from 23,936 validation comments. It is label-light, not label-free.
- Intervals cover test-set sampling only. The baselines were trained once with one seed.
- Many intervals are reported without adjustment for multiple comparisons. Treat any single interval near zero cautiously.
- F1 values have no intervals.
- Jigsaw labels are noisy, and part of JEV's measured gap may reflect label definitions that differ from the questions it is asked.
- The drop in every system's score from validation to test is unexplained.
- Latency was measured on one machine, on CPU only.
- Comments were sent to a third-party API during the JEV runs.
