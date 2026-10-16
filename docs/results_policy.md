# Moderation policy results

This document reports the moderation policy defined in Amendment 3 of `docs/experiment_protocol.md`. The rule, the targets and the test metrics were fixed and tagged (`policy-frozen`) before any policy number was computed on the test set, and the policy was evaluated on test once. Every number below comes from tables i–m of `docs/results_tables.md`. The ranking results are in `docs/results.md`.

## The policy

For each label, a calibrated probability p is compared with two thresholds:

- p below LOW: **ALLOW**
- p from LOW up to HIGH: **HUMAN REVIEW**
- p at or above HIGH: **ACTION** (automatic enforcement)

A comment receives the most severe outcome across its six labels. Thresholds are chosen per system and per label from validation data only, with the same targets for every system:

- **LOW** is the highest threshold at which at least 90% of that label's validation positives still score at or above it. At most about 10% of true positives are auto-allowed.
- **HIGH** is the lowest threshold at which precision on validation is at least 90% and at least 30 validation comments score at or above it. If no threshold qualifies, automatic action is switched off for that label and everything above LOW goes to human review.

The 0.90 targets are assumptions, not findings. To show how much they matter, a pre-specified grid of recall and precision targets (0.80, 0.90, 0.95) is reported as a secondary, descriptive analysis.

## What could be automated, on validation

| Label | Logistic regression | DistilBERT | JEV |
| --- | --- | --- | --- |
| toxic | ACTION | ACTION | ACTION |
| severe_toxic | review only | review only | review only |
| obscene | ACTION | ACTION | ACTION |
| threat | review only | review only | review only |
| insult | ACTION | ACTION | review only |
| identity_hate | review only | ACTION (37 validation comments at HIGH) | review only |

No system reached 90% precision with enough support on severe_toxic or threat, so for those labels the policy cannot act automatically and relies on human review. DistilBERT qualifies on identity_hate, but only just above the 30-comment floor. JEV can act automatically on two labels only, toxic and obscene.

## Results on the test set at the primary targets

For M1, M3 and M4 lower is better. For M2 higher is better. All 63,978 test comments were scored for every system, with 0 failed rows.

| System | ALLOW | HUMAN REVIEW | ACTION | M1 review share | M2 ACTION precision | M3 miss rate | M4 rare miss rate | ACTION coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | 48,798 | 8,703 | 6,477 | 0.136 | 0.658 | 0.054 | 0.018 | 0.691 |
| DistilBERT | 51,450 | 4,358 | 8,170 | 0.068 | 0.623 | 0.049 | 0.013 | 0.826 |
| JEV | 49,927 | 11,060 | 2,991 | 0.173 | 0.766 | 0.115 | 0.050 | 0.376 |

Definitions: M1 is the share of all comments sent to human review. M2 is the share of ACTION comments in which at least one label that reached its HIGH threshold is truly positive. M3 is the share of comments with at least one positive label that were allowed. M4 is the same for comments positive for threat, severe_toxic or identity_hate. ACTION coverage is the share of comments with any positive label that received ACTION.

Pre-specified paired contrasts for JEV (difference [95% interval], 1,000 resamples, none left out):

| Metric | JEV minus logistic regression | JEV minus DistilBERT |
| --- | --- | --- |
| M1 review share | +0.037 [0.034, 0.040] | +0.105 [0.102, 0.108] |
| M2 ACTION precision | +0.108 [0.093, 0.123] | +0.143 [0.128, 0.157] |
| M3 miss rate | +0.060 [0.051, 0.069] | +0.066 [0.058, 0.074] |
| M4 rare miss rate | +0.032 [0.019, 0.047] | +0.037 [0.025, 0.050] |

All eight intervals exclude zero. The reading:

- **JEV needs more human review and misses more harmful comments than either baseline.** Its miss rate is about twice logistic regression's and more than twice DistilBERT's, and the same holds for rare-label comments.
- **JEV's automatic actions are more precise, but there are far fewer of them.** It acts on 4.7% of traffic and covers 38% of harmful comments, against 69% for logistic regression and 83% for DistilBERT. Each system has automatic action enabled on a different set of labels, so M2 and coverage are not a like-for-like comparison; M1, M3 and M4 are the fairer ones.
- **DistilBERT is the strongest policy engine on this test set.** It has the lowest review load, the fewest misses and the highest coverage. Its ACTION precision is the lowest of the three.
- **The rare-label advantage at tight review budgets does not carry over.** At 1% and 2% budgets JEV recovered more rare-label positives (`docs/results.md`, section 5). A policy that must meet a recall target needs much larger review sets than that, and at those sizes JEV is behind (M4 0.050 against 0.018 and 0.013).

## The validation targets did not hold on test

The thresholds were fitted to give about 90% recall at LOW and about 90% precision at HIGH on validation. The per-label results on test show where that failed.

Recall at LOW (about 0.90 on validation by construction):

| Label | Logistic regression | DistilBERT | JEV |
| --- | --- | --- | --- |
| toxic | 0.934 | 0.952 | 0.788 |
| severe_toxic | 0.940 | 0.943 | 0.907 |
| obscene | 0.895 | 0.868 | 0.800 |
| threat | 0.929 | 0.972 | 0.891 |
| insult | 0.903 | 0.887 | 0.768 |
| identity_hate | 0.948 | 0.951 | 0.850 |

ACTION precision on test, where automatic action is enabled (validation precision at HIGH was 0.900 to 0.919):

| Label | Logistic regression | DistilBERT | JEV |
| --- | --- | --- | --- |
| toxic | 0.656 | 0.623 | 0.813 |
| obscene | 0.818 | 0.713 | 0.780 |
| insult | 0.932 | 0.952 | not enabled |
| identity_hate | not enabled | 0.958 | not enabled |

- **Recall held for the baselines and not for JEV.** Logistic regression stays at or near 0.90 on every label (lowest 0.895, obscene). DistilBERT is under target on obscene (0.868) and insult (0.887) only. JEV falls 10 to 13 points under target on toxic (0.788), obscene (0.800) and insult (0.768), and to 0.850 on identity_hate; on severe_toxic (0.907) and threat (0.891) it is close. This is why JEV's miss rates are higher.
- **The precision shortfall sits on the two most frequent labels.** On toxic and obscene, ACTION precision is 0.62 to 0.82 against a validation figure of about 0.90, for all three systems. On insult (0.932 and 0.952) and on identity_hate (0.958, DistilBERT only, about 130 actioned comments) precision is at or above target. Toxic dominates the volume of automatic actions, so the system-level M2 mostly reflects it. JEV's toxic precision (0.813) is the highest of the three but still about 0.09 below its validation figure.
- **Possible cause, not tested.** The pattern is the same shift between the held-out part of `train.csv` and the official test set that appears in the AP scores (`docs/results.md`, section 1), and it hits the three systems differently by label. I do not have an explanation, and test comments were not inspected.

In practice, thresholds fitted on one data source did not carry their guarantees to a different source. Before enabling automatic action in a real deployment, thresholds would need to be re-estimated on recent data from that deployment.

## Sensitivity to the targets (secondary, point estimates)

LOW depends on the recall target and HIGH on the precision target, with one interaction: when the computed LOW would exceed HIGH, LOW is set to HIGH. Away from that case, miss rates change with the recall target, and ACTION precision and coverage change with the precision target.

Miss rates by recall target (M3 / M4), shown at the 0.90 and 0.95 precision targets, where they are identical:

| System | 0.80 | 0.90 | 0.95 |
| --- | --- | --- | --- |
| Logistic regression | 0.135 / 0.043 | 0.054 / 0.018 | 0.023 / 0.006 |
| DistilBERT | 0.122 / 0.043 | 0.049 / 0.013 | 0.019 / 0.006 |
| JEV | 0.215 / 0.095 | 0.115 / 0.050 | 0.052 / 0.019 |

The one exception is DistilBERT at recall 0.80 with precision 0.80, where capping LOW at HIGH lowers the miss rates to 0.077 / 0.030.

ACTION precision (M2) and coverage by precision target:

| System | 0.80 | 0.90 | 0.95 |
| --- | --- | --- | --- |
| Logistic regression | 0.566 / 0.800 | 0.658 / 0.691 | 0.768 / 0.562 |
| DistilBERT | 0.516 / 0.923 | 0.623 / 0.826 | 0.713 / 0.724 |
| JEV | 0.588 / 0.589 | 0.766 / 0.376 | 0.931 / 0.119 |

- Observed ACTION precision is below the target in every row, for every system. The gap shrinks for JEV at the strictest target: at 0.95 it reaches 0.931, higher than both baselines, but it then acts on only 1.2% of traffic and covers 12% of harmful comments.
- JEV's miss rate sits above the per-label miss level at every recall target (0.215, 0.115 and 0.052 against 0.20, 0.10 and 0.05), while the baselines sit well below.
- At recall 0.80 and precision 0.80, DistilBERT's review band almost disappears (review share 0.0002). The cap to HIGH applies to toxic and obscene; a thin band remains on the other labels, and severe_toxic and threat stay review-only.

## Limitations

- The 0.90 targets and the 30-comment floor are assumptions. The grid shows the results for other targets but does not say which is right.
- No cost model: review and action are not priced, and human reviewers are treated as perfect.
- The test-set precision and recall shortfalls are unexplained, and no thresholds were refit after seeing them, in line with the protocol.
- Per-label precision for rarely actioned labels, such as identity_hate for DistilBERT, rests on about 130 comments and is noisy.
- Intervals cover test-set sampling only, and the grid and per-label tables have no intervals.
- Comments are treated as independent, and the systems are compared on one dataset.
