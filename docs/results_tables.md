# Results tables

Built by `python -m policyjev.evaluation.results_tables` from the committed aggregate files in `artifacts/metrics/`. Nothing is recomputed from predictions. Values are formatted to three decimals; intervals are 95% paired-bootstrap percentile intervals.

## a. Mean average precision per system

Source: `artifacts/metrics/test_comparison_with_stack.csv`.

| System | Mean AP [95% interval] |
| --- | --- |
| Logistic regression | 0.583 [0.566, 0.601] |
| DistilBERT | 0.677 [0.662, 0.693] |
| JEV | 0.527 [0.509, 0.544] |
| JEV stack (supervised) | 0.536 [0.520, 0.555] |

## b. Average precision per label

Source: `artifacts/metrics/test_comparison_with_stack.csv`.

| Label | Positives | Logistic regression | DistilBERT | JEV | JEV stack (supervised) |
| --- | --- | --- | --- | --- | --- |
| toxic | 6090 | 0.759 [0.750, 0.769] | 0.803 [0.793, 0.812] | 0.624 [0.611, 0.636] | 0.666 [0.653, 0.678] |
| severe_toxic | 367 | 0.308 [0.267, 0.362] | 0.376 [0.334, 0.432] | 0.357 [0.316, 0.404] | 0.244 [0.215, 0.279] |
| obscene | 3691 | 0.779 [0.766, 0.791] | 0.803 [0.790, 0.814] | 0.619 [0.603, 0.635] | 0.663 [0.648, 0.680] |
| threat | 211 | 0.452 [0.383, 0.525] | 0.586 [0.515, 0.651] | 0.533 [0.462, 0.602] | 0.526 [0.451, 0.595] |
| insult | 3427 | 0.705 [0.690, 0.720] | 0.794 [0.780, 0.805] | 0.548 [0.531, 0.565] | 0.606 [0.589, 0.624] |
| identity_hate | 712 | 0.494 [0.451, 0.535] | 0.703 [0.670, 0.735] | 0.481 [0.445, 0.518] | 0.511 [0.475, 0.550] |

## c. Paired AP differences (overall and per label)

Source: `artifacts/metrics/test_comparison_with_stack.csv`.

| Label | Positives | JEV minus logistic regression | JEV minus DistilBERT | JEV stack minus JEV |
| --- | --- | --- | --- | --- |
| toxic | 6090 | -0.136 [-0.147, -0.125] | -0.179 [-0.190, -0.168] | 0.042 [0.035, 0.049] |
| severe_toxic | 367 | 0.048 [-0.001, 0.094] | -0.020 [-0.070, 0.023] | -0.113 [-0.153, -0.072] |
| obscene | 3691 | -0.160 [-0.172, -0.146] | -0.184 [-0.195, -0.171] | 0.044 [0.034, 0.055] |
| threat | 211 | 0.081 [-0.004, 0.153] | -0.053 [-0.123, 0.016] | -0.007 [-0.067, 0.053] |
| insult | 3427 | -0.157 [-0.173, -0.141] | -0.245 [-0.259, -0.232] | 0.058 [0.047, 0.070] |
| identity_hate | 712 | -0.012 [-0.060, 0.038] | -0.222 [-0.262, -0.180] | 0.029 [0.008, 0.052] |
| mean (overall) |  | -0.056 [-0.075, -0.038] | -0.150 [-0.167, -0.134] | 0.009 [-0.004, 0.023] |

## d. F1 at the frozen validation-fit thresholds and at fixed 0.5

Source: `artifacts/metrics/test_threshold_metrics_with_stack.csv`.

Threshold basis `fixed_0.5` (sample status: not fitted).

| Label | Positives | Logistic regression | DistilBERT | JEV | JEV stack (supervised) | Sample status |
| --- | --- | --- | --- | --- | --- | --- |
| toxic | 6090 | 0.671 | 0.684 | 0.542 | 0.603 | not fitted |
| severe_toxic | 367 | 0.351 | 0.378 | 0.000 | 0.020 | not fitted |
| obscene | 3691 | 0.699 | 0.704 | 0.589 | 0.607 | not fitted |
| threat | 211 | 0.458 | 0.543 | 0.483 | 0.289 | not fitted |
| insult | 3427 | 0.649 | 0.719 | 0.483 | 0.559 | not fitted |
| identity_hate | 712 | 0.473 | 0.617 | 0.333 | 0.279 | not fitted |
| micro | 14498 | 0.655 | 0.686 | 0.527 | 0.572 | not fitted |
| macro |  | 0.550 | 0.607 | 0.405 | 0.393 | not fitted |

Threshold basis `validation_oof_fit` (sample status: out-of-sample (thresholds were fit on validation)).

| Label | Positives | Logistic regression | DistilBERT | JEV | JEV stack (supervised) | Sample status |
| --- | --- | --- | --- | --- | --- | --- |
| toxic | 6090 | 0.663 | 0.683 | 0.569 | 0.622 | out-of-sample (thresholds were fit on validation) |
| severe_toxic | 367 | 0.326 | 0.431 | 0.372 | 0.365 | out-of-sample (thresholds were fit on validation) |
| obscene | 3691 | 0.688 | 0.700 | 0.565 | 0.611 | out-of-sample (thresholds were fit on validation) |
| threat | 211 | 0.457 | 0.550 | 0.584 | 0.529 | out-of-sample (thresholds were fit on validation) |
| insult | 3427 | 0.636 | 0.716 | 0.568 | 0.572 | out-of-sample (thresholds were fit on validation) |
| identity_hate | 712 | 0.522 | 0.623 | 0.514 | 0.552 | out-of-sample (thresholds were fit on validation) |
| micro | 14498 | 0.637 | 0.680 | 0.555 | 0.593 | out-of-sample (thresholds were fit on validation) |
| macro |  | 0.549 | 0.617 | 0.529 | 0.542 | out-of-sample (thresholds were fit on validation) |

## e. Brier score, raw and calibrated

Source: `artifacts/metrics/test_comparison_with_stack.csv`.

| Label | Logistic regression raw | Logistic regression calibrated | DistilBERT raw | DistilBERT calibrated | JEV raw | JEV calibrated | JEV stack (supervised) raw | JEV stack (supervised) calibrated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toxic | 0.047 | 0.052 | 0.066 | 0.053 | 0.099 | 0.054 | 0.054 | 0.054 |
| severe_toxic | 0.005 | 0.005 | 0.005 | 0.005 | 0.043 | 0.004 | 0.005 | 0.005 |
| obscene | 0.025 | 0.027 | 0.030 | 0.027 | 0.059 | 0.036 | 0.033 | 0.033 |
| threat | 0.002 | 0.002 | 0.002 | 0.002 | 0.022 | 0.002 | 0.002 | 0.002 |
| insult | 0.027 | 0.028 | 0.023 | 0.022 | 0.145 | 0.032 | 0.031 | 0.031 |
| identity_hate | 0.008 | 0.007 | 0.006 | 0.006 | 0.026 | 0.007 | 0.007 | 0.007 |
| mean | 0.019 | 0.020 | 0.022 | 0.019 | 0.066 | 0.023 | 0.022 | 0.022 |

## f. Review budget

Source: `artifacts/metrics/review_budget_test_with_stack.csv`.

Recall at each review budget (value [95% interval]).

| Metric | Budget % | Logistic regression | DistilBERT | JEV | JEV stack (supervised) |
| --- | --- | --- | --- | --- | --- |
| recall_any | 1 | 0.098 [0.095, 0.101] | 0.101 [0.098, 0.104] | 0.096 [0.093, 0.099] | 0.096 [0.093, 0.099] |
| recall_any | 2 | 0.196 [0.191, 0.201] | 0.197 [0.192, 0.202] | 0.185 [0.179, 0.190] | 0.185 [0.180, 0.190] |
| recall_any | 5 | 0.440 [0.431, 0.450] | 0.446 [0.436, 0.455] | 0.394 [0.385, 0.403] | 0.379 [0.371, 0.388] |
| recall_any | 10 | 0.689 [0.678, 0.699] | 0.731 [0.720, 0.742] | 0.627 [0.617, 0.637] | 0.625 [0.614, 0.636] |
| recall_any | 20 | 0.915 [0.907, 0.922] | 0.954 [0.948, 0.959] | 0.862 [0.854, 0.871] | 0.865 [0.856, 0.874] |
| recall_rare | 1 | 0.195 [0.175, 0.216] | 0.125 [0.108, 0.143] | 0.307 [0.283, 0.332] | 0.206 [0.182, 0.226] |
| recall_rare | 2 | 0.386 [0.361, 0.417] | 0.243 [0.217, 0.265] | 0.462 [0.436, 0.490] | 0.416 [0.386, 0.441] |
| recall_rare | 5 | 0.688 [0.660, 0.714] | 0.603 [0.574, 0.631] | 0.674 [0.649, 0.702] | 0.622 [0.596, 0.651] |
| recall_rare | 10 | 0.855 [0.831, 0.874] | 0.863 [0.843, 0.884] | 0.817 [0.795, 0.839] | 0.842 [0.822, 0.863] |
| recall_rare | 20 | 0.957 [0.944, 0.969] | 0.982 [0.974, 0.990] | 0.934 [0.920, 0.949] | 0.938 [0.924, 0.951] |

Pre-specified paired contrasts (difference [95% interval]).

| Contrast | Metric | Budget % | Difference [95% interval] |
| --- | --- | --- | --- |
| JEV minus logistic regression | recall_any | 5 | -0.047 [-0.055, -0.038] |
| JEV minus logistic regression | recall_any | 10 | -0.062 [-0.072, -0.052] |
| JEV minus logistic regression | recall_rare | 5 | -0.014 [-0.038, 0.016] |
| JEV minus logistic regression | recall_rare | 10 | -0.038 [-0.061, -0.011] |
| JEV minus DistilBERT | recall_any | 5 | -0.052 [-0.060, -0.044] |
| JEV minus DistilBERT | recall_any | 10 | -0.104 [-0.114, -0.094] |
| JEV minus DistilBERT | recall_rare | 5 | 0.071 [0.046, 0.101] |
| JEV minus DistilBERT | recall_rare | 10 | -0.046 [-0.069, -0.022] |

## g. Second-pass stability (first pass against second pass)

Source: `artifacts/metrics/second_pass_report.csv`.

| Label | Positives | Mean abs. probability change | Share of rows changed | AP first pass | AP second pass | AP change | F1 first pass | F1 second pass | F1 change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toxic | 445 | 0.006 | 0.387 | 0.637 | 0.632 | -0.006 | 0.568 | 0.570 | 0.002 |
| severe_toxic | 38 | 0.003 | 0.256 | 0.468 | 0.478 | 0.011 | 0.408 | 0.419 | 0.011 |
| obscene | 275 | 0.004 | 0.269 | 0.640 | 0.641 | 0.001 | 0.590 | 0.588 | -0.002 |
| threat | 25 | 0.003 | 0.195 | 0.795 | 0.802 | 0.008 | 0.815 | 0.815 | 0.000 |
| insult | 257 | 0.007 | 0.419 | 0.567 | 0.569 | 0.003 | 0.588 | 0.589 | 0.001 |
| identity_hate | 52 | 0.003 | 0.244 | 0.514 | 0.507 | -0.008 | 0.585 | 0.571 | -0.013 |
| mean |  | 0.004 | 0.295 | 0.604 | 0.605 | 0.001 | 0.592 | 0.592 | 0.000 |

## h. Latency

Source: `artifacts/metrics/latency_cost.csv`.

| System | Device | Device name | Latency p50 (ms) | Latency p95 (ms) | Throughput (comments/s, batch 32) | API requests | Comments timed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | cpu | Intel64 Family 6 Model 158 Stepping 10, GenuineIntel | 1.799 | 2.268 | 5309.953 |  | 500 |
| DistilBERT | cpu | Intel64 Family 6 Model 158 Stepping 10, GenuineIntel | 40.811 | 147.628 | 7.122 |  | 500 |
| JEV | api |  | 362.959 | 685.581 |  | 72246 |  |

## i. Policy thresholds at the primary targets (fit on validation data)

Source: `artifacts/policy/policy_thresholds.csv`.

| System | Label | LOW | HIGH | ACTION enabled | Validation recall at LOW | Validation precision at HIGH | Validation comments at or above HIGH |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | toxic | 0.095 | 0.657 | yes | 0.903 | 0.900 | 1628 |
| Logistic regression | severe_toxic | 0.015 | inf | no | 0.907 |  | 0 |
| Logistic regression | obscene | 0.130 | 0.697 | yes | 0.905 | 0.904 | 881 |
| Logistic regression | threat | 0.019 | inf | no | 0.903 |  | 0 |
| Logistic regression | insult | 0.098 | 0.851 | yes | 0.904 | 0.905 | 242 |
| Logistic regression | identity_hate | 0.010 | inf | no | 0.909 |  | 0 |
| DistilBERT | toxic | 0.260 | 0.654 | yes | 0.900 | 0.900 | 1894 |
| DistilBERT | severe_toxic | 0.065 | inf | no | 0.903 |  | 0 |
| DistilBERT | obscene | 0.311 | 0.670 | yes | 0.901 | 0.900 | 1050 |
| DistilBERT | threat | 0.006 | inf | no | 0.903 |  | 0 |
| DistilBERT | insult | 0.217 | 0.847 | yes | 0.900 | 0.901 | 456 |
| DistilBERT | identity_hate | 0.027 | 0.825 | yes | 0.904 | 0.919 | 37 |
| JEV | toxic | 0.116 | 0.744 | yes | 0.902 | 0.904 | 1067 |
| JEV | severe_toxic | 0.048 | inf | no | 0.903 |  | 0 |
| JEV | obscene | 0.144 | 0.858 | yes | 0.901 | 0.909 | 596 |
| JEV | threat | 0.057 | inf | no | 0.903 |  | 0 |
| JEV | insult | 0.087 | inf | no | 0.906 |  | 0 |
| JEV | identity_hate | 0.080 | inf | no | 0.909 |  | 0 |

## j. Policy on the test set at the primary targets

Source: `artifacts/metrics/policy_test_summary.csv`.

| System | Comments | Failed rows | ALLOW | HUMAN REVIEW | ACTION | ALLOW share | M1 review share | M2 ACTION precision | M3 miss rate | M4 rare miss rate | ACTION share | ACTION coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | 63978 | 0 | 48798 | 8703 | 6477 | 0.763 | 0.136 | 0.658 | 0.054 | 0.018 | 0.101 | 0.691 |
| DistilBERT | 63978 | 0 | 51450 | 4358 | 8170 | 0.804 | 0.068 | 0.623 | 0.049 | 0.013 | 0.128 | 0.826 |
| JEV | 63978 | 0 | 49927 | 11060 | 2991 | 0.780 | 0.173 | 0.766 | 0.115 | 0.050 | 0.047 | 0.376 |

## k. Policy contrasts (paired bootstrap)

Source: `artifacts/metrics/policy_test_contrasts.csv`.

| Contrast | Metric | Difference [95% interval] | Resamples used | Resamples left out |
| --- | --- | --- | --- | --- |
| JEV minus logistic regression | M1 review share | 0.037 [0.034, 0.040] | 1000 | 0 |
| JEV minus logistic regression | M2 ACTION precision | 0.108 [0.093, 0.123] | 1000 | 0 |
| JEV minus logistic regression | M3 miss rate | 0.060 [0.051, 0.069] | 1000 | 0 |
| JEV minus logistic regression | M4 rare miss rate | 0.032 [0.019, 0.047] | 1000 | 0 |
| JEV minus DistilBERT | M1 review share | 0.105 [0.102, 0.108] | 1000 | 0 |
| JEV minus DistilBERT | M2 ACTION precision | 0.143 [0.128, 0.157] | 1000 | 0 |
| JEV minus DistilBERT | M3 miss rate | 0.066 [0.058, 0.074] | 1000 | 0 |
| JEV minus DistilBERT | M4 rare miss rate | 0.037 [0.025, 0.050] | 1000 | 0 |

## l. Policy sensitivity grid (point estimates)

Source: `artifacts/metrics/policy_test_grid.csv`.

| System | Recall target | Precision target | Primary | M1 review share | M2 ACTION precision | M3 miss rate | M4 rare miss rate | ACTION share | ACTION coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.800 | 0.800 |  | 0.032 | 0.566 | 0.135 | 0.043 | 0.136 | 0.800 |
| Logistic regression | 0.800 | 0.900 |  | 0.067 | 0.658 | 0.135 | 0.043 | 0.101 | 0.691 |
| Logistic regression | 0.800 | 0.950 |  | 0.097 | 0.768 | 0.135 | 0.043 | 0.071 | 0.562 |
| Logistic regression | 0.900 | 0.800 |  | 0.101 | 0.566 | 0.054 | 0.018 | 0.136 | 0.800 |
| Logistic regression | 0.900 | 0.900 | primary | 0.136 | 0.658 | 0.054 | 0.018 | 0.101 | 0.691 |
| Logistic regression | 0.900 | 0.950 |  | 0.166 | 0.768 | 0.054 | 0.018 | 0.071 | 0.562 |
| Logistic regression | 0.950 | 0.800 |  | 0.177 | 0.566 | 0.023 | 0.006 | 0.136 | 0.800 |
| Logistic regression | 0.950 | 0.900 |  | 0.211 | 0.658 | 0.023 | 0.006 | 0.101 | 0.691 |
| Logistic regression | 0.950 | 0.950 |  | 0.242 | 0.768 | 0.023 | 0.006 | 0.071 | 0.562 |
| DistilBERT | 0.800 | 0.800 |  | 0.000 | 0.516 | 0.077 | 0.030 | 0.173 | 0.923 |
| DistilBERT | 0.800 | 0.900 |  | 0.020 | 0.623 | 0.122 | 0.043 | 0.128 | 0.826 |
| DistilBERT | 0.800 | 0.950 |  | 0.049 | 0.713 | 0.122 | 0.043 | 0.098 | 0.724 |
| DistilBERT | 0.900 | 0.800 |  | 0.023 | 0.516 | 0.049 | 0.013 | 0.173 | 0.923 |
| DistilBERT | 0.900 | 0.900 | primary | 0.068 | 0.623 | 0.049 | 0.013 | 0.128 | 0.826 |
| DistilBERT | 0.900 | 0.950 |  | 0.098 | 0.713 | 0.049 | 0.013 | 0.098 | 0.724 |
| DistilBERT | 0.950 | 0.800 |  | 0.076 | 0.516 | 0.019 | 0.006 | 0.173 | 0.923 |
| DistilBERT | 0.950 | 0.900 |  | 0.120 | 0.623 | 0.019 | 0.006 | 0.128 | 0.826 |
| DistilBERT | 0.950 | 0.950 |  | 0.150 | 0.713 | 0.019 | 0.006 | 0.098 | 0.724 |
| JEV | 0.800 | 0.800 |  | 0.069 | 0.588 | 0.215 | 0.095 | 0.091 | 0.589 |
| JEV | 0.800 | 0.900 |  | 0.113 | 0.766 | 0.215 | 0.095 | 0.047 | 0.376 |
| JEV | 0.800 | 0.950 |  | 0.147 | 0.931 | 0.215 | 0.095 | 0.012 | 0.119 |
| JEV | 0.900 | 0.800 |  | 0.129 | 0.588 | 0.115 | 0.050 | 0.091 | 0.589 |
| JEV | 0.900 | 0.900 | primary | 0.173 | 0.766 | 0.115 | 0.050 | 0.047 | 0.376 |
| JEV | 0.900 | 0.950 |  | 0.207 | 0.931 | 0.115 | 0.050 | 0.012 | 0.119 |
| JEV | 0.950 | 0.800 |  | 0.201 | 0.588 | 0.052 | 0.019 | 0.091 | 0.589 |
| JEV | 0.950 | 0.900 |  | 0.245 | 0.766 | 0.052 | 0.019 | 0.047 | 0.376 |
| JEV | 0.950 | 0.950 |  | 0.280 | 0.931 | 0.052 | 0.019 | 0.012 | 0.119 |

## m. Policy per label on the test set at the primary targets

Source: `artifacts/metrics/policy_test_per_label.csv` and `artifacts/policy/policy_thresholds.csv`.

| System | Label | Positives | LOW | HIGH | ACTION enabled | Validation recall at LOW | Test recall at LOW | Validation precision at HIGH | Test ACTION precision | Review share | ACTION share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | toxic | 6090 | 0.095 | 0.657 | yes | 0.903 | 0.934 | 0.900 | 0.656 | 0.118 | 0.101 |
| Logistic regression | severe_toxic | 367 | 0.015 | inf | no | 0.907 | 0.940 | - | - | 0.066 | 0.000 |
| Logistic regression | obscene | 3691 | 0.130 | 0.697 | yes | 0.905 | 0.895 | 0.904 | 0.818 | 0.083 | 0.040 |
| Logistic regression | threat | 211 | 0.019 | inf | no | 0.903 | 0.929 | - | - | 0.024 | 0.000 |
| Logistic regression | insult | 3427 | 0.098 | 0.851 | yes | 0.904 | 0.903 | 0.905 | 0.932 | 0.130 | 0.009 |
| Logistic regression | identity_hate | 712 | 0.010 | inf | no | 0.909 | 0.948 | - | - | 0.107 | 0.000 |
| DistilBERT | toxic | 6090 | 0.260 | 0.654 | yes | 0.900 | 0.952 | 0.900 | 0.623 | 0.066 | 0.127 |
| DistilBERT | severe_toxic | 367 | 0.065 | inf | no | 0.903 | 0.943 | - | - | 0.034 | 0.000 |
| DistilBERT | obscene | 3691 | 0.311 | 0.670 | yes | 0.901 | 0.868 | 0.900 | 0.713 | 0.033 | 0.057 |
| DistilBERT | threat | 211 | 0.006 | inf | no | 0.903 | 0.972 | - | - | 0.046 | 0.000 |
| DistilBERT | insult | 3427 | 0.217 | 0.847 | yes | 0.900 | 0.887 | 0.901 | 0.952 | 0.078 | 0.016 |
| DistilBERT | identity_hate | 712 | 0.027 | 0.825 | yes | 0.904 | 0.951 | 0.919 | 0.958 | 0.045 | 0.002 |
| JEV | toxic | 6090 | 0.116 | 0.744 | yes | 0.902 | 0.788 | 0.904 | 0.813 | 0.147 | 0.036 |
| JEV | severe_toxic | 367 | 0.048 | inf | no | 0.903 | 0.907 | - | - | 0.049 | 0.000 |
| JEV | obscene | 3691 | 0.144 | 0.858 | yes | 0.901 | 0.800 | 0.909 | 0.780 | 0.088 | 0.027 |
| JEV | threat | 211 | 0.057 | inf | no | 0.903 | 0.891 | - | - | 0.012 | 0.000 |
| JEV | insult | 3427 | 0.087 | inf | no | 0.906 | 0.768 | - | - | 0.101 | 0.000 |
| JEV | identity_hate | 712 | 0.080 | inf | no | 0.909 | 0.850 | - | - | 0.040 | 0.000 |
