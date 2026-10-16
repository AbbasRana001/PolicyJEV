# EDA findings (train partition, 135,635 comments)

All figures come from artifacts/metrics/eda_*.csv. The test set was not used.

- Imbalance: 89.8% of comments carry no label. Label rates in train: toxic 9.6%, obscene 5.3%, insult 4.9%, severe_toxic 1.0%, identity_hate 0.9%, threat 0.3% (406 comments).
- Label structure: every severe_toxic comment (1,368) is also toxic. 93.6% of threat comments (380 of 406) and 92.9% of identity_hate comments (1,111 of 1,196) are also toxic, so 26 threat and 85 identity_hate comments are not labeled toxic.
- Length: the median is 36 words (205 characters), the 90th percentile 152 words, the 99th percentile 567 words, and the longest comments are exactly 5,000 characters. Toxic comments are shorter at the median (22 words) but have a longer tail than unlabeled ones at the 99th percentile (661 vs 561 words). The DistilBERT maximum sequence length is chosen in Module 2 from tokenizer counts, not from these word counts.
- Label noise: 34 groups of comments with identical text (after lowercasing and whitespace normalization) carry different labels. Some model errors in later modules will reflect label noise rather than model mistakes.
- Duplicates: 375 train rows share their text with another train row, and 56 validation rows (0.23% of validation) have text that also appears in train. No deduplication is applied.
