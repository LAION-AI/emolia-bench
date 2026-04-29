# Annotation Analysis

## Overview
- Raw annotations: 23991
- Deduplicated annotations: 23958
- Duplicate user-item rows removed: 33
- Complete 3-rater items used for agreement: 7984
- Incomplete items excluded from agreement: 4

## Agreement
- Exact 3-way ordinal agreement: 0.173
- Exact 3-way binary agreement: 0.327
- Fleiss' kappa (ordinal 3-level): 0.058
- Fleiss' kappa (binary present/absent): 0.086
- Mean pairwise Cohen's kappa (ordinal): 0.081
- Mean pairwise Cohen's kappa (binary): 0.113

## Task-Type Summary
- affirmative: 3998 items, majority-present 0.789, unanimous-binary 0.446, unanimous-ordinal 0.174
- contrastive_1: 1000 items, majority-present 0.349, unanimous-binary 0.212, unanimous-ordinal 0.177
- contrastive_2: 1000 items, majority-present 0.372, unanimous-binary 0.204, unanimous-ordinal 0.172
- ultimate: 994 items, majority-present 0.376, unanimous-binary 0.206, unanimous-ordinal 0.169
- penultimate: 992 items, majority-present 0.370, unanimous-binary 0.210, unanimous-ordinal 0.172

## Hardest Queried Emotions
- Sexual_Lust: majority-present 0.289 across 280 items
- Fear: majority-present 0.328 across 311 items
- Embarrassment: majority-present 0.448 across 317 items
- Intoxication_Altered_States_of_Consciousness: majority-present 0.455 across 606 items
- Triumph: majority-present 0.471 across 378 items

## Easiest Queried Emotions
- Thankfulness_Gratitude: majority-present 0.981 across 103 items
- Doubt: majority-present 0.960 across 100 items
- Interest: majority-present 0.901 across 101 items
- Hope_Enthusiasm_Optimism: majority-present 0.901 across 111 items
- Astonishment_Surprise: majority-present 0.890 across 100 items

## Benchmark Guidance
- Use `benchmark_labels.csv` as the item-level benchmark table.
- `majority_present` is the clean binary target for benchmarking retrieval/classification.
- `benchmark_bucket` separates stricter subsets such as `unanimous_present` and `unanimous_absent`.
- `all_agree_binary` and `all_agree_ordinal` are useful for confidence-tiered evaluation.

## Incomplete Items
- Excluded items: 4
- EN_B00013_S03009_W000013_speaker_reference.mp3 | Infatuation | affirmative | ratings observed: 1
- EN_wlt0Ae-TUG8_W000016_speaker_reference.mp3 | Longing | penultimate | ratings observed: 1
- EN_wlt0Ae-TUG8_W000016_speaker_reference.mp3 | Longing | affirmative | ratings observed: 2
- EN_B00013_S03009_W000013_speaker_reference.mp3 | Infatuation | ultimate | ratings observed: 2
