# Evaluation on simulated therapy attempts

1200 attempts from 40 speakers (none, mild, moderate, severe severity), ratings 1 to 5. 95% intervals from 1000 bootstrap samples over speakers.

| scoring | Spearman | weighted kappa |
|---|---|---|
| plain phoneme error rate | 0.707 (0.653 to 0.747) | 0.661 (0.609 to 0.703) |
| phonetically weighted | 0.711 (0.655 to 0.750) | 0.717 (0.665 to 0.758) |

Difference in Spearman (weighted minus plain): 0.004, 95% interval -0.005 to 0.015.

## By severity

| severity | attempts | mean rating | mean score | exact agreement | within one point |
|---|---|---|---|---|---|
| none | 300 | 4.73 | 96.0 | 0.77 | 0.98 |
| mild | 300 | 4.48 | 88.4 | 0.64 | 0.94 |
| moderate | 300 | 3.91 | 74.8 | 0.48 | 0.90 |
| severe | 300 | 3.44 | 65.4 | 0.46 | 0.91 |

## Low recognition confidence

44 attempts (3.7%) fall below the confidence threshold of 0.5. Exact agreement is 0.60 on the remaining attempts and 0.20 on the flagged ones.

