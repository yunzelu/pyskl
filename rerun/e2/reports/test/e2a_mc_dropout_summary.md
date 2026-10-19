# E2A Raw Predictive Branch Metrics

Split: test. Protocol: E1-B continuous windows. No temperature
scaling is applied. MC dropout uses 10 stochastic passes unless the
command line overrides `--num-passes`.

## Mean +- SD Across Folds

| Branch | Acc | Macro F1 | State Macro F1 | Transition Macro F1 | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deterministic | 0.9548 +- 0.0074 | 0.9262 +- 0.0147 | 0.9654 +- 0.0092 | 0.9066 +- 0.0184 | 0.1171 +- 0.0184 | 0.0669 +- 0.0102 | 0.0092 +- 0.0038 |
| mc_dropout | 0.9548 +- 0.0074 | 0.9263 +- 0.0144 | 0.9654 +- 0.0090 | 0.9068 +- 0.0181 | 0.1172 +- 0.0164 | 0.0666 +- 0.0096 | 0.0071 +- 0.0021 |

## Fold Metrics

| Branch | Fold | N | Acc | Macro F1 | State Macro F1 | Transition Macro F1 | NLL | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deterministic | A | 4334 | 0.9469 | 0.9124 | 0.9548 | 0.8912 | 0.1357 | 0.0778 | 0.0136 |
| deterministic | B | 4934 | 0.9558 | 0.9245 | 0.9704 | 0.9015 | 0.1166 | 0.0655 | 0.0072 |
| deterministic | C | 4964 | 0.9617 | 0.9416 | 0.9710 | 0.9270 | 0.0989 | 0.0575 | 0.0069 |
| mc_dropout | A | 4334 | 0.9469 | 0.9128 | 0.9551 | 0.8917 | 0.1338 | 0.0767 | 0.0075 |
| mc_dropout | B | 4934 | 0.9558 | 0.9247 | 0.9704 | 0.9018 | 0.1166 | 0.0654 | 0.0048 |
| mc_dropout | C | 4964 | 0.9617 | 0.9415 | 0.9708 | 0.9269 | 0.1011 | 0.0576 | 0.0089 |
