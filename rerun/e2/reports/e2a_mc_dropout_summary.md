# E2A Raw Predictive Branch Metrics

Split: validation. Protocol: E1-B continuous windows. No temperature
scaling is applied. MC dropout uses 10 stochastic passes unless the
command line overrides `--num-passes`.

## Mean +- SD Across Folds

| Branch | Acc | Macro F1 | State Macro F1 | Transition Macro F1 | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deterministic | 0.9466 +- 0.0109 | 0.9116 +- 0.0277 | 0.9537 +- 0.0230 | 0.8905 +- 0.0313 | 0.1430 +- 0.0382 | 0.0796 +- 0.0150 | 0.0187 +- 0.0098 |
| mc_dropout | 0.9467 +- 0.0106 | 0.9115 +- 0.0270 | 0.9541 +- 0.0225 | 0.8903 +- 0.0306 | 0.1406 +- 0.0347 | 0.0789 +- 0.0144 | 0.0151 +- 0.0075 |

## Fold Metrics

| Branch | Fold | N | Acc | Macro F1 | State Macro F1 | Transition Macro F1 | NLL | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deterministic | A | 5766 | 0.9376 | 0.8845 | 0.9274 | 0.8630 | 0.1845 | 0.0940 | 0.0254 |
| deterministic | B | 3870 | 0.9434 | 0.9106 | 0.9636 | 0.8840 | 0.1353 | 0.0808 | 0.0231 |
| deterministic | C | 4505 | 0.9587 | 0.9397 | 0.9700 | 0.9246 | 0.1092 | 0.0640 | 0.0074 |
| mc_dropout | A | 5766 | 0.9381 | 0.8851 | 0.9284 | 0.8634 | 0.1783 | 0.0928 | 0.0192 |
| mc_dropout | B | 3870 | 0.9434 | 0.9105 | 0.9637 | 0.8838 | 0.1336 | 0.0799 | 0.0196 |
| mc_dropout | C | 4505 | 0.9585 | 0.9391 | 0.9701 | 0.9236 | 0.1099 | 0.0639 | 0.0065 |
