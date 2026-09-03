# E2A Raw Predictive Branch Metrics

Split: validation. Protocol: E1-B continuous windows. No temperature scaling.
Laplace uses last-layer kron/Curvlinops approximations over cls_head.fc_cls.

## Mean +- SD Across Folds

| Branch | Acc | Macro F1 | State Macro F1 | Transition Macro F1 | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deterministic | 0.9466 +- 0.0109 | 0.9116 +- 0.0277 | 0.9537 +- 0.0230 | 0.8905 +- 0.0313 | 0.1430 +- 0.0382 | 0.0796 +- 0.0150 | 0.0187 +- 0.0098 |
| laplace | 0.9468 +- 0.0102 | 0.9118 +- 0.0261 | 0.9540 +- 0.0227 | 0.8907 +- 0.0291 | 0.1418 +- 0.0375 | 0.0792 +- 0.0149 | 0.0171 +- 0.0090 |
| mc_dropout | 0.9467 +- 0.0106 | 0.9115 +- 0.0270 | 0.9541 +- 0.0225 | 0.8903 +- 0.0306 | 0.1406 +- 0.0347 | 0.0789 +- 0.0144 | 0.0151 +- 0.0075 |

## Fold Metrics

| Branch | Fold | N | Acc | Macro F1 | State Macro F1 | Transition Macro F1 | NLL | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deterministic | A | 5766 | 0.9376 | 0.8845 | 0.9274 | 0.8630 | 0.1845 | 0.0940 | 0.0254 |
| deterministic | B | 3870 | 0.9434 | 0.9106 | 0.9636 | 0.8840 | 0.1353 | 0.0808 | 0.0231 |
| deterministic | C | 4505 | 0.9587 | 0.9397 | 0.9700 | 0.9246 | 0.1092 | 0.0640 | 0.0074 |
| laplace | A | 5766 | 0.9381 | 0.8858 | 0.9281 | 0.8647 | 0.1827 | 0.0935 | 0.0229 |
| laplace | B | 3870 | 0.9442 | 0.9116 | 0.9642 | 0.8853 | 0.1338 | 0.0803 | 0.0217 |
| laplace | C | 4505 | 0.9580 | 0.9380 | 0.9699 | 0.9221 | 0.1088 | 0.0638 | 0.0067 |
| mc_dropout | A | 5766 | 0.9381 | 0.8851 | 0.9284 | 0.8634 | 0.1783 | 0.0928 | 0.0192 |
| mc_dropout | B | 3870 | 0.9434 | 0.9105 | 0.9637 | 0.8838 | 0.1336 | 0.0799 | 0.0196 |
| mc_dropout | C | 4505 | 0.9585 | 0.9391 | 0.9701 | 0.9236 | 0.1099 | 0.0639 | 0.0065 |
