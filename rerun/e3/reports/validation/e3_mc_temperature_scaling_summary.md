# E3 MC Pool-Then-Calibrate Temperature Scaling

Main result split: validation subject. Temperature is fitted on the calibration subject only.
Deltas are calibrated minus raw, so negative values indicate improvement.

| Fold | T* | Raw NLL | Cal. NLL | Delta NLL | Raw Brier | Cal. Brier | Delta Brier | Raw ECE | Cal. ECE | Delta ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 1.0073 | 0.1783 | 0.1780 | -0.0004 | 0.0928 | 0.0927 | -0.0001 | 0.0192 | 0.0187 | -0.0005 |
| B | 0.9615 | 0.1336 | 0.1345 | 0.0009 | 0.0799 | 0.0805 | 0.0006 | 0.0196 | 0.0221 | 0.0025 |
| C | 1.1165 | 0.1099 | 0.1135 | 0.0036 | 0.0639 | 0.0641 | 0.0001 | 0.0065 | 0.0142 | 0.0077 |
| Mean +- SD | - | 0.1406 +- 0.0347 | 0.1420 +- 0.0329 | 0.0014 +- 0.0020 | 0.0789 +- 0.0144 | 0.0791 +- 0.0144 | 0.0002 +- 0.0003 | 0.0151 +- 0.0075 | 0.0183 +- 0.0039 | 0.0032 +- 0.0042 |

Saved but not emphasized: raw/calibrated center accuracy and macro-F1. They are expected to match because pool-temperature scaling preserves class ordering.
