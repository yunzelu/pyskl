# E3 MC Pool-Then-Calibrate Temperature Scaling

Main result split: outer test subject. Temperature is fitted on the calibration subject only.
Deltas are calibrated minus raw, so negative values indicate improvement.

| Fold | T* | Raw NLL | Cal. NLL | Delta NLL | Raw Brier | Cal. Brier | Delta Brier | Raw ECE | Cal. ECE | Delta ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 1.0073 | 0.1338 | 0.1338 | 0.0000 | 0.0766 | 0.0766 | -0.0000 | 0.0096 | 0.0090 | -0.0007 |
| B | 0.9615 | 0.1168 | 0.1167 | -0.0001 | 0.0653 | 0.0654 | 0.0001 | 0.0042 | 0.0049 | 0.0007 |
| C | 1.1165 | 0.1013 | 0.1070 | 0.0057 | 0.0579 | 0.0586 | 0.0007 | 0.0100 | 0.0173 | 0.0073 |
| Mean +- SD | - | 0.1173 +- 0.0163 | 0.1192 +- 0.0136 | 0.0019 +- 0.0033 | 0.0666 +- 0.0094 | 0.0669 +- 0.0091 | 0.0003 +- 0.0004 | 0.0080 +- 0.0032 | 0.0104 +- 0.0063 | 0.0024 +- 0.0042 |

Saved but not emphasized: raw/calibrated center accuracy and macro-F1. They are expected to match because pool-temperature scaling preserves class ordering.
