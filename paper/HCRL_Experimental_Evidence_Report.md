# HCRL / CAN-ATCNet Experimental Evidence Report

## Experimental protocol
- Seeds: 42, 123, 2026.
- Classes: Normal, DoS, Fuzzy, Gear, RPM.
- Causal window length: 32.
- Chronological development blocks with a 32-row separation.
- Training preprocessing fitted from training data.
- Development variants: TCN-CE, TCN-Focal, TCN-Predictive, CAN-ATCNet.
- Original held-out block: rows [400064, 600064), 4000 sampled windows.
- Later confirmation block: rows [600096, 800096), 4000 sampled windows.
- Confirmation combiner was fitted on validation outputs only, then frozen before confirmation evaluation.
- Confirmation is a later block from the same recordings; it is NOT independent external validation.

## Original held-out mean ± SD

| Model | Accuracy | Balanced accuracy | Macro-F1 | Macro-AP | Macro-AUC |
|---|---:|---:|---:|---:|---:|
| CAN-ATCNet | 0.98250 ± 0.00455 | 0.98520 ± 0.00223 | 0.96143 ± 0.00846 | 0.98341 ± 0.00114 | 0.99852 ± 0.00026 |
| Full classifier only | 0.98350 ± 0.00304 | 0.98348 ± 0.00446 | 0.96316 ± 0.00566 | 0.98235 ± 0.00220 | 0.99863 ± 0.00030 |
| TCN-CE | 0.98867 ± 0.00138 | 0.97701 ± 0.00805 | 0.97216 ± 0.00422 | 0.99088 ± 0.00405 | 0.99901 ± 0.00069 |
| TCN-Focal | 0.98492 ± 0.00176 | 0.98413 ± 0.00362 | 0.96584 ± 0.00389 | 0.98442 ± 0.00172 | 0.99873 ± 0.00043 |
| TCN-Predictive | 0.98308 ± 0.00270 | 0.97927 ± 0.00950 | 0.96201 ± 0.00477 | 0.98225 ± 0.00221 | 0.99864 ± 0.00029 |

Validation selected alpha=0.0 for the original global fusion under mean validation macro-AP. Therefore the validation-selected fusion reduced to classifier-only behavior.

## Mechanism diagnostics
### Calibration
TCN-CE ECE across seeds: approximately 0.0072–0.0099.
CAN-ATCNet fixed fusion ECE: approximately 0.0413–0.0696.
This shows materially worse probability calibration under the fixed fusion.

### Residual-derived consistency evidence
The score used here is the residual-derived consistency-head attack score.
- Seed 42: binary AP 0.94349, AUC 0.99008.
- Seed 123: binary AP 0.90935, AUC 0.98324.
- Seed 2026: binary AP 0.81812, AUC 0.96309.

The score is useful but seed-sensitive.

Diagnostic runs also found high correlation between classifier attack probability and consistency score (~0.82–0.91), explaining why global fusion often behaves as amplification rather than independent evidence.

### Paired held-out error comparison
TCN-CE versus fixed CAN-ATCNet:
- Seed 42: McNemar exact p≈0.0576.
- Seed 123: p≈0.000821, favoring TCN-CE.
- Seed 2026: p≈8.50e-06, favoring TCN-CE.

This was the evidence for abandoning the claim that fixed/global fusion outperformed the baseline.

## Selective-combiner confirmation results

| Seed | Model | Accuracy | Balanced accuracy | Macro-F1 | Macro-AP | Macro-AUC |
|---:|---|---:|---:|---:|---:|---:|
| 42 | TCN-CE | 0.99325 | 0.986134 | 0.983644 | 0.996016 | 0.999595 |
| 42 | Selective combiner | 0.99425 | 0.987086 | 0.985851 | 0.991194 | 0.997676 |
| 123 | TCN-CE | 0.99225 | 0.983287 | 0.981252 | 0.998152 | 0.999810 |
| 123 | Selective combiner | 0.99350 | 0.985942 | 0.985316 | 0.991218 | 0.996373 |
| 2026 | TCN-CE | 0.99000 | 0.970247 | 0.975635 | 0.996525 | 0.998792 |
| 2026 | Selective combiner | 0.99300 | 0.982687 | 0.984138 | 0.986316 | 0.999149 |

### Confirmation mean ± SD

| Model | Accuracy | Balanced accuracy | Macro-F1 | Macro-AP | Macro-AUC |
|---|---:|---:|---:|---:|---:|
| Selective combiner | 0.993583 ± 0.000629 | 0.985238 ± 0.002282 | 0.985102 ± 0.000877 | 0.989576 ± 0.002823 | 0.997733 ± 0.001389 |
| TCN-CE | 0.991833 ± 0.001665 | 0.979889 ± 0.008471 | 0.980177 ± 0.004111 | 0.996898 ± 0.001115 | 0.999399 ± 0.000537 |

### Confirmation paired errors
| Seed | Both correct | Both wrong | TCN-only correct | Combiner-only correct | McNemar exact p |
|---:|---:|---:|---:|---:|---:|
| 42 | 3969 | 19 | 4 | 8 | 0.387695 |
| 123 | 3968 | 25 | 1 | 6 | 0.125000 |
| 2026 | 3959 | 27 | 1 | 13 | 0.001831 |

The direction of hard-decision improvement is consistent across all three seeds. Seed 2026 shows a statistically significant paired error reduction. The three-seed sample is too small to make broad population-level statistical claims.

## Scientific interpretation
The selective mechanism improves accuracy, balanced accuracy and macro-F1, supporting the hypothesis that predictive-consistency information is complementary for final class decisions. However, macro-AP and macro-AUC decline, so the current combiner should not be presented as universally superior or better calibrated.

## Recommended paper emphasis
- Primary mechanism finding: predictive consistency can correct a subset of classifier errors.
- Strong metrics: balanced accuracy and macro-F1, because they expose minority-class decision behavior.
- Transparency: include the AP/AUC tradeoff.
- Do not tune further on the confirmation block.
- Treat the earlier held-out diagnostics as hypothesis-generating and the later confirmation block as within-recording confirmation, not external validation.

## Provenance
The confirmation metrics above were recovered from GitHub Actions run 37531969596. The workflow completed successfully and deleted its RunPod after result printing.
