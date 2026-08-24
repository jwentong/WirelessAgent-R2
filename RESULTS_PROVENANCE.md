# WirelessBench Results Provenance

This file records the numerical values currently reported in the revised manuscript. It
is documentation of the archived manuscript results; it is not a claim that a fresh
API run has been performed locally.

## Revised-manuscript held-out test results

| Benchmark | Score |
|---|---:|
| WCHW | 78.37% |
| WCNS | 90.95% |
| WCMSA | 97.07% |

## Validation-set workflow evolution

These values describe the optimization trajectory and must not be substituted for the
held-out test results above.

| Benchmark | Seed | Intermediate | Best/final validation phase |
|---|---:|---:|---:|
| WCHW | 62.44% (Round 1) | 80.86% (Round 2) | 81.78% (Round 14) |
| WCNS | 61.3% | 90.5% | 92.18% |
| WCMSA | 65.76% | 93.59% | 96.89% |

## Search-cost records

| Metric | WCHW | WCNS | WCMSA |
|---|---:|---:|---:|
| Search rounds | 19 | 11 | 11 |
| Wall-clock time (min) | 63 | 13 | 14 |
| Total search cost (USD) | 4.95 | 0.99 | 1.05 |
| Per-problem cost (USD) | 0.00083 | 0.00056 | 0.00068 |

## Dataset split records

| Benchmark | Validation | Test |
|---|---:|---:|
| WCHW | 348 | 1,044 |
| WCNS | 250 | 750 |
| WCMSA | 250 | 750 |

The historical comparison table in the manuscript contains a separate WCHW “Before”
and “After” record. That table is not an aggregate of the three held-out benchmark
scores above and should not be used to overwrite them.

## Reproduction notes

- API credentials are intentionally excluded from this repository. Copy
  `config/config2.example.yaml` to `config/config2.yaml` and provide credentials
  locally through the ignored file.
- Fresh executions can differ because API model versions, prompts, data files, and
  search trajectories must be fixed to reproduce an archived result.
