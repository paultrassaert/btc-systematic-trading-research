# Retrospective reference results

All returns below use the same 24-hour holding protocol within each year, monthly expanding refits and 7 bps per trading leg. Each year resets initial capital to 1. Passive buy-and-hold uses the same sampled interval and pays entry/exit costs once. The reported original −68.2% strategy used different dates, labels and execution; it is not directly comparable.

| Year | Logistic return | Logistic max drawdown | Logistic exposure coverage | Boosting return | Buy and hold |
|---|---:|---:|---:|---:|---:|
| 2021 | 19.6% | -31.7% | 33.8% | -26.8% | 59.8% |
| 2022 | -6.7% | -30.2% | 12.4% | -52.9% | -64.7% |
| 2023 | 12.2% | -6.2% | 3.6% | -9.5% | 154.8% |

The logistic model traded 123, 45 and 13 of 364 scheduled intervals in 2021, 2022 and 2023, respectively. Its 2023 Sharpe therefore describes a sparse retrospective selection, not a large sample of independent profitable trades. Classification AUCs were approximately 0.518, 0.459 and 0.526. No stable predictive edge is established.

## Cost sensitivity: logistic model

| Year | 0 bps per leg | 7 bps per leg | 15 bps per leg |
|---|---:|---:|---:|
| 2021 | 42.1% | 19.6% | -1.8% |
| 2022 | -0.6% | -6.7% | -13.2% |
| 2023 | 14.2% | 12.2% | 9.9% |

These are illustrative total one-way cost assumptions. Funding, borrowing, liquidation, market impact and execution venue constraints are omitted. Returns are hypothetical research-account returns. All years were analysed retrospectively; the original project had access to the full series. No model was retuned after these reported reruns.

![2021 equity](retrospective_2021/equity.png)

![2022 equity](retrospective_2022/equity.png)

![2023 equity](retrospective_2023/equity.png)

Machine-readable metrics, predictions, hourly account values, fitting audit records and data hashes accompany every run.
