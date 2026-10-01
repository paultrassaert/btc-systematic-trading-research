# Audit of the supplied trading notebook

Cell indices below are zero-based, counting markdown cells. Findings are based on saved source and outputs; the original Optuna/XGBoost computation was not rerun.

| Finding | Evidence | Interpretation / correction |
|---|---|---|
| Future-dependent HMM states | Cell 17 calls `model_hmm.predict(scaled_features.values)` on the complete series. | Sequence decoding can use later observations. Past-only parameter fitting does not make a full-series decoded state causal. Use forward filtering and refit all learned components on the eligible history; HMM excluded from the reference baseline. |
| HMM leakage into CV | HMM fitted once on all 2018–2020 before folds. | A held-out fold has contributed to the state model. Refit within chronological folds if HMM is reintroduced. |
| CV differs from live forecasting | Cell 29 trains on observations both before and after a held-out block. | Purged K-fold can be useful under stated assumptions, but it is not a past-only deployment simulation. Its percentage buffer is not derived from label information intervals. The present roughly 519-hour buffer is larger than the 72-hour scan, so this audit does **not** claim that this particular gap is too short. |
| Label availability | Cell 22 chooses the best slope over future windows; cell 23 sets 12–72h. | The winning horizon alone does not determine when the label is known: all candidate windows must be observed. Any future patch must use the maximum inspected time, not only the winning `t_end`. The current 72h expanding-window gap is consistent with the displayed 72h maximum; it should be derived automatically. |
| Target/execution mismatch | 12–72h trend labels; cell 67 switches directional exposure hourly. | A label can be correct while the next hourly return is negative. The reference instead predicts and trades the same 24h open-to-open interval. This is a new strategy, not a like-for-like model improvement. |
| Reproducibility failure | Cell 41 reads `primary_oof_probs` and `meta_y_train`; neither is defined anywhere in the supplied source. | Saved outputs are not evidence of clean-kernel execution. The reference excludes meta-labelling. A future implementation needs chronological primary predictions, mature labels and separately evaluated meta-model fitting. |
| Threshold not actually optimised | Cell 41 prints an optimisation message but fixes the threshold at 0.45; computed `meta_oof_probs` are not used to choose it. | Do not describe it as an optimised threshold. |
| Ambiguous cash code | Cell 41 writes zero for rejected signals, although zero already denotes the bearish class. | Keep labels `{0,1}` distinct from positions `{-1,0,+1}`. This variable is **not** the one used in the final expanding-window backtest, so it does not directly explain −68.2%. |
| Misleading selected-sample comparison | Filtered accuracy 56.60% on 3,341 points, with 1,899 bullish labels. | Always predicting bullish on that same subset gives 56.84%. Full-sample accuracy and selected-sample accuracy are not directly comparable. Report coverage and a matched baseline. |
| Missing trading costs | Cell 67 only multiplies lagged positions by close returns. | −68.2% is a gross result. Adding costs would generally worsen it; omitted costs did not cause the reported loss. A one-row lag alone does not validate execution or upstream feature causality. |
| No stable directional edge shown | One-shot XGBoost AUC 0.5159; expanding-window AUC 0.4721. | Low signal is a plausible explanation for poor results. The code does not establish which feature or modelling choice caused the loss. Reversing the model after seeing this test would be another selection decision. |
| Inconsistent explanation | Cell 24 says average horizon ~81h; saved cell 23 output is 50.10h and its maximum is 72h. | Narrative is stale. 81h is impossible under the displayed configuration. |
| Hurst implementation | Square root of the standard deviation of lag differences, applied to returns, with no correction of the fitted exponent. | This is not the documented conventional Hurst estimate. Excluded from the reference baseline. |
| Parkinson aggregation | Mean of per-bar square roots instead of square root of the mean squared log range. | Corrected aggregation in the reference features. |
| Full-training scaling before folds | Cell 33 fits the scaler before CV. | Preprocessing should be fitted within each fold. This is unlikely to materially explain these tree results because positive affine rescaling generally preserves tree orderings. |
| Sortino and drawdown details | Original downside denominator uses the standard deviation only among negative returns; drawdown omits initial wealth from the peak. | Reference uses root mean squared negative returns over all periods and includes initial wealth in the running peak. Neither proves a strategy has predictive value. |

## Original saved metrics

| Model | Accuracy | Macro F1 | ROC-AUC | Population |
|---|---:|---:|---:|---|
| Random Forest | 52.08% | 0.4182 | 0.4944 | 8,664 points |
| XGBoost, one-shot | 52.03% | 0.4567 | 0.5159 | 8,664 points |
| XGBoost, expanding | 48.19% | 0.4725 | 0.4721 | 8,664 points |
| XGBoost + meta-filter | 56.60% | 0.4655 | 0.5336 | 3,341 selected points |
| Always bullish | 53.06% | — | — | 8,664 points |
| Always bullish, selected subset | 56.84% | — | — | Same 3,341 selected points |

Original expanding strategy: return −68.2%, maximum drawdown −85.6%, hourly-annualised Sharpe −0.757. Its buy-and-hold comparison: +48.4%, drawdown −54.8%, Sharpe 0.899. These are saved notebook figures, not newly verified execution results.

Accuracy below 50% does not mechanically imply negative P&L: move sizes, holding times, exposure and costs matter. Similarly, leakage compromises evaluation but does not itself prove why a specific backtest lost money.

## Next research gate

Before further tuning, define an untouched chronological evaluation period and an economic hypothesis. Compare each added component against simple baselines under identical labels, execution, periods and costs. Reintroduce the HMM or trend scanning only as separately measured changes. Account for serial dependence and multiple trials before making statistical significance claims. A credible negative result is preferable to selecting a profitable-looking backtest after repeated test inspection.
