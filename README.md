# Bitcoin systematic trading research

An hourly-BTC machine-learning study, followed by an audit and a reproducible causal reference implementation. The aim is to test whether technical features carry useful directional information and whether that information survives trading costs.

**Main finding:** the original notebook's apparent methodological sophistication did not establish a trading edge. The revised baselines also show weak and unstable predictive performance. Positive returns in one market period are not evidence of alpha.

## Original study and authorship

Original coursework: **Paul Trassaert, Raphael Lebel, Thomas Lesage and Alexandre d’Hérissart**, IMT Atlantique, 2026.  Course context: [Machine Learning and Finance](https://hm-ai.github.io/IMT_ML_and_Finance/).

The original work explored 46 features, a three-state Gaussian HMM, trend-scanning labels, XGBoost/Optuna, Random Forest meta-labelling and expanding-window prediction. Its archived 2021 result was **−68.2% before costs**, versus **+48.4%** for its own benchmark interval. This repository does not relabel that result as profitable.

The `archive/` notebook preserves student code with outputs and external setup/template material removed. It is explicitly superseded: undefined meta-labelling variables and other issues prevent a reliable fresh-kernel execution. The corrected reference uses `research.py` and `run.py`. It is a methodological rewrite, **not an exact reproduction or isolated ablation of the original XGBoost strategy**. The audit and implementation were prepared with AI assistance and should be understood and reviewed by the authors before interview use.

## Research walkthrough: the reasoning behind the figures

The sections below integrate the team's original analysis with the saved graphs, including unsuccessful attempts and design decisions. Wording has been edited and stale claims corrected against the code. [The complete original wording](docs/original-analysis.md) is preserved separately. Figure-based interpretations describe the original experiment; source-backed corrections and proposed experiments are identified explicitly.

### 1. Features: replacing raw price with market descriptors

The team constructed 46 features covering momentum, trend, volatility, volume, serial dependence and market regimes. Examples include RSI, MACD, EMA ratios, ATR, Bollinger bands, Parkinson/Garman–Klass volatility, autocorrelation at several lags and HMM state indicators. Calendar variables were also explored.

The motivation was to describe relative market conditions rather than ask a model to extrapolate a raw Bitcoin price level. That is a reasonable hypothesis, but transforming prices does not automatically make every feature stationary or predictive. The original “fractional difference” and Hurst implementations also need care; the audit explains why they are not accepted as standard implementations in the reference baseline.

### 2. HMM regimes and the rolling-scaling decision

![HMM regimes over time](assets/original/cell-17-figure-1.png)

The team's original diagnosis was that a global scaler fitted on an older period handled the 2021 market poorly: unprecedented levels and volatility made observations look extreme relative to the historical distribution, and the HMM appeared to remain in a bullish regime too often. To address this, the team introduced a **720-hour rolling Z-score**, expressing each observation relative to roughly the previous 30 days.

The graph documents the resulting regime segmentation. The intended benefit was adaptation to local market conditions rather than dependence on one old scaling reference. Calling this “scale saturation” was the author's description of an observed problem, not an established property of StandardScaler.

**Code correction:** fitting HMM parameters only on past data is insufficient here. Cell 17 calls `model_hmm.predict` on the **entire sequence**. Sequence decoding can let later observations affect earlier states. A deployable version needs past-only forward filtering, and HMM fitting must be repeated within each training fold. The rolling scaler does not resolve those issues by itself.

![Forward returns by HMM regime](assets/original/cell-19-figure-1.png)

The forward-return comparison asks whether the identified states correspond to different subsequent return distributions. This is a useful diagnostic, but the state construction above prevents treating apparent separation as a causal trading signal. States also received bear/neutral/bull names using historical returns; the names are interpretations, not guarantees of future direction.

![Feature correlations](assets/original/cell-20-figure-1.png)

The correlation matrix investigates redundant technical indicators: many are transformations of the same price/volume history. This motivated the later clustering analysis. Forty-six columns are not necessarily forty-six independent sources of information; redundant predictors can make a sophisticated pipeline look more informative than it is.

### 3. Trend-scanning labels: what the model actually learns

![Trend-scanning labels and selected horizons](assets/original/cell-23-figure-1.png)

For each timestamp, the code fits a line to log prices over several future windows and chooses the slope with the largest absolute t-statistic. Its sign becomes the bullish/bearish target. The intended purpose was to focus on sustained trends rather than noisy one-hour moves. The implementation used parallel processing and `numpy.polyfit` to make the scan feasible over the hourly history.

**The saved run uses 12–72 hours and reports an average selected horizon of 50.10 hours.** The original prose discussing a 120-hour maximum and an 81-hour average belonged to another configuration and does not describe these outputs. Selecting the largest t-statistic across windows also does not, by itself, establish a statistically significant predictive opportunity.

This target uses future data by design, which is normal for supervised learning. The protection is to train only after its whole information interval has elapsed. Since the scan examines every candidate up to 72 hours, the target cannot be considered known after only the shorter winning horizon.

The economic question is equally important: **predicting a trend over roughly two days is not the same task as predicting the return over the next hour.** The original backtest later trades the latter using predictions trained on the former.

### 4. Static models, early stopping and directional bias

The original modelling sequence was a Random Forest baseline followed by XGBoost, with Optuna replacing a broader grid-search approach. Macro-F1 and dynamic class weights were intended to avoid rewarding a model that simply chose the more frequent direction. Calendar features were removed after the author observed unexpectedly high importance for `day_of_month`, which raised a concern about accidental seasonal patterns.

A high importance for a calendar variable is a reason to investigate, not proof that it is spurious. Likewise, removing it does not establish that the remaining technical indicators contain genuine alpha. Feature-selection choices should be made on chronological development folds rather than repeatedly inspecting the same final year.

The one-shot Optuna run returned an average early-stopping choice of **25 iterations**. The author tried reducing the learning rate to allow more trees, but reported no meaningful improvement. This is consistent with a limited generalisation benefit in that setup; early stopping alone cannot identify whether noise, drift, label design or hyperparameters are responsible.

Cell 38 provides a concrete bias diagnostic: the one-shot model predicted **bullish 81.2% of the time**, while the actual bullish-label share was **53.1%**. The author's interpretation was a “long by default” model, with weak probabilities converted into active decisions by a 0.50 threshold. That finding applies to the **one-shot** model; it must not be assumed to describe every expanding-window model without checking its own predictions.

### 5. Meta-labelling: why the apparent uplift needs a matched baseline

The Random Forest meta-model was intended as a trade-acceptance filter. The saved output rejected **61.4% of signals**, leaving 3,341 of 8,664 observations. Accuracy on the retained observations rose from the full-sample figure of 52.03% to **56.60%**. The author already expressed caution about luck, selection effects and the smaller sample.

The key comparison is on the **same retained observations**: always predicting bullish would score **56.84%**, slightly above 56.60%. The uplift alone therefore does not demonstrate useful selection skill. Coverage, trade outcomes and a matched benchmark are needed.

The saved source also references undefined `primary_oof_probs` and `meta_y_train`; the notebook cannot reproduce this step from a clean kernel as supplied. The threshold is fixed at **0.45**, despite an “optimising” message, and zero is ambiguously used for cash and the bearish label.

**Most importantly, this filter is not used in the final −68.2% backtest.** Cell 67 uses `ew_preds`, not the filtered one-shot predictions. Improvements reported for this separate experiment cannot explain or repair that final equity curve automatically.

### 6. Why expanding-window retraining was introduced

The team's response to a static model's apparent lack of adaptation was to retrain XGBoost on an expanding history every **48 hours**. Rather than reuse the one-shot hyperparameters, a second Optuna search simulated walk-forward prediction near the end of the training period.

Computational cost constrained this search: it used the last **1,000 hours of 2020**, about **42 days**, with **168-hour prediction chunks**. The final experiment retrained every 48 hours. This is a useful approximation for development, but it is a short calibration period and its retraining schedule differs from deployment. Thirty trials on this one slice can favour a configuration specific to that slice; that is a hypothesis to test across several development windows.

The expanding fit excludes the previous 72 hours, consistent with the displayed maximum label horizon. This is different from the earlier `PurgedKFold`, whose percentage buffer is about 519 observations and whose training sets can contain dates after the held-out block. Neither the 72-hour gap nor purging repairs future-dependent HMM features. Removing the gap merely to recover higher scores would reintroduce invalid information rather than improve the strategy.

### 7. Feature importance: what the model uses versus what generalises

![Feature importance by gain](assets/original/cell-49-figure-1.png)

This plot is **XGBoost gain**, although an earlier section heading called it Random Forest MDI. Gain describes improvements attributed to tree splits during fitting. A high value means that a feature helped the fitted model partition training data; it does not establish incremental predictive value on new market periods.

![Correlation clusters and dendrogram](assets/original/cell-52-figure-1.png)

The team grouped features by their Pearson-correlation structure to inspect families of related signals. This addresses a practical interpretation problem: importance may be divided among several indicators carrying similar information.

![Clustered feature importance](assets/original/cell-53-figure-1.png)

The grouped view summarises importance by feature family. In the supplied code, grouped importance is an aggregation of individual importances, not a separate experiment jointly removing or permuting an entire family. A true group-ablation test would refit or perturb the family together under the same validation protocol.

![Permutation feature importance](assets/original/cell-55-figure-1.png)

Permutation importance asks how macro-F1 changes when a feature is shuffled. Positive values indicate that shuffling harmed this model's score in this sample; values near zero or below zero suggest limited or unstable benefit. Error bars describe variability across permutations, not confidence intervals across future market regimes.

The original prose described permutation importance as unbiased. In fact, correlated predictors can substitute for each other, and random shuffling disrupts temporal structure. Cell 55 also uses the 2021 test labels. It is useful for a retrospective diagnosis, but choosing new features from this plot makes that year development data for the next version.

![Gain versus permutation importance](assets/original/cell-56-figure-1.png)

The comparison is intended to distinguish a feature that looks important during fitting from one whose information helps on the evaluation sample. Disagreement is a prompt for controlled removal experiments, not an automatic rule for deleting a feature. Both these diagnostics concern `xgb_model`, the one-shot model; they do not directly attribute losses in the separate expanding-window backtest.

### 8. SHAP: the team's interpretation of the decision logic

![SHAP summary](assets/original/cell-58-figure-1.png)

The original SHAP interpretation emphasised **HMM regimes**, with favourable states pushing the model toward bullish predictions. Weekly autocorrelation was interpreted as a possible mean-reversion signal, while volatility appeared to have a nonlinear role: conditions associated with momentum could differ from extreme, potentially exhausted moves. The author also highlighted the transformed-price feature as a contributor to the model's directional bias.

These explanations connect the fitted model to the feature design, but they do not prove economic causality or future profitability. In particular, high importance for a potentially future-dependent HMM feature is a reason to retest the whole pipeline with causal states. The transformed-price implementation does not justify a claim of mathematically established fractional differencing or stationarity. This plot explains the one-shot model, not the complete sequence of expanding-window refits.

### 9. Classification results and probability diagnostics

![Confusion matrices](assets/original/cell-62-figure-1.png)

| Original experiment | Accuracy | Macro-F1 | ROC-AUC | Evaluation population |
|---|---:|---:|---:|---|
| Random Forest | 52.08% | 0.4182 | 0.4944 | 8,664 observations |
| XGBoost, one-shot | 52.03% | 0.4567 | 0.5159 | 8,664 observations |
| XGBoost, expanding | 48.19% | 0.4725 | 0.4721 | 8,664 observations |
| XGBoost + meta-filter | 56.60% | 0.4655 | 0.5336 | 3,341 selected observations |
| Always bullish | 53.06% | — | — | Full sample |
| Always bullish on filtered subset | 56.84% | — | — | Same selected subset |

The confusion matrices make directional imbalance visible. The full-sample classifiers do not outperform the always-bullish baseline on accuracy, while the expanding model's AUC is below 0.5 in this saved sample. These results do not establish a stable directional advantage. They also do not prove that simply reversing every prediction will work on a fresh period.

![ROC and precision-recall curves](assets/original/cell-63-figure-1.png)

ROC and precision-recall curves examine ranking across thresholds rather than only the 0.50 decision. They show why changing a threshold is a separate problem from improving the information in the model. When probability ranking is weak, threshold tuning may only select noise; any tuning must use chronological development data and report the coverage it leaves.

![Directional signals and prediction probabilities](assets/original/cell-64-figure-1.png)

The signal plot connects predictions with the market path and their probability scores. Its practical question is whether the strategy should take a position whenever a score falls just above or below 0.50. The original long/short rule offers no neutral state, even when the model is uncertain. A 0.51 score is not evidence of a cost-adjusted opportunity, especially when class weights and model fitting have not produced calibrated probabilities.

### 10. Original backtest: reading the loss correctly

![Original equity, drawdown, positions and rolling Sharpe](assets/original/cell-67-figure-1.png)

The saved expanding-window backtest lost **68.2% before costs**, compared with **+48.4%** for buy-and-hold over its own interval. Maximum drawdown was **−85.6%**, and the reported hourly-annualised Sharpe was **−0.757**. These are historical notebook outputs, not a live trading record or a newly reproduced run.

Cell 67 maps `ew_preds` to **−1/+1 exposure**, shifts the position by one row and multiplies by hourly close-to-close returns. It neither uses the meta-filter nor includes a cash region, transaction costs, funding or borrowing. A lagged position prevents one particular timing error; it does not establish realistic fills or remove leakage introduced earlier in the pipeline.

The original conclusion attributed the curve to lagging reversals and being on the wrong side of crashes and rallies. That is a plausible interpretation to investigate, but the aggregate loss and drawdown alone cannot establish it. A trade-level attribution separating long P&L, short P&L, turnover and market regimes is needed.

![Original monthly returns: strategy versus buy-and-hold](assets/original/cell-67-figure-2.png)

The monthly comparison helps inspect whether underperformance is concentrated in specific market phases or persists over time. It supports asking when the model fails, rather than selecting a more attractive annual headline. Exact contributions require the original aligned prediction and return series; the figures alone cannot quantify each cause.

## Why the original strategy performed poorly

There is no measured decomposition assigning a percentage of the loss to each defect. The strongest supported diagnosis is a **weak predictive signal combined with a mismatch between the learned target and the traded return, applied as continuous long/short exposure**. Evaluation defects make it harder to decide whether additional model complexity helps.

| Issue | Evidence in this notebook | Implication |
|---|---|---|
| Target and trade horizon differ | Labels scan 12–72h; P&L uses the next hourly return. | A correct multi-day trend label can still lose money under the hourly trading rule. |
| Little demonstrated discrimination | Expanding AUC 0.4721; one-shot AUC 0.5159. | Hyperparameter complexity has not produced reliable directional information. |
| Classification objective differs from trading objective | Optuna selects macro-F1; trading depends on move size, exposure and costs. | Better classification scores need not produce better net returns. |
| No neutral exposure | Final positions are only −1 or +1. | Weak and uncertain forecasts still create market risk. |
| Separate experiments were conflated | Meta-filter and SHAP analyse the one-shot model; final equity uses expanding predictions. | Those diagnostics cannot be presented as direct explanations or improvements of the final backtest. |
| Short calibration and possible drift | 30 walk-forward trials on the last 1,000 training hours; 168h tuning versus 48h final refits. | The selected configuration may not transfer to other periods; this needs controlled comparison. |
| Future-dependent features | Full-sequence HMM decoding; HMM fitted outside individual CV folds. | Evaluation is not fully causal, regardless of the final position lag. |

**What does not explain the reported loss:** fees were omitted, so they did not cause −68.2%; adding realistic costs would generally make the result worse. Accuracy below 50% does not mechanically imply losses because correct and incorrect moves have different magnitudes. Leakage invalidates an evaluation but does not prove the cause or direction of its P&L. Nor should the 72-hour label-availability gap be removed to make results look better.

## How to improve it: ordered, falsifiable experiments

The first objective is to make the experiment answer one well-defined trading question. Profitability is a hypothesis to test, not an expected consequence of correcting code.

1. **Align the target with an executable holding interval.** Start with a fixed 24-hour return target and the same 24-hour holding rule, using only information available before entry. Alternatively retain trend scanning but define exits in advance; never use the ex-post winning horizon to decide how long a live trade would last. The existing reference implements the fixed-horizon approach.
2. **Establish a simple, causal baseline.** Use a small documented feature set, logistic regression and buy-and-hold/cash benchmarks on identical dates and costs. Fit transformations within each training window. Exclude the HMM first; add forward-filtered states only if a controlled ablation supports their value.
3. **Measure the actual trading failure.** For saved timestamped predictions, report long and short contributions, performance by probability bucket, average winning/losing move, exposure, turnover and cost sensitivity. Compare one-shot, expanding and filtered strategies under exactly the same execution rules. This is the missing attribution needed to explain the magnitude of the original loss.
4. **Allow abstention, with development-only selection.** Test long/flat first, then long/flat/short with separate thresholds. Select thresholds and any probability calibration on chronological development folds, using a predeclared net-return/risk objective and minimum sample coverage. A 0.65 threshold was an original suggestion, not an optimised or guaranteed solution. A higher threshold can reduce trading while making selection less reliable.
5. **Test adaptation as an ablation.** Compare expanding and fixed rolling histories, with matched refit schedules in tuning and evaluation. Use several development periods and a small regularised model search before adding complexity. Refitting more often is not automatically better.
6. **Test the team's proposed research directions separately.** Compare 1h, 4h and daily sampling with carefully specified horizons. Test on-chain, funding/open-interest or other external features only with genuine publication-time availability and an economic rationale. Coarser bars reduce sample count; extra data can add leakage as easily as signal.
7. **Reintroduce meta-labelling last.** Generate chronological out-of-fold primary predictions, construct labels for actual net trade outcomes, and fit the filter only on mature historical outcomes. Keep `{0,1}` class labels separate from `{-1,0,+1}` positions. Evaluate the accepted trades against a baseline on the same subset, including coverage and costs.
8. **Freeze the final specification before new evaluation.** The original analysis inspected 2021 and the later reference inspected 2021–2023, so these years are retrospective development evidence. Use a genuinely unexamined chronological period or a prospectively logged paper-trading period for the next final assessment. Retain failed experiments and account for the number of trials.

### What has already been improved, and what remains unproven

The repository's reference implementation aligns a fixed 24-hour prediction/holding interval, uses chronological monthly fits, waits for labels to mature and includes trading costs. It uses logistic regression and histogram gradient boosting, **not a repaired rerun of the original XGBoost/HMM/meta-labelling stack**.

At 7 bps per leg, the logistic reference returned **+19.6%, −6.7%, +12.2%** in 2021–2023; the boosting reference returned **−26.8%, −52.9%, −9.5%**. Logistic AUC remained approximately **0.518, 0.459, 0.526**, and it traded only 13 scheduled intervals in 2023. These outcomes do not establish persistent alpha. They illustrate why methodological correctness and a profitable model are separate achievements.

The cost check is also instructive: logistic 2021 return falls from **42.1% at zero cost**, to **19.6% at 7 bps per leg**, to **−1.8% at 15 bps per leg**. This belongs to the reference experiment, not an attribution of the original −68.2% loss. Detailed curves and records are below and in [results/RESULTS.md](results/RESULTS.md).

Technical references: [hmmlearn decoding semantics](https://hmmlearn.readthedocs.io/en/stable/api.html) and [scikit-learn's separation of classification from decision-threshold optimisation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html). Time-series applications require chronological validation rather than blindly using default random folds.


## Reproduce the reference experiment

Use Python 3.11+ from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
python run.py --csv data/btc_hourly.csv --out results/my_2021_run
```

The original notebook obtains its CSV from this [course-provided download](https://drive.google.com/uc?id=1P_5ykYLd5521QUdCxC_cMytdJ3PqESTw). Download it into `data/btc_hourly.csv` if you have access. Raw data is not redistributed here; its venue, construction and redistribution licence have not been independently established.

Expected schema: first column `date`, then `coin,open,high,low,close,volume`. The inspected file contains **54,496 continuous hourly BTC bars**, from **2018-01-01 01:00** through **2024-03-20 16:00**, without duplicate timestamps or missing values. SHA-256, package versions and date ranges are recorded for each run. OHLC ordering and positive prices are checked. The supplied timestamps have no timezone; the code preserves this convention and **assumes bar-start timestamps**. Confirm the source's timestamp convention before any execution use.

For the additional retrospective checks:

```bash
python run.py --csv data/btc_hourly.csv --start 2022-01-01 --end '2022-12-31 23:00:00' --out results/my_2022_run
python run.py --csv data/btc_hourly.csv --start 2023-01-01 --end '2023-12-31 23:00:00' --out results/my_2023_run
```

## Temporal and execution protocol

1. Features for bar `t` become available at `t + 1h`. No backfill, smoothed HMM states or fitted full-sample transform is used.
2. Execute at the open of `t + 2h`, one full bar after signal availability. The target is the return from that open to the open 24 hours later.
3. Refit monthly on an expanding history, using only labels whose **availability time is strictly earlier than the fit time**. Training audits record this boundary for every fit. A scaler is fitted inside the logistic pipeline on the eligible history only.
4. Use non-overlapping 24-hour holding periods. Hold long above probability 0.55, short below 0.45, otherwise cash. Parameters and thresholds were fixed before these reruns; no search on the reported years was performed.
5. Hold a fixed quantity within each trade and mark the account hourly. Close and reopen in full at each holding boundary, with no netting. Baseline costs are **5 bps fees + 2 bps slippage per entry/exit leg**; 0 and 15 bps sensitivities are also reported. These are illustrative cost scenarios, not broker quotes.
6. A passive buy-and-hold comparator pays entry/exit costs once over the same time interval. Cash earns zero interest. A simple past-24-hour momentum comparator uses the same holding and cost conventions as the active models.

The account is a hypothetical cash-collateralised long/short research model. Borrow/funding costs, margin/liquidation rules, bid/ask data, liquidity and market impact are not modelled. Entry notional equals initial account equity for that trade; leverage can drift while the fixed quantity is held. Results are not executable derivatives returns. Sharpe annualisation uses hourly returns and does not correct serial dependence.

## Results

See `results/RESULTS.md` for all years, models, cost sensitivity and drawdowns. The classifier evaluates 364 scheduled decisions per year; only a subset produce trades. All periods are **retrospective research**: 2021 was already used during the original development, and the entire price history had appeared in the original analysis. None is presented as a pristine prospective holdout.

The baseline drops the HMM, Hurst proxy, fractional-difference level and meta-model until they can justify incremental value. `forward_filter` provides a tested causal HMM state recursion for future controlled work; it is not used to generate the reported baseline results.

## Reference backtests: 2021, 2022 and 2023

These are the later causal reference runs, with separate models and accounting from the original study. Each year starts at equity 1; costs are 7 bps per entry/exit leg. Full tables and cost sensitivity are in [results/RESULTS.md](results/RESULTS.md).

### 2021

![Reference account equity, 2021](results/retrospective_2021/equity.png)

### 2022

![Reference account equity, 2022](results/retrospective_2022/equity.png)

### 2023

![Reference account equity, 2023](results/retrospective_2023/equity.png)

## Files

- `research.py`: features, targets, label-availability filtering, models and accounting.
- `run.py`: reproducible runs, baselines, cost sensitivity, audit CSVs and plots.
- `test_research.py`: tests of future-data invariance, label maturity, execution alignment and cost accounting.
- `notebooks/reference_experiment.ipynb`: a small executable front end.
- `AUDIT.md`: original-code findings and remaining limitations.
- `archive/original_student_research.ipynb`: superseded student code, not the reference execution path.

No open-source licence is asserted over the group's original work; contributor agreement is needed before adding one.
