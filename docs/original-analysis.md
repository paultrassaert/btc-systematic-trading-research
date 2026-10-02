# Original analysis and figures

These are the author/team-written analysis cells and saved figures from the supplied original notebook, preserved in their original order. Source cell numbers are zero-based. The figures are historical saved outputs, not newly rerun experiments. Course/interview instructions, private API documentation, data and connection code are excluded.

**Reading notes:** these historical explanations include claims of leakage-free or “live” evaluation that are not established by the implementation. Forward-window labels require purging by their full information interval, HMM inference must be causal, and the supplied fractional-difference implementation was a proxy. Below-50% directional accuracy does not by itself determine profitability. The original backtest omitted costs and is distinct from the later causal reference. See [the audit](../AUDIT.md), which supersedes such methodological claims.

<!-- Original notebook cell 13 -->

---
## 1 - Feature Engineering

We construct features across six categories:

| Category | Features |
|---|---|
| Momentum | RSI (72h, 168h), MACD, Stochastic Oscillator, Williams %R, CCI |
| Trend | EMA ratios (24h, 72h, 168h), ADX, Ichimoku cloud position |
| Volatility | ATR, Bollinger Bands (width + %B), Parkinson, Garman-Klass (120h, 336h), Realized Vol |
| Volume | Volume MA ratio, Chaikin Money Flow (CMF) |
| Serial Correlation | ACF lags (1, 3, 12, 24, 168h), Lagged returns |
| Regime | HMM hidden state (Bear/Neutral/Bull), trained on `frac_diff_04` + `gk_vol_336` |
| Fractional Diff. | Memory-preserving stationary log-price (`d=0.4`) |
| Calendar | Hour of day, day of week, day of month |

> **Rationale**: Rather than feeding raw prices (non-stationary) to the model, we transform them into stationary features capturing different aspects of market dynamics. The combination of short-term momentum, medium-term trend, and volatility regime features gives the model orthogonal information sources.

<!-- Original notebook cell 16 -->

### Regime Detection — Gaussian HMM

Market regimes are identified via a 3-state Gaussian HMM fitted on two rolling z-scored features (`frac_diff_04`, `gk_vol_336`), trained exclusively on in-sample data to avoid lookahead bias. States are relabeled post-hoc by ascending mean 24h return: 0 = Bear, 1 = Neutral, 2 = Bull.

![HMM regimes over time](../assets/original/cell-17-figure-1.png)

*HMM regimes over time. Original saved output, cell 17.*

<!-- Original notebook cell 18 -->

In the initial development phase, I applied a global standard scaler across the entire dataset, but this approach proved inadequate when tested on the 2021 period. During that year, Bitcoin experienced unprecedented price levels and volatility, far beyond the distribution observed in the 2018–2020 training data. As a result, the model encountered a phenomenon I describe as “scale saturation,” where most 2021 observations appeared as extreme outliers. Without a dynamic reference point, the Hidden Markov Model effectively became locked into a persistent bull regime, as inputs consistently fell several standard deviations above the historical mean. This severely limited the usefulness of the regime signal for risk management.

To address this issue, I implemented a rolling Z-score normalization using a 720-hour (30-day) window, where each observation is scaled relative to its recent history. This dynamic approach enables contextual normalization, allowing the model to interpret market conditions based on the current environment rather than outdated statistics. As a result, the HMM was able to distinguish between local bullish and bearish phases within the broader 2021 uptrend, restoring meaningful regime variation. More importantly, this method improves robustness in real-world applications by adapting continuously to evolving market conditions, while avoiding look-ahead bias.

![Forward returns by HMM regime](../assets/original/cell-19-figure-1.png)

*Forward returns by HMM regime. Original saved output, cell 19.*

![Feature correlations](../assets/original/cell-20-figure-1.png)

*Feature correlations. Original saved output, cell 20.*

<!-- Original notebook cell 21 -->

---
## 2 - Creating Labels

**Trend Scanning** (Marcos López de Prado, *Advances in Financial Machine Learning*, 2018) generates binary labels by:

1. For each observation $t$, consider a set of look-forward windows $[t, t+L]$ for $L \in [L_{\min}, L_{\max}]$
2. Fit an OLS regression $\log P_s = \alpha + \beta s$ over each window
3. Record the $t$-statistic of $\hat{\beta}$
4. Assign label $= \text{sign}(\hat{\beta}^*)$ where $L^* = \arg\max_L |t_{\hat{\beta}}|$

This approach avoids look-ahead bias by only using data in the forward window to determine the label, and produces labels that reflect *statistically significant* trend directions rather than arbitrary fixed-horizon returns.

To address the high computational intensity of scanning multiple look-forward windows across a multi-year hourly dataset, I implemented parallel processing via n_jobs=-1 and utilized the highly optimized numpy.polyfit function, ensuring both high-speed execution and the mathematical precision required for OLS-based trend detection.

> **Design choice**: We use $L_{\min}=12$ hours and $L_{\max}=120$ hours, targeting intermediate-term trends, consistent with typical crypto trading horizons.

![Trend-scanning labels and selected horizons](../assets/original/cell-23-figure-1.png)

*Trend-scanning labels and selected horizons. Original saved output, cell 23.*

<!-- Original notebook cell 24 -->

The implementation of the Trend Scanning algorithm yielded an average optimal look-forward horizon of approximately 81 hours. This result is quantitatively significant as it indicates that the most robust price trends within the Bitcoin hourly dataset typically mature over a roughly 3.4-day window. By dynamically selecting this horizon through the maximization of the absolute $t$-statistic, the labeling process effectively filters out high-frequency intraday noise and mean-reverting volatility that often lead to false signals in shorter timeframes.

An average horizon of 81 hours represents a 'macro-structural' scale of price action, providing a stable and reliable target for the subsequent XGBoost model. This duration is long enough to capture genuine, sustained market momentum while remaining reactive enough to identify major regime shifts. This balance ensures that the primary model learns to identify patterns associated with significant directional moves rather than over-fitting to localized, stochastic price spikes.

By targeting these multi-day trends, the strategy significantly enhances the overall signal-to-noise ratio, focusing the model's predictive power on the most statistically meaningful price developments.

<!-- Original notebook cell 26 -->

---
## 3 - Model Development

### Strategy Overview

We implement the following pipeline:

1. **Purged K-Fold CV** — avoids label leakage by purging overlapping samples at fold boundaries
2. **Baseline: Random Forest** — fast, interpretable
3. **Primary: XGBoost** — state-of-the-art gradient boosting, optimised via Optuna
4. **Expanding-Window Production Training** — mimics real-time deployment

> **On look-ahead bias**: The expanding-window strategy ensures the model is never trained on data from the future. Features are computed on the full dataset for computational efficiency (rolling indicators), but the model sees only past data during training.

<!-- Original notebook cell 27 -->

### 3.1 Strategy Rationale & Technical Challenges

Note to the reader: The following section outlines the transition from a naive baseline to a robust quantitative framework. This evolution was driven by the specific challenges of the 2021 Bitcoin market.

The implementation of a Machine Learning strategy in the cryptocurrency market highlights a fundamental divergence from classic predictive approaches: extreme non-stationarity and a low Signal-to-Noise Ratio. During the initial development phase, our static 'one-shot' model was immediately confronted with the massive Concept Drift of 2021. The unprecedented Bitcoin bull run rendered static learning obsolete, leading the model to develop a significant directional bias—becoming a 'perma-bull' unable to distinguish signal from noise. This model 'laziness' was also manifested by premature early stopping; a symptom not of poor tuning, but of the model’s inability to extract persistent alpha beyond a few trees in a highly noisy environment.

To address these structural flaws, we adopted a more rigorous approach. We replaced exhaustive GridSearch with Bayesian optimization via Optuna to more efficiently explore the complex XGBoost hyperparameter space. Furthermore, to combat class imbalance and avoid the 'accuracy trap,' we redefined the objective function around the F1-macro score and integrated a dynamic ratio, forcing the model to better detect rare bearish reversals rather than simply following the dominant trend. Finally, to ensure robustness and statistical validity, we abandoned classic cross-validation in favor of a Purged K-Fold scheme with a strict 72-hour embargo, eliminating all look-ahead bias and ensuring our 'expanding window' approach remains fully compatible with real-world trading conditions.

However, it is crucial to note that the introduction of this embargo led to a significant drop in observed performance. This suggests that a portion of previous results may have been influenced by data leakage. We view this degradation as the necessary cost of a more realistic and honest evaluation. Nevertheless, a question remains regarding the fine-tuning of this embargo: while indispensable for model integrity, we seek to ensure it is optimally configured so as not to excessively penalize performance beyond what is strictly necessary.

<!-- Original notebook cell 28 -->

### 3.2 Purged K-Fold Cross-Validation

<!-- Original notebook cell 30 -->

### 3.3 - Random Forest Baseline

<!-- Original notebook cell 32 -->

Initially, I included calendar-based features such as hour_of_day, day_of_week, and day_of_month in the model. The rationale was to allow the algorithm to capture potential market microstructure effects, such as institutional trading hours or specific weekend volatility patterns that often impact the 24/7 cryptocurrency market. However, a subsequent SHAP analysis revealed a significant risk: day_of_month emerged as the most influential feature for the model’s predictions.

This high importance was identified as a critical flaw rather than a discovery. In financial machine learning, relying heavily on a feature like 'day of the month' often indicates that the model is overfitting to spurious correlations, random coincidences in the training data that have no fundamental predictive power for the future . By relying on these 'lazy' seasonal markers, the model was ignoring genuine market signals like volatility regimes and momentum.

Consequently, I made the strategic decision to drop these temporal features. This forced the XGBoost model to rely exclusively on technical and statistical indicators, ensuring the final strategy is grounded in actual market dynamics rather than calendar noise.

<!-- Original notebook cell 34 -->

### 3.4.1 - XGBoost With Bayesian Hyperparameter Optimisation

<!-- Original notebook cell 36 -->

A striking observation during this optimization phase is the model's extremely rapid convergence, with Early Stopping locking the optimal number of trees at only 25 iterations. This behavior is symptomatic of an environment with a very low Signal-to-Noise Ratio, which is typical for hourly cryptocurrency data.

I attempted to force a more granular exploration by significantly lowering the learning_rate, hoping to allow the model to build more trees and capture persistent micro-structures.

However, this approach yielded no significant performance gains: the model reaches its generalization peak very early. Beyond these 25 trees, the algorithm immediately begins to overfit the market's random noise, causing the validation error to diverge. This confirms that in a 'One-Shot' regime, the amount of exploitable alpha is limited, and model simplicity remains the best defense against data instability.

<!-- Original notebook cell 39 -->

The diagnostic highlights a clear directional bias in the model: although the 2021 test data is relatively balanced (~53% bullish), the XGBoost model predicts a bullish regime over 80% of the time. This behavior reflects a common issue in financial machine learning, where models tend to default to the historically dominant class. Trained on a period characterized by strong upward trends, the model has effectively learned a “long by default” strategy, especially when signals are weak or noisy.

Combined with the high noise of hourly crypto features and the use of a standard 0.50 classification threshold, even low-confidence predictions are converted into active positions. As a result, the model behaves like a biased buy-and-hold strategy, capturing upside moves but failing to adequately protect against drawdowns—leading to weaker overall performance despite strong recall on bullish periods.

<!-- Original notebook cell 40 -->

#### Meta-Labeling

<!-- Original notebook cell 42 -->

The implementation of a Random Forest-based Risk Manager, calibrated with a 0.45 threshold, led to a visible shift in performance metrics. By filtering out 61.4% of the primary model's signals, the classification accuracy rose from 52.03% to 56.60%.

While these figures suggest that Meta-Labeling could act as an effective noise filter, they must be interpreted with significant caution. In the highly non-stationary environment of 2021, such an uplift might partially result from stochastic luck or selection bias. By drastically reducing the trade frequency, we are essentially working with a smaller sample size where the impact of a few 'lucky' winning streaks is amplified.

Furthermore, it remains unclear whether this improvement represents a true structural discovery of Alpha or simply a lucky alignment between the Meta-Model’s fixed logic and the specific volatility regimes of that year. While the reduction in trade frequency theoretically lowers cumulative transaction costs, the risk of overfitting the Risk Manager to the 2021 'noise' is real. These preliminary results are encouraging, but they highlight the need for further robustness tests across multiple market cycles before claiming a definitive advantage.

<!-- Original notebook cell 43 -->

### 3.4.2 - Expanding-Window Training strategy

<!-- Original notebook cell 44 -->

Initially, I adopted a naive approach by using hyperparameters optimized for the 'one-shot' model. However, this proved ineffective because the one-shot model aims to generalize over a long static period, whereas the Expanding Window focuses on short-term horizons.

To bridge this gap, I designed a Walk-Forward optimization that mimics the actual 2021 test conditions. I used the 2018–2020 dataset, reserving the final 1,000 hours (approximately two months) as a validation proxy. While my original intent was to simulate 48-hour windows over three months to perfectly match the final test phase, the computational cost was prohibitive. Consequently, I optimized the study by using a 168-hour (7-day) chunk size over the final two months of 2020, striking a balance between rigorous calibration and execution time.

<!-- Original notebook cell 47 -->

---
## 4 - Feature Importance Analysis <a id='4'></a>

We perform three complementary analyses:

1. **MDI (Mean Decrease Impurity)**: built-in Random Forest importance based on Gini impurity reduction — fast but biased towards high-cardinality and correlated features.

2. **Permutation Feature Importance (PFI)**: model-agnostic importance by measuring accuracy drop when a feature is randomly shuffled — slower but unbiased.

3. **Clustered Feature Importance**: features are grouped by their correlation structure, then MDI/PFI is computed at the *cluster level* to handle multi-collinearity.

4. **SHAP (SHapley Additive exPlanations)**: A game-theoretic approach to explain the output of our XGBoost model. Unlike MDI or PFI, SHAP provides both magnitude and direction: it tells us not only which features are important but how they push the probability towards a "Bull" or "Bear" signal.

<!-- Original notebook cell 48 -->

### 4.1 - Mean Decrease Impurity (MDI)

![Feature importance by gain](../assets/original/cell-49-figure-1.png)

*Feature importance by gain. Original saved output, cell 49.*

<!-- Original notebook cell 50 -->

### 4.2 - Clustered Feature Importance

![Correlation clusters and dendrogram](../assets/original/cell-52-figure-1.png)

*Correlation clusters and dendrogram. Original saved output, cell 52.*

![Clustered feature importance](../assets/original/cell-53-figure-1.png)

*Clustered feature importance. Original saved output, cell 53.*

<!-- Original notebook cell 54 -->

### 4.3 - Permutation Feature Importance (PFI)

![Permutation feature importance](../assets/original/cell-55-figure-1.png)

*Permutation feature importance. Original saved output, cell 55.*

![Gain versus permutation importance](../assets/original/cell-56-figure-1.png)

*Gain versus permutation importance. Original saved output, cell 56.*

<!-- Original notebook cell 57 -->

### 4.4 - SHAP Explainability

![SHAP summary](../assets/original/cell-58-figure-1.png)

*SHAP summary. Original saved output, cell 58.*

<!-- Original notebook cell 59 -->

The SHAP summary analysis reveals that the XGBoost model follows a structured, regime-dependent decision logic rather than relying on isolated indicators. Latent market regimes extracted from the HMM emerge as the dominant drivers, strongly pushing predictions toward bullish outcomes when the model identifies favorable states.

At the same time, weekly autocorrelation introduces a mean-reversion signal, where highly trending conditions tend to shift predictions toward bearish expectations. Volatility features exhibit a non-linear impact: elevated volatility is generally associated with bullish momentum, although extreme conditions can signal potential exhaustion and downside risk.

Additionally, fractional differentiation plays a key role by preserving long-term memory while maintaining stationarity, enabling the model to capture underlying growth dynamics. Overall, the model combines regime detection, temporal dependence, and volatility structure, which explains its strong bullish bias during sustained trending periods such as 2021.

<!-- Original notebook cell 60 -->

## 5 - Model Evaluation

![Confusion matrices](../assets/original/cell-62-figure-1.png)

*Confusion matrices. Original saved output, cell 62.*

![ROC and precision-recall curves](../assets/original/cell-63-figure-1.png)

*ROC and precision-recall curves. Original saved output, cell 63.*

![Directional signals and prediction probabilities](../assets/original/cell-64-figure-1.png)

*Directional signals and prediction probabilities. Original saved output, cell 64.*

<!-- Original notebook cell 66 -->

## 6 - Backtesting Engine & Visualisation

![Original backtest: chart 1](../assets/original/cell-67-figure-1.png)

*Original backtest: chart 1. Original saved output, cell 67.*

![Original backtest: chart 2](../assets/original/cell-67-figure-2.png)

*Original backtest: chart 2. Original saved output, cell 67.*

<!-- Original notebook cell 68 -->

## 7 - Conclusions & Perspectives

<!-- Original notebook cell 69 -->

### 7.1 2021 Out-of-Sample Results

The 2021 results expose a stark underperformance of the strategy ($-68.2\%$) relative to a passive Buy & Hold benchmark ($+48.4\%$), revealing the model's limitations in a live market environment.

**Directional signal failure.** With a prediction accuracy of $48.19\%$, the model performs worse than random chance. In a Long/Short framework, accuracy below $50\%$ means the model systematically takes contrarian positions, mechanically destroying performance rather than generating alpha.

**Structural risk profile issue.** The Max Drawdown of $-85.6\%$ indicates the model remained long during crash phases while being absent or short during rallies. This behavior reflects a temporal lag: the model reacts to past price movements without anticipating market reversals.

**Impact of Purged K-Fold with embargo.** Although these techniques significantly degraded apparent in-sample performance, they successfully eliminated the data leakage biases present in earlier versions. The results reported here therefore reflect a more realistic performance estimate, highlighting the model's inability to extract alpha in a high-noise environment subject to violent regime breaks as observed throughout 2021.

<!-- Original notebook cell 70 -->

### 7.2 General Conclusion: Process over Outcome

Beyond raw performance figures, this project should be evaluated through the lens of methodological rigor rather than profitability alone.

The goal was not merely to generate returns, but to build a robust infrastructure aligned with the most demanding standards of modern quantitative finance. To that end, we integrated advanced techniques including fractional differentiation to preserve time-series memory, Hidden Markov Models to capture latent market regimes, and purged cross-validation with embargo to guarantee the statistical integrity of all reported results.

The central takeaway is that model sophistication does not guarantee alpha generation. Despite the use of advanced tools such as Bayesian hyperparameter optimization, the signal-to-noise ratio of high-frequency Bitcoin data remains extremely low. The model suffered in particular from a persistent directional bias and an inability to adapt to the violent concept drift observed in 2021. This underscores a fundamental truth in highly efficient market environments: risk management is more critical than directional prediction.

<!-- Original notebook cell 71 -->

### 7.3 Limitations & Future Directions

Several improvements could meaningfully strengthen the robustness and relevance of the model:

- **Stricter meta-labeling:** Execute trades only when predicted probability exceeds a high confidence threshold (e.g., $65\%$), filtering out weak signals and accepting neutrality for the majority of timesteps.
- **Coarser time resolution:** Shifting to 4-hour or daily data would reduce microstructural noise and improve the exploitable signal-to-noise ratio.
- **Alternative features:** Incorporating exogenous data sources — on-chain metrics, market sentiment indicators — would provide information that price action alone cannot capture, potentially improving regime detection and directional accuracy.

