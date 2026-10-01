"""Causal reference experiment; hourly bars, delayed execution, fixed holding horizon.

This is a methodological rewrite, not a reproduction of the historical XGBoost run.
Timestamps are assumed to identify bar starts. Bars at t become available at t+1h.
Signals from bar t execute at open[t+2h], with one full bar of latency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HOUR = pd.Timedelta(hours=1)


def validate_prices(p: pd.DataFrame) -> None:
    cols = ['open', 'high', 'low', 'close', 'volume']
    if not isinstance(p.index, pd.DatetimeIndex) or not p.index.is_monotonic_increasing:
        raise ValueError('A sorted DatetimeIndex is required.')
    if p.index.has_duplicates or not (p.index.to_series().diff().dropna() == HOUR).all():
        raise ValueError('Require unique, continuous hourly bars; do not silently fill gaps.')
    if len(p) < 200 or not np.isfinite(p[cols].to_numpy()).all():
        raise ValueError('Insufficient or non-finite OHLCV data.')
    if (p[cols[:4]] <= 0).any().any() or (p.volume < 0).any():
        raise ValueError('Prices must be positive and volume non-negative.')
    if (p.high < p[['open', 'close', 'low']].max(axis=1)).any() or (p.low > p[['open', 'close', 'high']].min(axis=1)).any():
        raise ValueError('Invalid OHLC ordering.')
    if 'coin' in p and (p.coin.nunique() != 1 or p.coin.iloc[0] != 'BTC'):
        raise ValueError('Expected one BTC series.')


def features(p: pd.DataFrame) -> pd.DataFrame:
    """No fitted transforms, backfill, state smoothing or absolute price levels."""
    c, o, h, l, v = (p[k].astype(float) for k in ['close', 'open', 'high', 'low', 'volume'])
    r = np.log(c).diff()
    x = pd.DataFrame(index=p.index)
    for w in [1, 6, 24, 72, 168]:
        x[f'log_return_{w}h'] = np.log(c / c.shift(w))
    for w in [24, 72, 168]:
        x[f'volatility_{w}h'] = r.rolling(w).std()
        x[f'ma_distance_{w}h'] = c / c.rolling(w).mean() - 1
    x['parkinson_120h'] = np.sqrt(np.log(h / l).pow(2).rolling(120).mean() / (4 * np.log(2)))
    gk = .5 * np.log(h / l).pow(2) - (2 * np.log(2) - 1) * np.log(c / o).pow(2)
    x['garman_klass_120h'] = np.sqrt(gk.rolling(120).mean().clip(lower=0))
    x['volume_ratio_168h'] = v / v.rolling(168).mean().replace(0, np.nan)
    x['intrabar_return'] = np.log(c / o)
    x['range_ratio'] = (h - l) / c
    # A zero-range bar is valid; its location within the range is defined as neutral.
    x['close_location'] = ((c - l) / (h - l).replace(0, np.nan)).fillna(.5)
    return x.replace([np.inf, -np.inf], np.nan).dropna()


def targets(p: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError('horizon must be positive')
    entry, exit_price = p.open.shift(-2), p.open.shift(-(horizon + 2))
    forward = exit_price / entry - 1
    out = pd.DataFrame({'forward_return': forward,
                        'label': (forward > 0).astype(int),
                        'available_at': p.index + (horizon + 2) * HOUR}, index=p.index)
    return out.loc[forward.notna()]


def eligible_training_rows(x: pd.DataFrame, y: pd.DataFrame, decision_time: pd.Timestamp) -> pd.DatetimeIndex:
    idx = x.index.intersection(y.index)
    # Conservative strict inequality: no label arriving exactly at fit time is used.
    return idx[(idx + HOUR < decision_time) & (y.loc[idx, 'available_at'] < decision_time)]


def forward_filter(log_emissions: np.ndarray, startprob: np.ndarray, transmat: np.ndarray) -> np.ndarray:
    """P(state_t | observations up to t), never Viterbi/full-series smoothing.

    Model parameters and emission transforms must themselves be fitted only on past data.
    Not enabled in the reference baseline; supplied to support a future HMM ablation.
    """
    e = np.asarray(log_emissions, dtype=float)
    start, trans = np.asarray(startprob, float), np.asarray(transmat, float)
    if e.ndim != 2 or e.shape[1] != len(start) or trans.shape != (len(start), len(start)):
        raise ValueError('Incompatible HMM dimensions')
    if np.any(start < 0) or np.any(trans < 0) or not np.isclose(start.sum(), 1) or not np.allclose(trans.sum(axis=1), 1):
        raise ValueError('Invalid probabilities')
    with np.errstate(divide='ignore'):
        log_a, previous = np.log(trans), np.log(start)
    ans = np.empty_like(e)
    for i, row in enumerate(e):
        prior = previous if i == 0 else logsumexp(previous[:, None] + log_a, axis=0)
        posterior = prior + row
        norm = logsumexp(posterior)
        if not np.isfinite(norm):
            raise ValueError('Impossible observation under every state')
        previous = posterior - norm
        ans[i] = np.exp(previous)
    return ans


def walk_forward(p: pd.DataFrame, start: str, end: str, horizon: int = 24,
                 model_name: str = 'logistic', confidence: float = .55) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly expanding refits; fixed hyperparameters; non-overlapping holding intervals."""
    validate_prices(p)
    if not .5 <= confidence < 1:
        raise ValueError('confidence must be in [0.5, 1)')
    x, y = features(p), targets(p, horizon)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    # Define the schedule before prediction, independently of the realized labels.
    idx = x.loc[start:end].index
    idx = idx[(idx + (horizon + 2) * HOUR <= end) & idx.isin(y.index)][::horizon]
    if len(idx) == 0:
        raise ValueError('Empty evaluation interval')
    result, audit = [], []
    for _, dates in pd.Series(idx, index=idx).groupby(idx.to_period('M')):
        dates = pd.DatetimeIndex(dates.to_numpy())
        decision = dates[0] + HOUR
        train_idx = eligible_training_rows(x, y, decision)
        if len(train_idx) < 1000 or y.loc[train_idx, 'label'].nunique() < 2:
            raise ValueError('Need at least 1,000 mature training examples and both classes')
        if model_name == 'logistic':
            model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000, random_state=42))
        elif model_name == 'hist_gradient_boosting':
            model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7,
                    learning_rate=.05, l2_regularization=10, early_stopping=False, random_state=42)
        else:
            raise ValueError('Unknown model')
        model.fit(x.loc[train_idx], y.loc[train_idx, 'label'])
        prob = model.predict_proba(x.loc[dates])[:, 1]
        side = np.where(prob >= confidence, 1, np.where(prob <= 1-confidence, -1, 0))
        rows = pd.DataFrame({'prob_up': prob, 'position': side,
                             'prediction': (prob >= .5).astype(int),
                             'label': y.loc[dates, 'label'].to_numpy(),
                             'forward_return': y.loc[dates, 'forward_return'].to_numpy()}, index=dates)
        result.append(rows)
        audit.append({'fit_at': decision, 'n_train': len(train_idx),
                      'last_training_label_available_at': y.loc[train_idx, 'available_at'].max(),
                      'first_signal_bar': dates[0], 'last_signal_bar': dates[-1]})
    return pd.concat(result), pd.DataFrame(audit)


def backtest(p: pd.DataFrame, signals: pd.DataFrame, horizon: int = 24,
             fee_bps: float = 5, slippage_bps: float = 2) -> pd.Series:
    """Fixed quantity for each H-hour trade; close fully at exit, then reopen.

    Each leg is charged proportional costs on its actual traded notional. No netting
    between consecutive trades, no leverage above initial one-times notional per trade.
    Short borrow/funding are excluded: this is a hypothetical long/short research account.
    Hourly mark-to-market captures intrahorizon drawdowns; cash earns zero interest.
    """
    if fee_bps < 0 or slippage_bps < 0:
        raise ValueError('Costs cannot be negative')
    cost = (fee_bps + slippage_bps) / 10000
    capital, points, previous_exit = 1., {}, None
    for t, row in signals.iterrows():
        enter, leave = t + 2*HOUR, t + (horizon+2)*HOUR
        if previous_exit is not None and enter != previous_exit:
            raise ValueError('Signals must define contiguous non-overlapping holding intervals')
        prices = p.open.loc[enter:leave]
        if len(prices) != horizon+1:
            raise ValueError('Incomplete holding period')
        side = float(row.position)
        if side not in [-1., 0., 1.]:
            raise ValueError('Position must be short, cash or long')
        qty = side * capital / prices.iloc[0]
        wealth = capital + qty * (prices - prices.iloc[0]) - cost * abs(qty * prices.iloc[0])
        wealth.iloc[-1] -= cost * abs(qty * prices.iloc[-1])
        if (wealth <= 0).any():
            raise ValueError('Account insolvent; strategy must stop before further trading')
        points.update(wealth.to_dict())
        capital, previous_exit = float(wealth.iloc[-1]), leave
    return pd.Series(points, name='wealth').sort_index()


def buy_hold(p: pd.DataFrame, index: pd.DatetimeIndex, cost_bps: float = 7) -> pd.Series:
    price = p.open.loc[index]
    f = cost_bps/10000
    wealth = price/price.iloc[0] - f
    wealth.iloc[-1] -= f * price.iloc[-1]/price.iloc[0]
    return wealth.rename('wealth')


def performance(wealth: pd.Series) -> dict:
    r = wealth.pct_change()
    r.iloc[0] = wealth.iloc[0]-1
    vol = r.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(r, 0)**2))
    peak = wealth.cummax().clip(lower=1)
    return {'total_return': float(wealth.iloc[-1]-1),
            'max_drawdown': float((wealth/peak-1).min()),
            'sharpe_hourly_annualized': float(np.sqrt(8760)*r.mean()/vol) if vol > 0 else None,
            'sortino_hourly_annualized': float(np.sqrt(8760)*r.mean()/downside) if downside > 0 else None}


def classification(signals: pd.DataFrame) -> dict:
    y, pred, p = signals.label, signals.prediction, signals.prob_up
    kept = signals.position != 0
    return {'n_decisions': len(y), 'accuracy': float(accuracy_score(y, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'roc_auc': float(roc_auc_score(y, p)) if y.nunique() == 2 else None,
            'always_up_accuracy': float(y.mean()), 'trade_coverage': float(kept.mean()),
            'selected_accuracy': float(accuracy_score(y[kept], pred[kept])) if kept.any() else None,
            'selected_always_up_accuracy': float(y[kept].mean()) if kept.any() else None}
