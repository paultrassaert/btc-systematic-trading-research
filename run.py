"""Run an explicitly retrospective comparison, not a held-out performance claim."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from research import walk_forward, backtest, buy_hold, classification, performance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--start', default='2021-01-01')
    parser.add_argument('--end', default='2021-12-31 23:00:00')
    parser.add_argument('--horizon', type=int, default=24)
    parser.add_argument('--out', default='results/run')
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.csv, index_col=0, parse_dates=True)
    config = vars(args).copy()
    config['csv'] = Path(args.csv).name
    config['out'] = out.name
    config.update({'data_sha256': hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(),
                   'data_rows': len(data), 'data_first_bar': str(data.index.min()),
                   'data_last_bar': str(data.index.max()), 'python': platform.python_version(),
                   'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__,
                   'confidence': .55, 'fee_bps_per_leg': 5, 'slippage_bps_per_leg': 2,
                   'status': 'retrospective research; original 2021 period was already inspected'})
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    summaries, equities = {}, {}
    for model in ['logistic', 'hist_gradient_boosting']:
        signals, audit = walk_forward(data, args.start, args.end, args.horizon, model)
        signals.to_csv(out/f'{model}_signals.csv', index_label='signal_bar')
        audit.to_csv(out/f'{model}_training_audit.csv', index=False)
        stats = classification(signals)
        for bps in [0, 7, 15]:
            wealth = backtest(data, signals, args.horizon, fee_bps=bps, slippage_bps=0)
            stats[f'cost_{bps}bps_per_leg'] = performance(wealth)
            if bps == 7:
                equities[model] = wealth
        summaries[model] = stats
        print(model, json.dumps(stats), flush=True)
    index = equities['logistic'].index
    equities['buy_and_hold'] = buy_hold(data, index)
    equities['cash'] = pd.Series(1., index=index)
    summaries['buy_and_hold_7bps'] = performance(equities['buy_and_hold'])
    summaries['cash'] = performance(equities['cash'])
    # Momentum baseline: same holding periods, same execution and cost convention.
    momentum = signals.copy()
    momentum['position'] = np.sign(data.close.pct_change(24).loc[momentum.index])
    equities['24h_momentum'] = backtest(data, momentum, args.horizon)
    summaries['24h_momentum_7bps'] = performance(equities['24h_momentum'])
    (out/'metrics.json').write_text(json.dumps(summaries, indent=2, allow_nan=False)+'\n')
    frame = pd.DataFrame(equities)
    frame.to_csv(out/'equity.csv', index_label='open_timestamp')
    ax = frame.plot(figsize=(11, 5), linewidth=1.2)
    ax.set(title=f'Retrospective BTC comparison | fixed {args.horizon}h holding periods | costs included', ylabel='Account value (initial = 1)', xlabel='')
    ax.grid(alpha=.2)
    plt.tight_layout(); plt.savefig(out/'equity.png', dpi=160); plt.close()


if __name__ == '__main__':
    main()
