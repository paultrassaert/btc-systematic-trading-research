import unittest
import numpy as np
import pandas as pd
from research import features, targets, eligible_training_rows, forward_filter, backtest, HOUR


def sample(n=1200):
    rng = np.random.default_rng(42)
    o = 100*np.exp(np.cumsum(rng.normal(0, .004, n)))
    c = o*np.exp(rng.normal(0, .002, n))
    return pd.DataFrame({'open': o, 'close': c, 'high': np.maximum(o,c)*1.01,
                         'low': np.minimum(o,c)*.99, 'volume': rng.uniform(1,100,n)},
                         index=pd.date_range('2020-01-01', periods=n, freq='h'))


class TemporalIntegrityTests(unittest.TestCase):
    def test_future_prices_cannot_change_past_features(self):
        p = sample(); q = p.copy(); q.iloc[800:, :4] *= 3
        pd.testing.assert_frame_equal(features(p).loc[:p.index[799]], features(q).loc[:p.index[799]])

    def test_future_emissions_cannot_change_filtered_states(self):
        e = np.random.default_rng(1).normal(size=(40, 2)); alt = e.copy(); alt[20:] *= 10
        a = np.array([[.95, .05], [.1, .9]])
        np.testing.assert_allclose(forward_filter(e, [.5,.5], a)[:20], forward_filter(alt, [.5,.5], a)[:20])

    def test_training_never_sees_unrealized_labels(self):
        p = sample(); x, y = features(p), targets(p, 24); t = p.index[900]+HOUR
        idx = eligible_training_rows(x,y,t)
        self.assertTrue((y.loc[idx,'available_at'] < t).all())
        self.assertNotIn(p.index[899], idx)

    def test_target_matches_execution_prices(self):
        p = sample(); y = targets(p, 24); i = 800
        self.assertAlmostEqual(y.iloc[i].forward_return, p.open.iloc[i+26]/p.open.iloc[i+2]-1)
        self.assertEqual(y.iloc[i].available_at, p.index[i+26])

    def test_costs_cash_and_long_short_accounting(self):
        p = sample(250); p.loc[:,'open'] = 100.
        s = pd.DataFrame({'position':[1]}, index=[p.index[200]])
        self.assertAlmostEqual(backtest(p,s,24,5,0).iloc[-1], .999)
        s.position = 0
        self.assertAlmostEqual(backtest(p,s,24,5,0).iloc[-1], 1.)
        p.loc[p.index[226],'open'] = 110
        s.position = 1
        self.assertAlmostEqual(backtest(p,s,24,0,0).iloc[-1],1.1)
        s.position = -1
        self.assertAlmostEqual(backtest(p,s,24,0,0).iloc[-1],.9)


if __name__ == '__main__':
    unittest.main()
