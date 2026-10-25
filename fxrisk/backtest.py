import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .data import PAIRS

STRATEGIES = [
    dict(name='enhanced_1', stop=.015, take=.03, hold=10, trend=5, buy=40, sell=65, volatility_multiplier=.8),
    dict(name='enhanced_2', stop=.02, take=.04, hold=15, trend=3, buy=35, sell=65, volatility_multiplier=1.),
    dict(name='enhanced_3', stop=.025, take=.05, hold=20, trend=2, buy=30, sell=70, volatility_multiplier=1.2),
    dict(name='combined', stop=.018, take=.036, hold=12, trend=4, buy=38, sell=62, volatility_multiplier=.9),
    dict(name='trend_filter', stop=.022, take=.055, hold=25, trend=5, buy=35, sell=68, volatility_multiplier=1.1),
]


def single_asset(prices):
    """Appendix L rules; no commission, no position sizing, closed trades only."""
    base = prices.set_index('Date').loc['2023-02-20':'2024-12-31'].copy()
    results = []
    for cfg in STRATEGIES:
        df = base.copy()
        df['return'] = df['Close'].pct_change()
        change = df['Close'].diff()
        rs = change.clip(lower=0).rolling(14).mean() / (-change.clip(upper=0)).rolling(14).mean()
        df['rsi'] = 100 - 100/(1+rs)
        df['volatility'] = df['return'].rolling(10).std()
        df['trend'] = df['Close'].rolling(cfg['trend']).mean()
        df = df.dropna()
        entry, days, trades = None, 0, []
        for i in range(1, len(df)):
            row = df.iloc[i]
            if entry is None:
                if row.rsi < cfg['buy'] and row.Close > row.trend:
                    entry, days = row.Close, 0
            else:
                ret = (row.Close-entry)/entry
                days += 1
                if row.rsi > cfg['sell'] or ret <= -cfg['stop'] or ret >= cfg['take'] or days >= cfg['hold']:
                    trades.append(ret)
                    entry, days = None, 0
        results.append({'strategy': cfg['name'], 'trades': len(trades),
                        'total_return': float(np.prod(1+np.array(trades))-1) if trades else 0.})
    return pd.DataFrame(results)


def portfolio(prices, predictions):
    """Appendix K with explicit asset order; full-period risk gate retained."""
    returns = {}
    bounds = []
    for pair in PAIRS:
        close = prices[pair].set_index('Date').loc['2023-02-20':'2024-12-31']['Close']
        returns[pair] = close.pct_change()
        pred = predictions[pair].copy()
        pred['Date'] = pd.to_datetime(pred['Date'])
        pred = pred[pred.Date.between('2023-02-20', '2024-12-31')]
        if pred.empty:
            raise ValueError(f'No predictions in backtest interval for {pair}')
        bounds.append((.05, .05) if (pred.Predicted_Class == 2).mean() > .3 else (0., 1.))
    data = pd.concat(returns, axis=1).reindex(columns=PAIRS).dropna()
    if len(data) < 26:
        raise ValueError('Fewer than 26 aligned return rows.')
    output, summary = {}, {}
    def neg_sharpe(w, mean, cov):
        vol = np.sqrt(w @ cov @ w)
        return -(w @ mean - .025) / vol if vol > 0 else np.inf
    for name, limits in [('standard', [(0., 1.)]*5), ('risk_filtered', bounds)]:
        previous = np.full(5, .2)
        daily, dates = [], []
        failures = 0
        for start in range(0, len(data)-21-5+1, 5):
            train, test = data.iloc[start:start+21], data.iloc[start+21:start+26]
            fit = minimize(neg_sharpe, np.full(5, .2), args=(train.mean()*365, train.cov()*365),
                           method='SLSQP', bounds=limits,
                           constraints={'type': 'eq', 'fun': lambda w: w.sum()-1})
            if not fit.success:
                failures += 1
                continue
            turnover = abs(fit.x-previous).sum()
            daily.extend(test @ fit.x - turnover*.001/5)
            dates.extend(test.index)
            previous = fit.x
        if not daily:
            raise ValueError(f'All optimization windows failed for {name}')
        values = np.asarray(daily)
        output[name] = pd.Series(values, index=dates)
        vol = values.std(ddof=1)*np.sqrt(365)
        summary[name] = {'annualized_return_arithmetic': float(values.mean()*365),
                         'annualized_volatility': float(vol),
                         'sharpe': float((values.mean()*365-.025)/vol),
                         'test_rows': len(values), 'optimizer_failures': failures}
    summary['risk_bounds'] = dict(zip(PAIRS, bounds))
    summary['risk_gate_triggered'] = any(b != (0., 1.) for b in bounds)
    return pd.DataFrame(output).rename_axis('Date'), summary
