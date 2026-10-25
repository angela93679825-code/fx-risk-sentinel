import numpy as np
import pandas as pd
import ta
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


def make_target(prices):
    """The report's exact target: trailing 20-row std shifted back 5 rows."""
    return prices.set_index('Date')['Close'].pct_change().rolling(20).std().shift(-5).dropna().rename('target_variable')


def technical_indicators(prices):
    df = prices.copy()
    ret = df['Close'].pct_change()
    df['atr_14_lagged'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14).shift(1)
    df['FX_Daily_Returns_lag1'] = ret.shift(1)
    df['FX_Daily_Returns_lag3'] = ret.shift(3)
    df['FX_Returns_rolling_mean_10'] = ret.rolling(10).mean().shift(5)
    df['FX_Returns_rolling_std_10'] = ret.rolling(10).std().shift(5)
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    middle, std = df['Close'].rolling(20).mean(), df['Close'].rolling(20).std()
    df['BB_Middle'] = middle
    df['BB_Upper'] = (middle + 2 * std).shift(10)
    df['BB_Lower'] = (middle - 2 * std).shift(10)
    return df.drop(columns=['Open', 'High', 'Low', 'Close'])


def legacy_select(features, target):
    """Historical C-F selection. Full-sample target use is retained and disclosed."""
    x = features.copy()
    corr = x.corr().abs()
    drops = set()
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            a, b = corr.columns[i], corr.columns[j]
            if corr.iloc[i, j] > 0.85:
                drops.add(a if x[a].std() < x[b].std() else b)
    x = x.drop(columns=drops)
    x, y = x.align(target, join='inner', axis=0)
    x = x.dropna(axis=1, how='all')
    if x.empty:
        raise ValueError('No aligned numeric features.')
    imp = SimpleImputer(strategy='median').fit_transform(x)
    rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=1)
    rf.fit(imp, y)
    importance = pd.Series(rf.feature_importances_, index=x.columns)
    x = x[importance[importance >= 0.005].index]
    ic = x.corrwith(y, method='spearman')
    ic_std = {}
    for col in x:
        values = []
        for i in range(12, len(x)):
            a, b = x[col].iloc[i-12:i], y.iloc[i-12:i]
            values.append(np.nan if a.isna().any() or b.isna().any() else spearmanr(a, b).statistic)
        ic_std[col] = np.nanstd(values)
    scores = pd.DataFrame({'IC': ic, 'rolling_IC_std_legacy_IR': pd.Series(ic_std), 'RF_importance': importance})
    selected = [c for c in x if abs(ic[c]) >= 0.02 and abs(ic_std[c]) >= 0.3]
    if not selected:
        raise ValueError('Historical thresholds selected no features; no silent fallback is applied.')
    return x[selected], scores
