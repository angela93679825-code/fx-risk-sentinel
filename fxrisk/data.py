from pathlib import Path
import numpy as np
import pandas as pd

PAIRS = ['USDJPY', 'EURUSD', 'EURJPY', 'USDCNH', 'JPYCNY']
COLUMN_MAP = {'日期': 'Date', '开盘价(元)': 'Open', '最高价(元)': 'High',
              '最低价(元)': 'Low', '收盘价(元)': 'Close'}


def normalize_prices(frame):
    df = frame.rename(columns=COLUMN_MAP).copy()
    required = ['Date', 'Open', 'High', 'Low', 'Close']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'Missing price columns: {sorted(missing)}')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    df = df[required]
    if df['Date'].duplicated().any():
        raise ValueError('Duplicate price dates; resolve explicitly before modeling.')
    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce').interpolate(limit_direction='both')
    if df[required[1:]].isna().any().any() or (df['Close'] <= 0).any():
        raise ValueError('Missing or nonpositive prices after normalization.')
    return df


def read_prices(path):
    path = Path(path)
    df = pd.read_excel(path) if path.suffix.lower() == '.xlsx' else pd.read_csv(path)
    return normalize_prices(df)


def read_features(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, errors='raise')
    df = df.apply(pd.to_numeric, errors='coerce').sort_index()
    return df.groupby(level=0).mean() if df.index.duplicated().any() else df


def midas_expand(series):
    """Appendix A weighting; fixed 30/90-day offsets, not release-date alignment."""
    median_gap = series.index.to_series().diff().dt.days.median()
    if 80 <= median_gap <= 100:
        length, step = 4, 90
        j = np.arange(1, length + 1)
        weights = (1 - j / length) ** 0.5
    elif 25 <= median_gap <= 35:
        length, step = 6, 30
        j = np.arange(1, length + 1)
        weights = j ** 1.5 * (length - j + 1) ** 2.5
    else:
        return None
    weights = weights / weights.sum()
    lags = pd.concat([series.shift(k * step, freq='D') for k in range(length)], axis=1)
    return lags.mul(weights, axis=1).sum(axis=1).rename(series.name + '_MIDAS').resample('D').ffill()


def combine_features(indicators, macro_dir, start='2015-08-31', end='2024-12-31'):
    """Appendix A using first two columns of each mapped macro input."""
    daily, mixed, notes = [], [], []
    for path in sorted(Path(macro_dir).rglob('*.csv')):
        df = pd.read_csv(path, usecols=[0, 1])
        dates = pd.to_datetime(df.iloc[:, 0], errors='coerce', format='mixed')
        values = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        name = path.parent.name + '__' + path.stem.replace(' ', '_')
        series = pd.Series(values.to_numpy(), index=dates, name=name)
        series = series[series.index.notna()].dropna().sort_index()
        if series.index.duplicated().any():
            raise ValueError(f'Duplicate macro dates in {path.name}')
        if series.empty:
            notes.append(f'Empty numeric series: {path.name}')
            continue
        daily.append(series.resample('D').ffill().interpolate(limit=3))
        expanded = midas_expand(series)
        if expanded is not None:
            mixed.append(expanded)
        else:
            notes.append(f'No monthly/quarterly MIDAS transformation: {path.name}')
    for col in indicators.columns:
        if col != 'Date':
            series = indicators.set_index('Date')[col].dropna()
            daily.append(series.resample('D').ffill())
    raw = pd.concat(daily, axis=1).loc[start:end]
    names_with_midas = {s.name.removesuffix('_MIDAS') for s in mixed}
    combined = pd.concat(mixed + [raw[[c for c in raw.columns if c not in names_with_midas]]], axis=1)
    combined = combined.loc[start:end].sort_index()
    if not combined.columns.is_unique:
        raise ValueError('Duplicate feature names; resolve country/source mapping explicitly.')
    combined.index.name = 'Date'
    return combined, notes
