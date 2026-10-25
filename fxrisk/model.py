import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


def risk_classes(values, pair):
    if pair == 'JPYCNY':
        low, high = 0.003916, 0.007042
    else:
        low, high = np.quantile(values, [0.25, 0.75])
    return pd.cut(values, [-np.inf, low, high, np.inf], labels=[0, 1, 2]).astype(int)


def fit_predict(features, target, pair, out, explain=False, synthetic=False):
    x, y = features.align(target, join='inner', axis=0)
    x = x.dropna(axis=1, how='all')
    split = int(np.floor(0.8 * len(x)))
    if split < 30 or len(x)-split < 10 or x.shape[1] == 0:
        raise ValueError('Insufficient aligned training/test data.')
    train, test = x.iloc[:split], x.iloc[split:]
    model = Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('scaler', StandardScaler()),
                      ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42,
                                           n_estimators=500, learning_rate=0.01, max_depth=4,
                                           subsample=0.8, colsample_bytree=0.8, n_jobs=1))])
    model.fit(train, y.iloc[:split])
    prediction = model.predict(test)
    actual = y.iloc[split:]
    denominator = (abs(actual.to_numpy()) + abs(prediction)) / 2
    smape = np.mean(np.divide(abs(actual.to_numpy()-prediction), denominator,
                             out=np.zeros_like(denominator), where=denominator != 0)) * 100
    metrics = {'pair': pair, 'features': x.shape[1], 'train': len(train), 'test': len(test),
               'rmse': float(np.sqrt(mean_squared_error(actual, prediction))),
               'mae': float(mean_absolute_error(actual, prediction)), 'r2': float(r2_score(actual, prediction)),
               'smape_percent': float(smape), 'test_start': str(test.index.min()), 'test_end': str(test.index.max())}
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({'Date': test.index, 'Actual': actual.to_numpy(), 'Predicted': prediction,
                          'Predicted_Class': risk_classes(prediction, pair),
                          'True_Class': risk_classes(actual.to_numpy(), pair)})
    frame.to_csv(out/'predictions.csv', index=False)
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(test.index, actual, label='Actual', color='#334155', lw=1)
    ax.plot(test.index, prediction, label='Predicted', color='#0891b2', lw=1)
    ax.set(title=f'{pair}: test split' + (' / SYNTHETIC DEMO' if synthetic else ' / historical reconstruction'), ylabel='Target volatility')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'prediction.png', dpi=140); plt.close(fig)
    if explain:
        import shap
        transformed = model.named_steps['scaler'].transform(model.named_steps['imp'].transform(test))
        values = shap.Explainer(model.named_steps['xgb'])(pd.DataFrame(transformed, columns=test.columns, index=test.index))
        shap.plots.beeswarm(values, max_display=20, show=False)
        plt.tight_layout(); plt.savefig(out/'shap.png', dpi=140); plt.close('all')
    return frame, metrics
