from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from threadpoolctl import threadpool_limits
from .data import PAIRS, read_prices, read_features, combine_features
from .features import make_target, technical_indicators, legacy_select
from .model import fit_predict
from .backtest import single_asset, portfolio

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description='Run the documented historical FX workflow.')
    parser.add_argument('--demo', action='store_true', help='Use labelled synthetic data included in this repository.')
    parser.add_argument('--data', type=Path, help='Prepared directory with prices/, features/ and optionally macro/.')
    parser.add_argument('--selection', choices=['supplied', 'rebuilt'], default='supplied')
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts/latest')
    parser.add_argument('--explain', action='store_true', help='Also export SHAP beeswarm plots.')
    args = parser.parse_args()
    if not args.demo and args.data is None:
        parser.error('Choose --demo or --data PATH.')
    if args.demo and args.data is not None:
        parser.error('--demo and --data are mutually exclusive.')
    if args.demo and args.selection == 'rebuilt':
        parser.error('The demo contains supplied synthetic features; use --selection supplied.')
    directory = ROOT/'data/demo' if args.demo else args.data
    threadpool_limits(1)
    args.out.mkdir(parents=True, exist_ok=True)
    prices, predictions, metrics = {}, {}, []
    for pair in PAIRS:
        price = read_prices(directory/'prices'/f'{pair}.csv')
        target = make_target(price)
        prices[pair] = price
        out = args.out/pair
        out.mkdir(exist_ok=True)
        target.to_csv(out/'target.csv')
        if args.selection == 'rebuilt':
            indicators = technical_indicators(price)
            features, notes = combine_features(indicators, directory/'macro'/pair)
            features.to_csv(out/'combined_features.csv')
            features, scores = legacy_select(features, target)
            features.to_csv(out/'selected_features.csv')
            scores.to_csv(out/'selection_scores.csv')
            (out/'data_notes.json').write_text(json.dumps(notes, indent=2))
        else:
            features = read_features(directory/'features'/f'{pair}.csv')
        frame, score = fit_predict(features, target, pair, out, explain=args.explain, synthetic=args.demo)
        predictions[pair] = frame
        metrics.append(score)
        print(f'{pair}: train={score["train"]}, test={score["test"]}, R2={score["r2"]:.4f}')
    daily, stats = portfolio(prices, predictions)
    daily.to_csv(args.out/'portfolio_daily.csv')
    (args.out/'portfolio_metrics.json').write_text(json.dumps(stats, indent=2))
    single_asset(prices['USDCNH']).to_csv(args.out/'single_asset.csv', index=False)
    pd.DataFrame(metrics).to_csv(args.out/'model_metrics.csv', index=False)
    (args.out/'run.json').write_text(json.dumps({'dataset_kind': 'synthetic_demo' if args.demo else 'user_supplied',
                                                'selection': args.selection, 'historical_method': True}, indent=2))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    (1+daily).cumprod().plot(ax=ax, color=['#0891b2', '#f59e0b'])
    ax.set(title='Historical portfolio calculation' + (' / SYNTHETIC DEMO' if args.demo else ''), ylabel='Growth of 1')
    fig.tight_layout(); fig.savefig(args.out/'portfolio.png', dpi=140); plt.close(fig)
    print(f'Outputs: {args.out}')
