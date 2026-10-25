"""Prepare the supplied FX surf archive locally; never adds it to the public repo."""
from pathlib import Path, PurePosixPath
import argparse
import io
import json
import sys
import zipfile
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fxrisk.data import PAIRS, normalize_prices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('data/private'))
    args = parser.parse_args()
    log = []
    with zipfile.ZipFile(args.archive) as archive:
        names = [n for n in archive.namelist() if not n.startswith('__MACOSX') and not n.endswith('/') and not PurePosixPath(n).name.startswith('.')]
        for pair in PAIRS:
            candidates = [n for n in names if n.endswith('.xlsx') and
                          ('/code/FX_cleaned/'+pair+'/' in n or (pair == 'USDCNH' and n.endswith('/USDCNH/USDCNH.FX.xlsx')))]
            if not candidates:
                raise ValueError(f'No exact price input for {pair}')
            source = candidates[0]
            frame = pd.read_excel(io.BytesIO(archive.read(source)))
            clean = normalize_prices(frame)
            (args.out/'prices').mkdir(parents=True, exist_ok=True)
            clean.to_csv(args.out/'prices'/f'{pair}.csv', index=False)
            matches = [n for n in names if n.endswith('/'+pair+'/excel_outputs/icir_filtered_features.xlsx')]
            if not matches:
                raise ValueError(f'No supplied selected features for {pair}')
            f = pd.read_excel(io.BytesIO(archive.read(matches[0])))
            (args.out/'features').mkdir(exist_ok=True)
            f.to_csv(args.out/'features'/f'{pair}.csv', index=False)
            macro_count = 0
            for n in names:
                parts = PurePosixPath(n).parts
                if len(parts) < 4 or not n.endswith('.xlsx') or parts[-3] != pair:
                    continue
                country = parts[-2]
                if country not in {'US', 'CHN', 'JAP', 'EUR', 'UK'}:
                    continue
                # Prefer the archived per-pair feature tree; avoid duplicate copies.
                if '/multi/' in n:
                    continue
                dest = args.out/'macro'/pair/country/(Path(parts[-1]).stem+'.csv')
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                pd.read_excel(io.BytesIO(archive.read(n)), usecols=[0, 1]).to_csv(dest, index=False)
                macro_count += 1
            log.append({'pair': pair, 'price_source': source, 'rows': len(clean),
                        'dropped_date_rows': len(frame)-len(clean), 'macro_files': macro_count,
                        'feature_source': matches[0]})
    (args.out/'import_manifest.json').write_text(json.dumps(log, indent=2))
    print(f'Prepared local inputs: {args.out}')


if __name__ == '__main__':
    main()
