"""Create a debug subset of a dataset by taking the first N rows from each split."""
import os
import argparse
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description='Create debug subset of dataset')
    parser.add_argument('-s', '--source', default='USPTO_FULL', help='Source dataset name')
    parser.add_argument('-t', '--target', default='USPTO_FULL_debug', help='Target dataset name')
    parser.add_argument('-n', '--num-rows', type=int, default=500, help='Number of rows per split')
    args = parser.parse_args()

    source_dir = PROJECT_ROOT / 'data' / args.source / 'raw'
    target_dir = PROJECT_ROOT / 'data' / args.target / 'raw'
    target_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'val', 'test']:
        src = source_dir / f'raw_{split}.csv'
        dst = target_dir / f'raw_{split}.csv'
        df = pd.read_csv(src, nrows=args.num_rows)
        df.to_csv(dst, index=False)
        print(f'{split}: wrote {len(df)} rows to {dst}')

if __name__ == '__main__':
    main()