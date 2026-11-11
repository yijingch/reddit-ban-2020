"""
filter raw Reddit data by a subreddit list from a CSV.
"""

from __future__ import annotations

import argparse
import os
import re
import glob
from typing import Optional
import polars as pl
import pandas as pd
import yaml

from src.utils.data_filter_polars import filter_raw_data_by_subr
from src.utils.argparse_helper import parse_years, fill_path

def subrs_from_csv(path: str, column: str, where: Optional[str]) -> set[str]:
    df = pd.read_csv(path)
    if where:
        df = df.query(where)
    return {str(s).strip() for s in df[column].dropna() if str(s).strip()}


def subrs_from_parquet_dir(path: str, input_years: Optional[list[int]]) -> set[str]:
    files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    if input_years is not None:
        keep = []
        for f in files:
            m = re.search(r"(\d{4})(?=\.parquet$)", os.path.basename(f))
            if m and int(m.group(1)) in input_years:
                keep.append(f)
        files = keep
    if not files:
        raise ValueError(f"No parquet files to read in {path} (after year filter).")
    subrs = (
        pl.scan_parquet(files)
        .select("subreddit")
        .filter(pl.col("subreddit").is_not_null())
        .unique()
        .collect()["subreddit"]
        .to_list()
    )
    return {str(u).strip() for u in subrs if str(u).strip()}

def main() -> None:
    p = argparse.ArgumentParser(description="Filter raw Reddit data by a subreddit list from a CSV.")
    p.add_argument("--input", nargs="+", required=True, help="One or more .csv files and/or filtered data parquet dirs. Subreddits are unioned.")
    p.add_argument("--input-years",
                   help="Only read parquet files whose filename year is in this range, e.g. 2005:2019.")
    p.add_argument("--where", help="Optional pandas query on the CSV, e.g. 'if_banned == 0'.")
    p.add_argument("--exclude", nargs="*", default=[],
                help="CSVs and/or parquet dirs whose subreddits are removed from the final list.")
    p.add_argument("--exclude-where",
                help="Optional pandas query applied to CSV exclude lists, e.g. 'if_banned == 1'.")
    p.add_argument("--column", default="subreddit", help="Column holding subreddit names (default: subreddit).")
    p.add_argument("--filetype", required=True, choices=["submissions", "comments"])
    p.add_argument("--years", required=True, help="'2005:2019' (inclusive) or '2018,2019'.")
    p.add_argument("--output", required=True, help="Output dir. Supports {DATAPATH},{covs},{X},{N}.")
    p.add_argument("--covs")
    p.add_argument("--X")
    p.add_argument("--N")
    p.add_argument("--config", default="src/configs.yml")
    args = p.parse_args()

    with open(args.config) as fh:
        configs = yaml.safe_load(fh)
    subs = {
        "DATAPATH": configs["SERVER_DATA"],
        "ROOTPATH": configs["SERVER_WD"],
        "covs": args.covs,
        "X": args.X,
        "N": args.N,
    }
    inputs = [fill_path(x, subs) for x in args.input]
    output_fpath = fill_path(args.output, subs)
    years = parse_years(args.years)
    input_years = parse_years(args.input_years) if args.input_years else None
    excludes = [fill_path(x, subs) for x in args.exclude]

    subrs: set[str] = set()
    for path in inputs:
        if path.lower().endswith(".csv"):
            subrs |= subrs_from_csv(path, args.column, args.where)
        else:
            subrs |= subrs_from_parquet_dir(path, input_years)

    excluded: set[str] = set()
    for path in excludes:
        if path.lower().endswith(".csv"):
            excluded |= subrs_from_csv(path, args.column, args.exclude_where)
        else:
            excluded |= subrs_from_parquet_dir(path, input_years)

    subrs = {s.lower() for s in subrs}
    excluded = {s.lower() for s in excluded}
    n_before = len(subrs)
    n_removed = len(subrs & excluded)
    subrs = sorted(s for s in (subrs - excluded) if not s.startswith("u_"))
    print(f"# subreddits: {n_before} -> {len(subrs)} ({n_removed} removed by --exclude)")

    os.makedirs(output_fpath, exist_ok=True)
    print("input:", inputs)
    print("# of subreddits to filter for:", len(subrs))
    print("years:", years)
    print("filetype:", args.filetype)
    print("output:", output_fpath)

    filter_raw_data_by_subr(
        subrs=subrs,
        years=years,
        filetype=args.filetype,
        output_fpath=output_fpath,
        raw_data_dir=configs["RAW_DATA"],
    )

if __name__ == "__main__":
    main()

"""
Examples:

# cleaned banned-2020 list
## submissions -- 20260709 done!
python src/preprocess/filtering/filter_raw_data_by_subr_cli.py \
    --input '{DATAPATH}/data/samples/sample_banned_random_3274_250629.csv' \
    --where 'if_banned == 1' \
    --filetype submissions \
    --years 2005:2019 \
    --output '{DATAPATH}/filtered-data-new/submissions/subm-from-subr-banned-2020-subset3274/'

## comments -- 20260710 done!
python src/preprocess/filtering/filter_raw_data_by_subr_cli.py \
    --input '{DATAPATH}/data/samples/sample_banned_random_3274_250629.csv' \
    --where 'if_banned == 1' \
    --filetype comments \
    --years 2005:2019 \
    --output '{DATAPATH}/filtered-data-new/comments/comm-from-subr-banned-2020-subset3274/'

# snowballed samples - neighborhood for banned subreddits
# 650,433 if pooling comment and submission users BEFORE filtering
## submissions -- 20260715 done!
python src/preprocess/filtering/filter_raw_data_by_subr_cli.py \
    --input \
        '{DATAPATH}/filtered-data-new/submissions/subm-from-user-banned-2020-subset3274-lite/' \
        '{DATAPATH}/filtered-data-new/comments/comm-from-user-banned-2020-subset3274-lite/' \
    --input-years 2005:2019 \
    --exclude '{DATAPATH}/data/samples/sample_banned_random_3274_250629.csv' \
    --exclude-where 'if_banned == 1' \
    --filetype submissions \
    --years 2005:2019 \
    --output '{DATAPATH}/filtered-data-new/submissions/subm-from-subr-banned-2020-subset3274-neighbors-only/'

## comments -- 20260715 done! 
python src/preprocess/filtering/filter_raw_data_by_subr_cli.py \
    --input \
        '{DATAPATH}/filtered-data-new/submissions/subm-from-user-banned-2020-subset3274-lite/' \
        '{DATAPATH}/filtered-data-new/comments/comm-from-user-banned-2020-subset3274-lite/' \
    --input-years 2005:2019 \
    --exclude '{DATAPATH}/data/samples/sample_banned_random_3274_250629.csv' \
    --exclude-where 'if_banned == 1' \
    --filetype comments \
    --years 2019:2019 \
    --output '{DATAPATH}/filtered-data-new/comments/comm-from-subr-banned-2020-subset3274-neighbors-only/'

## notes: only 3249 (instead of 3274 will be removed from the list because 25 subreddits only have authors as [deleted])

# random matched sample, controls only
python src/preprocess/filtering/filter_raw_data_by_subr_cli.py \
    --input '{DATAPATH}/data/samples/sample_banned_random_3274_250629.csv' \
    --where 'if_banned == 0' \
    --filetype comments \
    --years 2005:2019 \
    --output '{DATAPATH}/filtered-data-new/comments/comm-from-subr-banned-2020-subset3274-random/'

# matched table (X/N/covs fill both paths)
python src/preprocess/filtering/filter_raw_data_by_subr_cli.py \
    --covs lifetime_d_n_post_author_till2019_log --X 20 --N 10 \
    --input '{DATAPATH}/data/samples/cem-batch4-new/matched_table_banned_3274_cem_X{X}N{N}_{covs}.csv' \
    --where 'if_banned == 0' \
    --filetype submissions \
    --years 2005:2020 \
    --output '{DATAPATH}/filtered-data-new/submissions/subm-from-subr-banned-2020-subset3274-match-{covs}-X{X}N{N}/'

# NEW! now that we have the merged list in data-public/subreddit_sets.csv
for kind in submissions comments; do
  python src/preprocess/filtering/filter_raw_data_by_subr_cli.py \
    --input data-public/subreddit_sets.csv --where "set == 'banned'" \
    --filetype "$kind" --years 2005:2019 \
    --output "{DATAPATH}/filtered-data-test/$kind/${kind:0:4}-from-subr-banned-2020-test/"
done
"""