"""
Minimal CLI: filter raw Reddit data by a user list.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Optional

import pandas as pd
import polars as pl
import yaml

from src.utils.data_filter_polars import filter_raw_data_by_user
from src.utils.argparse_helper import parse_years, fill_path


def users_from_csv(path: str, column: str, where: Optional[str]) -> set[str]:
    df = pd.read_csv(path)
    if where:
        df = df.query(where)
    return {str(u).strip() for u in df[column].dropna() if str(u).strip()}


def users_from_parquet_dir(path: str, input_years: Optional[list[int]]) -> set[str]:
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
    authors = (
        pl.scan_parquet(files)
        .select("author")
        .filter(pl.col("author").is_not_null())
        .unique()
        .collect()["author"]
        .to_list()
    )
    return {str(u).strip() for u in authors if str(u).strip()}


def main() -> None:
    p = argparse.ArgumentParser(description="Filter raw Reddit data by a user list.")
    p.add_argument("--input", nargs="+", required=True,
                   help="One or more .csv files and/or filtered-data parquet dirs. Users are unioned.")
    p.add_argument("--input-years",
                   help="Only read parquet files whose filename year is in this range, e.g. 2005:2019.")
    p.add_argument("--where", help="Optional pandas query applied to CSV inputs, e.g. 'if_banned == 1'.")
    p.add_argument("--column", default="author", help="Username column for CSV inputs (default: author).")
    p.add_argument("--filetype", required=True, choices=["submissions", "comments"],
                   help="Raw data filetype to filter with the user list.")
    p.add_argument("--years", required=True, help="Raw-data years to filter: '2005:2019' or '2018,2019'.")
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

    users: set[str] = set()
    for path in inputs:
        if path.lower().endswith(".csv"):
            users |= users_from_csv(path, args.column, args.where)
        else:
            users |= users_from_parquet_dir(path, input_years)

    users -= {"[deleted]"}

    os.makedirs(output_fpath, exist_ok=True)
    print("inputs:", inputs)
    print("# of users to filter for:", len(users))
    print("years:", years)
    print("filetype:", args.filetype)
    print("output:", output_fpath)

    filter_raw_data_by_user(
        users=users,
        years=years,
        filetype=args.filetype,
        output_fpath=output_fpath,
        raw_data_dir=configs["RAW_DATA"],
    )


if __name__ == "__main__":
    main()


"""
Examples:

# users from filtered subm+comm dirs of the banned sample
python src/preprocess/filtering/filter_raw_data_by_user_cli.py \
    --input \
        '{DATAPATH}/filtered-data-new/submissions/subm-from-subr-banned-2020-subset3274/' \
        '{DATAPATH}/filtered-data-new/comments/comm-from-subr-banned-2020-subset3274/' \
    --input-years 2005:2019 \
    --filetype submissions \
    --years 2005:2019 \
    --output '{DATAPATH}/filtered-data-new/submissions/subm-from-user-banned-2020-subset3274-lite/'

# users from filtered subm+comm dirs of the banned sample
python src/preprocess/filtering/filter_raw_data_by_user_cli.py \
    --input \
        '{DATAPATH}/filtered-data-new/submissions/subm-from-subr-banned-2020-subset3274/' \
        '{DATAPATH}/filtered-data-new/comments/comm-from-subr-banned-2020-subset3274/' \
    --input-years 2005:2019 \
    --filetype comments \
    --years 2005:2019 \
    --output '{DATAPATH}/filtered-data-new/comments/comm-from-user-banned-2020-subset3274-lite/'


# users from random subs
python src/preprocess/filtering/filter_raw_data_by_user_cli.py \
    --input \
        '{DATAPATH}/filtered-data/submissions/subm-from-subr-banned-2020-subset3274-random/' \
        '{DATAPATH}/filtered-data/comments/comm-from-subr-banned-2020-subset3274-random/' \
    --input-years 2005:2019 \
    --filetype submissions \
    --years 2005:2019 \
    --output '{DATAPATH}/filtered-data/submissions/subm-from-user-banned-2020-subset3274-random-lite/'


python src/preprocess/filtering/filter_raw_data_by_user_cli.py \
    --input \
        '{DATAPATH}/filtered-data/submissions/subm-from-subr-banned-2020-subset3274-random/' \
        '{DATAPATH}/filtered-data/comments/comm-from-subr-banned-2020-subset3274-random/' \
    --input-years 2005:2019 \
    --filetype comments \
    --years 2005:2019 \
    --output '{DATAPATH}/filtered-data/comments/comm-from-user-banned-2020-subset3274-random-lite/'

"""