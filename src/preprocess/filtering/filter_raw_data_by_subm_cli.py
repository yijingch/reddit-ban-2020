"""
Minimal CLI: filter raw Reddit data by a submission-ID list.

Submission IDs can be loaded from:
1. CSV files.
2. Existing filtered submission parquet directories.
3. Existing filtered comment parquet directories.

IDs are normalized by removing the Reddit fullname prefix "t3_".
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import gc
from itertools import islice  
from pathlib import Path

import pandas as pd
import polars as pl
from typing import Optional
import yaml

from src.utils.data_filter_polars import filter_raw_data_by_subm_id, filter_prefiltered_data_by_subm_id
from src.utils.argparse_helper import parse_years, fill_path


SUBM_ID_CANDIDATES = (
    "subm_id",
    "submission_id",
    "link_id",
    "id",
)

def get_subm_id_cache_files(cache_path: str) -> list[str]:
    """Return cached ID parquet files from a single file or cache directory."""
    cache_path = Path(cache_path)

    if cache_path.is_file():
        return [str(cache_path)] if cache_path.suffix == ".parquet" else []

    if not cache_path.exists():
        return []

    return sorted(str(path) for path in cache_path.glob("subm_ids_*.parquet"))


def save_subm_ids_cache(
    subm_ids: set[str],
    cache_dir: str,
    chunk_size: int = 10_000_000,
) -> list[str]:
    """
    Save a large set of normalized submission IDs into chunked Parquet files.
    """

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(cache_dir.glob("subm_ids_*.parquet"))

    if existing_files:
        raise FileExistsError(
            f"Cache directory already contains {len(existing_files)} "
            f"submission-ID files: {cache_dir}"
        )

    iterator = iter(subm_ids)
    output_files = []
    chunk_index = 0
    n_saved = 0

    while True:
        chunk = list(islice(iterator, chunk_size))

        if not chunk:
            break

        output_path = (cache_dir / f"subm_ids_{chunk_index:05d}.parquet")

        pl.DataFrame(
            {"subm_id": chunk}, schema={"subm_id": pl.String}
        ).write_parquet(output_path, compression="zstd", statistics=True)

        output_files.append(str(output_path))
        n_saved += len(chunk)

        print(
            f"Saved ID cache chunk {chunk_index:05d}: "
            f"{len(chunk):,} IDs "
            f"({n_saved:,} total)"
        )

        del chunk
        chunk_index += 1

    if not output_files:
        raise ValueError("No submission IDs were written to the cache.")

    print(
        f"Saved {n_saved:,} submission IDs to "
        f"{len(output_files):,} files in {cache_dir}"
    )

    return output_files


def scan_subm_id_cache(cache_dir: str) -> pl.LazyFrame:
    """Lazily scan an existing submission-ID cache. """

    files = get_subm_id_cache_files(cache_dir)

    if not files:
        raise ValueError(f"No submission-ID cache files found in {cache_dir}")

    schema = pl.scan_parquet(files[0]).collect_schema()

    if "subm_id" not in schema.names():
        raise ValueError(f"Cached files in {cache_dir} do not contain a 'subm_id' column.")

    return (
        pl.scan_parquet(files)
        .select(pl.col("subm_id").cast(pl.String).alias("_subm_id"))
    )


def normalize_subm_id(value: object) -> Optional[str]:
    """Return a bare Reddit submission ID, without the t3_ prefix."""
    if value is None or pd.isna(value):
        return None

    subm_id = str(value).strip()

    if not subm_id:
        return None

    if subm_id.startswith("t3_"):
        subm_id = subm_id[3:]

    return subm_id or None


def subm_ids_from_csv(path: str, column: str, where: Optional[str]) -> set[str]:
    """Read submission IDs from a CSV file."""
    df = pd.read_csv(path)

    if where:
        df = df.query(where)

    if column not in df.columns:
        raise ValueError(
            f"Column {column!r} not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    return {
        subm_id
        for value in df[column].dropna()
        if (subm_id := normalize_subm_id(value)) is not None
    }


def get_parquet_files(path: str, input_years: Optional[list[int]]) -> list[str]:
    """Find parquet files and optionally restrict them by filename year."""
    files = sorted(glob.glob(os.path.join(path, "*.parquet")))

    if input_years is not None:
        files = [
            f
            for f in files
            if (
                (match := re.search(
                    r"(\d{4})(?=\.parquet$)",
                    os.path.basename(f),
                ))
                and int(match.group(1)) in input_years
            )
        ]

    if not files:
        raise ValueError(
            f"No parquet files to read in {path} "
            f"after applying the year filter."
        )

    return files


def detect_subm_id_column(files: list[str]) -> str:
    """
    Detect the submission-ID column in parquet input.

    Preference order:
        subm_id -> submission_id -> link_id -> id

    For filtered submission files, this will usually select "id".
    For filtered comment files, it will usually select "link_id" or "subm_id".
    """
    schema = pl.scan_parquet(files[0]).collect_schema()
    columns = set(schema.names())

    for column in SUBM_ID_CANDIDATES:
        if column in columns:
            return column

    raise ValueError(
        "Could not find a submission-ID column in parquet input. "
        f"Tried {SUBM_ID_CANDIDATES}. "
        f"Available columns: {sorted(columns)}"
    )


def subm_ids_from_parquet_dir(path: str, input_years: Optional[list[int]], column: Optional[str] = None) -> set[str]:
    """Read unique submission IDs from a parquet directory."""
    files = get_parquet_files(path, input_years)
    id_column = column or detect_subm_id_column(files)

    schema = pl.scan_parquet(files[0]).collect_schema()
    if id_column not in schema.names():
        raise ValueError(
            f"Column {id_column!r} not found in {path}. "
            f"Available columns: {schema.names()}"
        )

    ids = (
        pl.scan_parquet(files)
        .select(
            pl.col(id_column)
            .cast(pl.String)
            .str.strip_chars()
            .str.replace(r"^t3_", "")
            .alias("subm_id")
        )
        .filter(
            pl.col("subm_id").is_not_null()
            & (pl.col("subm_id") != "")
        )
        .unique()
        .collect()["subm_id"]
        .to_list()
    )

    return set(ids)



def main() -> None:
    parser = argparse.ArgumentParser(description="Filter raw Reddit data by a submission-ID list.")

    parser.add_argument("--input", nargs="+", default=[], help=(
            "One or more CSV files and/or filtered-data parquet directories. "
            "Submission IDs from all inputs are unioned."))
    parser.add_argument("--exclude", nargs="*", default=[], help=(
            "CSV files and/or filtered-data parquet directories whose "
            "submission IDs are removed from the final list."))
    parser.add_argument("--input-subm-id-column", default="subm_id", help=(
            "Submission-ID column for CSV inputs. For parquet inputs, this "
            "column is used when present; otherwise it is auto-detected. "
            "Default: subm_id."))
    parser.add_argument("--subm-id-cache", required=True, help=(
            "Directory containing the cached submission-ID Parquet files. "
            "If it does not exist, the IDs are constructed from --input and "
            "--exclude, saved there, and then loaded lazily."))
    parser.add_argument("--input-years", help=("Only read parquet files whose filename year is in this range, e.g., 2005:2019."))
    parser.add_argument("--where", help=("Optional pandas query applied to CSV inputs, e.g., 'if_banned == 1'."))
    parser.add_argument("--exclude-years", help=("Only load exclusion parquet files whose filename year is in this range."))
    parser.add_argument("--exclude-where", help=("Optional pandas query applied to CSV exclusion inputs."))
    parser.add_argument("--exclude-subm-id-column", default="subm_id", help="Submission-ID column for CSV inputs whose submission IDs are removed.")
    parser.add_argument("--from-prefiltered", action="store_true", help="Extract data from pre-filtered instead of raw data.")
    parser.add_argument("--prefiltered-path", default=None, help="Prefiltered-data parquet directories to further filter.")
    parser.add_argument("--filetype", required=True, choices=["submissions", "comments"], help="Raw Reddit filetype to filter using the submission-ID list.")
    parser.add_argument("--years", required=True, help="Raw-data years to filter, such as '2005:2019' or '2018,2019'.")
    parser.add_argument("--output", required=True, help="Output directory. Supports {DATAPATH}, {covs}, {X}, and {N}.")
    parser.add_argument("--covs")
    parser.add_argument("--X")
    parser.add_argument("--N")
    parser.add_argument("--config", default="src/configs.yml")
    

    args = parser.parse_args()

    # --subm-dir is only meaningful in the sampling pipeline
    if args.from_prefiltered and args.prefiltered_path is None:
        parser.error("--prefiltered-path is required when filter from prefiltered data.")

    with open(args.config) as fh:
        configs = yaml.safe_load(fh)

    subs = {
        "DATAPATH": configs["SERVER_DATA"],
        "ROOTPATH": configs["SERVER_WD"],
        "covs": args.covs,
        "X": args.X,
        "N": args.N,
    }

    inputs = [
        fill_path(path, subs)
        for path in args.input
    ]
    excludes = [fill_path(path, subs) for path in args.exclude]
    output_fpath = fill_path(args.output, subs)
    prefiltered_fpath = (fill_path(args.prefiltered_path, subs) if args.prefiltered_path is not None else None)
    subm_id_cache = fill_path(args.subm_id_cache, subs)

    years = parse_years(args.years)
    input_years = (parse_years(args.input_years) if args.input_years else None)
    exclude_years = (parse_years(args.exclude_years) if args.exclude_years else input_years)

    cache_files = get_subm_id_cache_files(subm_id_cache)

    if cache_files:
        print(f"Using existing submission-ID cache: {subm_id_cache}")
    else:
        if not inputs:
            parser.error("--input is required when --subm-id-cache does not exist.")

        subm_ids: set[str] = set()

        for path in inputs:
            if path.lower().endswith(".csv"):
                these_ids = subm_ids_from_csv(
                    path=path,
                    column=args.input_subm_id_column,
                    where=args.where,
                )
                source_column = args.input_subm_id_column

            elif os.path.isdir(path):
                # Use --column when it exists. Otherwise auto-detect an ID column.
                files = get_parquet_files(path, input_years)
                schema_names = (
                    pl.scan_parquet(files[0])
                    .collect_schema()
                    .names()
                )

                parquet_column = (
                    args.input_subm_id_column
                    if args.input_subm_id_column in schema_names
                    else None
                )

                source_column = (
                    parquet_column
                    or detect_subm_id_column(files)
                )

                these_ids = subm_ids_from_parquet_dir(
                    path=path,
                    input_years=input_years,
                    column=parquet_column,
                )

            else:
                raise ValueError(
                    f"Input must be a CSV file or parquet directory: {path}"
                )

            print(
                f"Loaded {len(these_ids):,} submission IDs "
                f"from {path} using column {source_column!r}."
            )
            subm_ids |= these_ids
            del these_ids

        # load submission IDs to exclude using the same CSV/parquet logic
        excluded_subm_ids: set[str] = set()

        for path in excludes:
            if path.lower().endswith(".csv"):
                these_ids = subm_ids_from_csv(
                    path=path,
                    column=args.exclude_subm_id_column,
                    where=args.exclude_where,
                )
            elif os.path.isdir(path):
                files = get_parquet_files(path, exclude_years)
                schema_names = pl.scan_parquet(files[0]).collect_schema().names()
                parquet_column = (
                    args.exclude_subm_id_column
                    if args.exclude_subm_id_column in schema_names
                    else None
                )
                these_ids = subm_ids_from_parquet_dir(
                    path=path,
                    input_years=exclude_years,
                    column=parquet_column,
                )
            else:
                raise ValueError(
                    f"Exclude input must be a CSV file or parquet directory: {path}"
                )

            excluded_subm_ids |= these_ids
            del these_ids

        # keep only target submission IDs outside the excluded set
        n_before = len(subm_ids)
        subm_ids -= excluded_subm_ids
        n_after = len(subm_ids)
        n_removed = n_before - n_after
        del excluded_subm_ids

        if not subm_ids:
            raise ValueError("No submission IDs remain after applying --exclude.")

        # IMPORTANT! save once, then release the large Python sets before filtering.
        save_subm_ids_cache(subm_ids, subm_id_cache)
        del subm_ids
        gc.collect()

    # IMPORTANT filtering always uses the on-disk lazy ID source.
    subm_ids_lf = scan_subm_id_cache(subm_id_cache)

    os.makedirs(output_fpath, exist_ok=True)

    if not cache_files:
        print("inputs:", inputs)
        # report the effect of the exclusion
        print(
            f"# of unique submission IDs: {n_before:,} -> {n_after:,} "
            f"({n_removed:,} removed by --exclude)"
        )
    print("raw-data years:", years)
    print("filetype:", args.filetype)
    print("output:", output_fpath)

    if args.from_prefiltered:
        print("*********** FILTERING FROM PRE-FILTERED DATA ***********")
        filter_prefiltered_data_by_subm_id(
            subm_ids_lf=subm_ids_lf,
            prefiltered_path=prefiltered_fpath,
            years=years,
            filetype=args.filetype,
            output_fpath=output_fpath
        )
    else:
        print("*********** FILTERING FROM RAW DATA ***********")
        filter_raw_data_by_subm_id(
            subm_ids_lf=subm_ids_lf,
            years=years,
            filetype=args.filetype,
            output_fpath=output_fpath,
            raw_data_dir=configs["RAW_DATA"],
        )


if __name__ == "__main__":
    main()


"""
Examples
--------


# submissions for user activities (submission only) outside of banned subreddits
python src/preprocess/filtering/filter_raw_data_by_subm_cli.py \
    --input \
        '{DATAPATH}/filtered-data-new/submissions/subm-from-user-banned-2020-subset3274-lite/' \
    --input-years 2005:2019 \
    --filetype submissions \
    --years 2005:2019 \
    --exclude \
        '{DATAPATH}/filtered-data-new/submissions/subm-from-subr-banned-2020-subset3274/' \
    --exclude-years 2005:2019 \
    --from-prefiltered \
    --prefiltered-path \
        '{DATAPATH}/filtered-data-new/submissions/subm-from-subr-banned-2020-subset3274-neighbors-only/' \
    --subm-id-cache \
        '{DATAPATH}/filtered-data-new/submission-ids/user-from-banned-2020-subset3274-neighbors-only-subm-activity-only/' \
    --output \
        '{DATAPATH}/filtered-data-new/submissions/subm-from-subm-by-user-banned-2020-subset3274-neighbors-only/'

120,590,761 (ids from subm)
# of unique submission IDs: 120,590,761 -> 113,760,460 (7,878,260 removed by --exclude)

# comments for user activities outside of banned subreddits
python src/preprocess/filtering/filter_raw_data_by_subm_cli.py \
    --filetype comments \
    --years 2005:2019 \
    --from-prefiltered \
    --prefiltered-path \
        '{DATAPATH}/filtered-data-new/comments/comm-from-subr-banned-2020-subset3274-neighbors-only/' \
    --subm-id-cache \
        '{DATAPATH}/filtered-data-new/submission-ids/user-from-banned-2020-subset3274-neighbors-only-subm-activity-only/' \
    --output \
        '{DATAPATH}/filtered-data-new/comments/comm-from-subm-by-user-banned-2020-subset3274-neighbors-only/'



# if we count both submission and comment activites 

# submissions for user activities outside of banned subreddits
python src/preprocess/filtering/filter_raw_data_by_subm_cli.py \
    --input \
        '{DATAPATH}/filtered-data-new/submissions/subm-from-user-banned-2020-subset3274-lite/' \
        '{DATAPATH}/filtered-data-new/comments/comm-from-user-banned-2020-subset3274-lite/' \
    --input-years 2005:2019 \
    --filetype submissions \
    --years 2005:2019 \
    --exclude \
        '{DATAPATH}/filtered-data-new/submissions/subm-from-subr-banned-2020-subset3274/' \
    --exclude-years 2005:2019 \
    --from-prefiltered \
    --prefiltered-path \
        '{DATAPATH}/filtered-data-new/submissions/subm-from-subr-banned-2020-subset3274-neighbors-only/' \
    --subm-id-cache \
        '{DATAPATH}/filtered-data-new/submission-ids/user-from-banned-2020-subset3274-neighbors-only/' \
    --output \
        '{DATAPATH}/filtered-data-new/submissions/subm-from-subm-comm-by-user-banned-2020-subset3274-neighbors-only/'

120,590,761 (ids from subm)
281,569,213 (ids from comm)
# of unique submission IDs: 339,162,443 -> 331,284,183 (7,878,260 removed by --exclude)

# comments for user activities outside of banned subreddits
python src/preprocess/filtering/filter_raw_data_by_subm_cli.py \
    --filetype comments \
    --years 2005:2019 \
    --from-prefiltered \
    --prefiltered-path \
        '{DATAPATH}/filtered-data-new/comments/comm-from-subr-banned-2020-subset3274-neighbors-only/' \
    --subm-id-cache \
        '{DATAPATH}/filtered-data-new/submission-ids/user-from-banned-2020-subset3274-neighbors-only/' \
    --output \
        '{DATAPATH}/filtered-data-new/comments/comm-from-subm-comm-by-user-banned-2020-subset3274-neighbors-only/'

"""