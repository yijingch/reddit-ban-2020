import polars as pl
import numpy as np
import glob, os
import json 
from typing import List  # NEW
import shutil  # NEW

from src.utils.data_filter_polars import filter_raw_data_by_subr
from src.utils.functions import extract_subreddit_from_permalink  

import yaml
with open("src/configs.yml", "r") as configs:
    configs = yaml.safe_load(configs)

DATAPATH = configs["SERVER_DATA"]
ROOTPATH = configs["SERVER_WD"]
RAW_DATA = configs["RAW_DATA"]
print(configs)


# ---------------------------------------------------------
# This helper avoids loading all comments into memory.
# It writes reduced shards: subreddit | author | year
# -------------------------------------------------------------
def write_author_year_shards_by_subr(
    subrs: List[str],
    years: List[int],
    filetype: str,
    tmp_dir: str,
    flush_every_n_files: int = 20,
    exclude_deleted: bool = True,
) -> None:
    """
    Memory-safe version of filtering for cumulative author counting.

    Instead of returning all filtered rows, this function writes small parquet
    shards containing unique subreddit-author-year observations.

    Output schema:
        subreddit | author | year
    """

    os.makedirs(tmp_dir, exist_ok=True)

    subrs = [s.lower() for s in subrs if s is not None]
    subr_set = set(subrs)

    print(f"****** Writing {filetype} subreddit-author-year shards ******")
    print("- Number of subreddits:", len(subrs))
    print(f"- Time range: {min(years)}-{max(years)}")
    print("- Temporary output:", tmp_dir)

    shard_buffer = []
    shard_id = 0

    for this_year in years:
        print(f"\nProcessing year: {this_year}")

        if this_year == 2005:
            months = range(6, 13)
        else:
            months = range(1, 13)

        for this_month in months:
            print(f"\tProcessing month: {this_month}")

            data_dir = f"{RAW_DATA}/{filetype}/year={this_year}/month={this_month}/"
            parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))

            if len(parquet_files) == 0:
                continue

            for i, this_file in enumerate(parquet_files):
                print(f"\t\tfile {i + 1}/{len(parquet_files)}")

                # Read only the columns needed for cumulative author counting.
                # Do not read id, created_utc, body, title, etc.
                if (
                    filetype == "submissions"
                    and this_year in [
                        2005, 2006, 2007, 2008, 2009,
                        2010, 2011, 2012, 2013, 2014, 2021
                    ]
                ):
                    df = pl.read_parquet(
                        this_file,
                        columns=["author", "permalink"]
                    )

                    df = df.with_columns(
                        pl.col("permalink")
                        .map_elements(
                            extract_subreddit_from_permalink,
                            return_dtype=pl.Utf8,
                        )
                        .str.to_lowercase()
                        .alias("subreddit")
                    )

                else:
                    df = pl.read_parquet(
                        this_file,
                        columns=["subreddit", "author"]
                    )

                    df = df.with_columns(
                        pl.col("subreddit")
                        .cast(pl.Utf8)
                        .str.to_lowercase()
                        .alias("subreddit")
                    )

                df = (
                    df
                    .select(["subreddit", "author"])
                    .filter(
                        pl.col("subreddit").is_in(subr_set) &
                        pl.col("author").is_not_null()
                    )
                    .with_columns(
                        pl.lit(this_year).cast(pl.Int32).alias("year")
                    )
                )

                if exclude_deleted:
                    df = df.filter(
                        ~pl.col("author").is_in(["[deleted]"])
                    )

                # Reduce immediately to the only unit we need.
                df = df.unique(["subreddit", "author", "year"])

                if df.height > 0:
                    shard_buffer.append(df)

                # Periodically flush to disk so memory does not keep growing.
                if len(shard_buffer) >= flush_every_n_files:
                    out_df = (
                        pl.concat(shard_buffer, how="vertical", rechunk=True)
                        .unique(["subreddit", "author", "year"])
                    )

                    out_path = os.path.join(
                        tmp_dir,
                        f"{filetype}_author_year_shard_{shard_id:05d}.parquet"
                    )

                    out_df.write_parquet(out_path, compression="snappy")
                    print(f"\t\tWrote shard {shard_id}: {out_df.height:,} rows")

                    shard_buffer = []
                    shard_id += 1

    # Final flush
    if len(shard_buffer) > 0:
        out_df = (
            pl.concat(shard_buffer, how="vertical", rechunk=True)
            .unique(["subreddit", "author", "year"])
        )

        out_path = os.path.join(
            tmp_dir,
            f"{filetype}_author_year_shard_{shard_id:05d}.parquet"
        )

        out_df.write_parquet(out_path, compression="snappy")
        print(f"\tWrote final shard {shard_id}: {out_df.height:,} rows")

    print(f"Done writing {filetype} shards.")


# ----------------------------------------------------------------
# cumulative counts from reduced shards.
# This replaces cumulative_unique_authors() for the combined path
# ----------------------------------------------------------------
def cumulative_unique_authors_from_shards(
    shard_dirs: List[str],
    all_subrs: List[str],
    years: List[int],
    prefix: str,
) -> pl.DataFrame:
    """
    Count cumulative unique authors from reduced subreddit-author-year shards.

    This is memory safer than counting from raw submissions/comments.
    """

    shard_paths = []

    for shard_dir in shard_dirs:
        shard_paths.extend(
            glob.glob(os.path.join(shard_dir, "*.parquet"))
        )

    if len(shard_paths) == 0:
        raise ValueError("No shard files found.")

    year_start = min(years)
    year_end = max(years)

    all_subrs = [s.lower() for s in all_subrs if s is not None]

    first_year_df = (
        pl.scan_parquet(shard_paths)
        .filter(
            pl.col("subreddit").is_not_null() &
            pl.col("author").is_not_null() &
            pl.col("year").is_between(year_start, year_end)
        )
        .select(["subreddit", "author", "year"])
        .unique()
        .group_by(["subreddit", "author"])
        .agg(
            pl.col("year").min().alias("first_year")
        )
        .collect(streaming=True)
    )

    yearly_new_authors = (
        first_year_df
        .group_by(["subreddit", "first_year"])
        .agg(
            pl.len().alias("n_new_author")
        )
        .rename({"first_year": "year"})
    )

    grid = (
        pl.DataFrame({"subreddit": all_subrs})
        .join(
            pl.DataFrame({"year": list(years)}),
            how="cross"
        )
    )

    cumulative_long = (
        grid
        .join(yearly_new_authors, on=["subreddit", "year"], how="left")
        .with_columns(
            pl.col("n_new_author").fill_null(0)
        )
        .sort(["subreddit", "year"])
        .with_columns(
            pl.col("n_new_author")
            .cum_sum()
            .over("subreddit")
            .alias("acc_author")
        )
    )

    cumulative_wide = (
        cumulative_long
        .with_columns(
            (pl.lit(prefix) + pl.col("year").cast(pl.Utf8)).alias("year_col")
        )
        .select(["subreddit", "year_col", "acc_author"])
        .pivot(
            values="acc_author",
            index="subreddit",
            columns="year_col",
        )
        .fill_null(0)
    )

    expected_cols = ["subreddit"] + [f"{prefix}{y}" for y in years]

    for col in expected_cols:
        if col not in cumulative_wide.columns:
            cumulative_wide = cumulative_wide.with_columns(
                pl.lit(0).alias(col)
            )

    cumulative_wide = cumulative_wide.select(expected_cols)

    return cumulative_wide


def cumulative_unique_authors(
    df: pl.DataFrame,
    all_subrs: list[str],
    years: list[int],
    prefix: str,
    exclude_deleted: bool = True,
) -> pl.DataFrame:
    """
    Count cumulative unique authors by subreddit.

    For each subreddit and each year_end, this returns the number of unique
    authors who appeared in that subreddit between min(years) and year_end.

    Returns a Polars DataFrame in wide format:
        subreddit | acc_author2005 | acc_author2006 | ... | acc_author2020
    """

    year_start = min(years)
    year_end = max(years)

    if df.height == 0:
        return pl.DataFrame({"subreddit": all_subrs}).with_columns([
            pl.lit(0).alias(f"{prefix}{y}") for y in years
        ])

    df = df.with_columns(
        pl.col("created_utc").dt.year().cast(pl.Int32).alias("year")
    )

    df = df.filter(
        pl.col("subreddit").is_not_null() &
        pl.col("author").is_not_null() &
        pl.col("year").is_between(year_start, year_end)
    )

    if exclude_deleted:
        df = df.filter(
            ~pl.col("author").is_in(["[deleted]"])
        )

    first_year_df = (
        df.select(["subreddit", "author", "year"])
        .unique()
        .group_by(["subreddit", "author"])
        .agg(
            pl.col("year").min().alias("first_year")
        )
    )

    yearly_new_authors = (
        first_year_df
        .group_by(["subreddit", "first_year"])
        .agg(
            pl.len().alias("n_new_author")
        )
        .rename({"first_year": "year"})
    )

    grid = (
        pl.DataFrame({"subreddit": all_subrs})
        .join(
            pl.DataFrame({"year": list(years)}),  # CHANGED: cast range to list
            how="cross"
        )
    )

    cumulative_long = (
        grid
        .join(yearly_new_authors, on=["subreddit", "year"], how="left")
        .with_columns(
            pl.col("n_new_author").fill_null(0)
        )
        .sort(["subreddit", "year"])
        .with_columns(
            pl.col("n_new_author")
            .cum_sum()
            .over("subreddit")
            .alias("acc_author")
        )
    )

    cumulative_wide = (
        cumulative_long
        .with_columns(
            (pl.lit(prefix) + pl.col("year").cast(pl.Utf8)).alias("year_col")
        )
        .select(["subreddit", "year_col", "acc_author"])
        .pivot(
            values="acc_author",
            index="subreddit",
            columns="year_col",
        )
        .fill_null(0)
    )

    expected_cols = ["subreddit"] + [f"{prefix}{y}" for y in years]

    for col in expected_cols:
        if col not in cumulative_wide.columns:
            cumulative_wide = cumulative_wide.with_columns(pl.lit(0).alias(col))

    cumulative_wide = cumulative_wide.select(expected_cols)

    return cumulative_wide


if __name__ == "__main__":

    # -----------------------------
    # Parameters
    # -----------------------------
    TARGET_YEAR = 2020
    YEARS = range(2005, TARGET_YEAR+1)

    ACTIVITY_THRES = 10
    AUTHOR_THRES = 5

    COUNT_DIR = f"{DATAPATH}/data/counts-raw"
    FILTER_OUTPUT_FPATH = ""  # empty filepath to avoid saving

    CLEAN_UP_SHARDS = True
    INCLUDE_COMMENTS = True

    # -----------------------------
    # Get subreddits to count
    # -----------------------------
    label = "2020active"
    active_df = pl.read_parquet(f"{COUNT_DIR}/count_submissions_n_item_n_author_minmax_utc_{TARGET_YEAR}.parquet")
    print(f"# of active subreddits in {TARGET_YEAR}:", active_df.height)

    active_df = active_df.filter(
        (pl.col("n_subm") >= ACTIVITY_THRES) &
        (pl.col("n_subm_author") >= AUTHOR_THRES)
    )
    all_subrs = active_df.get_column("subreddit").str.to_lowercase().drop_nulls().unique().sort().to_list()

    print(f"# of active subreddits in {TARGET_YEAR} after filtering:", active_df.height)
    print("# of unique subreddits to count:", len(all_subrs))

    # load submissions for either counting mode
    subm_subr = filter_raw_data_by_subr(
        subrs=all_subrs,
        years=YEARS,
        filetype="submissions",
        output_fpath=FILTER_OUTPUT_FPATH,
    )
    # if counting submission authors only
    if not INCLUDE_COMMENTS:
        subr_df_subm = cumulative_unique_authors(
            df=subm_subr,
            all_subrs=all_subrs,
            years=YEARS,
            prefix="acc_subm_author",
            exclude_deleted=True,
        )
        subr_df_subm.write_parquet(
            f"{COUNT_DIR}/{label}_subreddits_{len(all_subrs)}_acc_n_subm_author.parquet", 
            compression="snappy"
        )

    else:
        # reduce submissions and comments to small author-year shards.
        tmp_base = f"{DATAPATH}/data/tmp/subreddit_author_year"
        tmp_subm_dir = f"{tmp_base}/submissions_{label}"
        tmp_comm_dir = f"{tmp_base}/comments_{label}"

        # avoid mixing with stale shards from previous interrupted runs
        shutil.rmtree(tmp_subm_dir, ignore_errors=True)
        shutil.rmtree(tmp_comm_dir, ignore_errors=True)

        # already loaded subm_subr above for Option 1. To avoid re-reading raw
        # submissions, write submission author-year shards directly from subm_subr.
        subm_shard_path = os.path.join(tmp_subm_dir,"submissions_author_year_from_loaded_subm.parquet")
        os.makedirs(tmp_subm_dir, exist_ok=True)
        os.makedirs(tmp_comm_dir, exist_ok=True)

        subm_author_year = (
            subm_subr
            .with_columns(
                pl.col("created_utc").dt.year().cast(pl.Int32).alias("year")
            )
            .select(["subreddit", "author", "year"])
            .filter(
                pl.col("subreddit").is_not_null() &
                pl.col("author").is_not_null() &
                ~pl.col("author").is_in(["[deleted]"])
            )
            .unique(["subreddit", "author", "year"])
        )

        subm_author_year.write_parquet(
            subm_shard_path,
            compression="snappy"
        )

        print("Wrote submission author-year shard:", subm_shard_path)
        print("Submission author-year rows:", subm_author_year.height)

        # for comments, never return the full filtered comment dataframe.
        # Read raw files one by one, reduce immediately, and write shards.
        write_author_year_shards_by_subr(
            subrs=all_subrs,
            years=list(YEARS),
            filetype="comments",
            tmp_dir=tmp_comm_dir,
            flush_every_n_files=20,
            exclude_deleted=True,
        )

        # now compute cumulative unique authors from the reduced shards.
        # This counts users across submissions + comments together.
        subr_df_subm_comm = cumulative_unique_authors_from_shards(
            shard_dirs=[tmp_subm_dir, tmp_comm_dir],
            all_subrs=all_subrs,
            years=list(YEARS),
            prefix="acc_post_author",
        )

        print(subr_df_subm_comm.head())
        print(subr_df_subm_comm.shape)

        subr_df_subm_comm.write_parquet(
            f"{COUNT_DIR}/{label}_subreddits_{len(all_subrs)}_acc_n_post_author.parquet", 
            compression="snappy"
        )

        if CLEAN_UP_SHARDS:
            shutil.rmtree(tmp_subm_dir, ignore_errors=True)
            shutil.rmtree(tmp_comm_dir, ignore_errors=True)
            print("Deleted temporary shard directories.")


# (reddit-ban) yijingch@asc.upenn.edu@cindsrv01:~/workingdir/reddit-ban-2020$
# python src/preprocess/counting/acc-activity-author-by-subreddit-by-year-polars.py