import pandas as pd 
import numpy as np 
import polars as pl
from typing import List
from tqdm import tqdm
import glob, os
from src.utils.functions import extract_subreddit_from_permalink

import yaml
with open("src/configs.yml", "r") as configs:
    configs = yaml.safe_load(configs)

DATAPATH = configs["SERVER_DATA"]
ROOTPATH = configs["SERVER_WD"]
RAW_DATA = configs["RAW_DATA"]
print(configs)

# ============================================================
# Configurations
# ============================================================

COLS_TO_KEEP_MAP = {
    "submissions": ["id", "author", "subreddit", "created_utc"],
    "comments": ["id", "link_id", "parent_id", "created_utc", "author", "subreddit"],
}
OUTPUT_DIR = f"/cinddata/reddit-ban-study/data/counts-raw/"

def get_months(this_year):
    if this_year == 2005:
        return range(6, 13)
    return range(1, 13)


def normalize_subreddit(df: pl.DataFrame, filetype: str, this_year: int) -> pl.DataFrame:
    """
    Normalize subreddit column across comments/submissions and older Reddit files.
    """
    if filetype == "comments" or this_year in [2015, 2016, 2017, 2018, 2019, 2020, 2022]:
        df = df.with_columns(
            pl.col("subreddit")
            .cast(pl.Utf8)
            .str.to_lowercase()
            .alias("subreddit")
        )

    elif this_year in [
        2005, 2006, 2007, 2008, 2009,
        2010, 2011, 2012, 2013, 2014, 2021
    ]:
        df = df.with_columns(
            pl.col("permalink")
            .map_elements(
                extract_subreddit_from_permalink,
                return_dtype=pl.Utf8
            )
            .str.to_lowercase()
            .alias("subreddit")
        )

    return df


def load_filetype_year(filetype: str, this_year: int) -> pl.DataFrame:
    """
    Load one filetype, either comments or submissions, for one year.
    Returns a normalized dataframe with a shared schema.
    """
    print(f"\nProcessing filetype: {filetype}")

    cols_to_keep = COLS_TO_KEEP_MAP[filetype]
    months = get_months(this_year)
    year_dfs: List[pl.DataFrame] = []

    for this_month in months:
        print("\t\tProcessing month:", this_month)

        data_dir = f"{RAW_DATA}/{filetype}/year={this_year}/month={this_month}/"
        parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))

        month_dfs: List[pl.DataFrame] = []

        for i, this_file in enumerate(tqdm(parquet_files, desc="files", unit="file", leave=False)):

            df = pl.read_parquet(this_file)

            df = normalize_subreddit(df, filetype, this_year)

            df = (
                df
                .select(cols_to_keep)
                .with_columns([
                    pl.col("id").cast(pl.Utf8),
                    pl.col("author").cast(pl.Utf8),
                    pl.col("subreddit").cast(pl.Utf8),
                    pl.col("created_utc").cast(pl.Datetime("ms")),
                    pl.lit(filetype).alias("item_type"),
                ])
                .select([
                    "id",
                    "author",
                    "subreddit",
                    "created_utc",
                    "item_type",
                ])
            )

            if df.height > 0:
                month_dfs.append(df)

        if month_dfs:
            df_month = pl.concat(month_dfs, how="vertical", rechunk=True)
            year_dfs.append(df_month)

    if year_dfs:
        return pl.concat(year_dfs, how="vertical", rechunk=True)

    return pl.DataFrame(
        schema={
            "id": pl.Utf8,
            "author": pl.Utf8,
            "subreddit": pl.Utf8,
            "created_utc": pl.Datetime("ms"),
            "item_type": pl.Utf8,
        }
    )


def aggregate_separate(df_year: pl.DataFrame, filetype: str) -> pl.DataFrame:
    """
    Current behavior:
    count comments OR submissions separately.
    """
    prefix = filetype[:4]  # "comm" or "subm"

    return (
        df_year
        .with_columns(
            pl.col("created_utc").dt.year().alias("year")
        )
        .group_by(["subreddit", "year"])
        .agg([
            pl.col("id").n_unique().alias(f"n_{prefix}"),
            pl.col("author").n_unique().alias(f"n_{prefix}_author"),
            pl.col("created_utc").min().dt.date().alias("min_date"),
            pl.col("created_utc").max().dt.date().alias("max_date"),
        ])
    )


def aggregate_joint(df_year: pl.DataFrame) -> pl.DataFrame:
    """
    Joint behavior:
    count comments and submissions together, while also keeping separate counts.

    Important:
    n_author is the unique number of authors across BOTH comments and submissions.
    """
    return (
        df_year
        .with_columns(
            pl.col("created_utc").dt.year().alias("year")
        )
        .group_by(["subreddit", "year"])
        .agg([
            # item counts
            pl.col("id")
            .filter(pl.col("item_type") == "comments")
            .n_unique()
            .alias("n_comm"),

            pl.col("id")
            .filter(pl.col("item_type") == "submissions")
            .n_unique()
            .alias("n_subm"),

            # separate author counts
            pl.col("author")
            .filter(pl.col("item_type") == "comments")
            .n_unique()
            .alias("n_comm_author"),

            pl.col("author")
            .filter(pl.col("item_type") == "submissions")
            .n_unique()
            .alias("n_subm_author"),

            # joint author count across comments AND submissions
            pl.col("author")
            .n_unique()
            .alias("n_author"),

            # temporal coverage
            pl.col("created_utc").min().dt.date().alias("min_date"),
            pl.col("created_utc").max().dt.date().alias("max_date"),
        ])
    )


if __name__ == "__main__":

    YEARS = range(2005, 2023)
    COUNT_MODE = "separate"  # options: "separate", "joint"
    # Used only when COUNT_MODE == "separate"
    FILETYPE = "comments"  # options: "comments", "submissions"

    if COUNT_MODE not in ["separate", "joint"]:
        raise ValueError("COUNT_MODE must be either 'separate' or 'joint'.")

    for this_year in YEARS:
        print("\n***********************")
        print(f"Processing {this_year}:")
        print("***********************")

        if COUNT_MODE == "separate":
            # current behavior: only comments OR only submissions
            df_year = load_filetype_year(filetype=FILETYPE, this_year=this_year)

            print("aggregating dataframe...")
            df_aggr = aggregate_separate(df_year, filetype=FILETYPE)

            output_fname = f"count_{FILETYPE}_n_item_n_author_minmax_utc_{this_year}.parquet"

        elif COUNT_MODE == "joint":
            # new behavior: comments AND submissions together
            df_comments = load_filetype_year(filetype="comments", this_year=this_year)
            df_submissions = load_filetype_year(filetype="submissions", this_year=this_year)

            df_year = pl.concat(
                [df_comments, df_submissions],
                how="vertical",
                rechunk=True
            )

            print("aggregating dataframe...")
            df_aggr = aggregate_joint(df_year)

            output_fname = f"count_joint_n_item_n_author_minmax_utc_{this_year}.parquet"

        print(df_aggr.head())

        print("saving counts...")
        df_aggr.write_parquet(
            f"{OUTPUT_DIR}{output_fname}",
            compression="snappy"
        )

# (reddit-ban) yijingch@asc.upenn.edu@cindsrv01:~/workingdir/reddit-ban$
# python src/counting/activity-author-by-subreddit-by-year-polars.py