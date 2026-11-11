import pandas as pd 
import numpy as np 
import polars as pl
from typing import List
import glob, os
from src.utils.functions import extract_subreddit_from_permalink
from tqdm.auto import tqdm

import yaml
with open("src/configs.yml", "r") as configs:
    configs = yaml.safe_load(configs)

DATAPATH = configs["SERVER_DATA"]
ROOTPATH = configs["SERVER_WD"]
RAW_DATA = configs["RAW_DATA"]
print(configs)


def count_author_subreddit_year(
    years,
    filetype,
    output_dir,
    cols_to_keep=None,
):
    """
    Count number of submissions/comments each author posted in each subreddit-year.

    Output schema:
        author
        subreddit
        year
        n_subm / n_comm
        min_utc
        max_utc
    """

    os.makedirs(output_dir, exist_ok=True)

    item_col = f"n_{filetype[:4]}"  # n_subm or n_comm

    for this_year in years:
        print("\n***********************")
        print(f"Processing {this_year}:")
        print("***********************")

        output_fname = f"author_subreddit_count_{filetype}_minmax_utc_{this_year}.parquet"
        output_path = os.path.join(output_dir, output_fname)

        if os.path.exists(output_path):
            print(f"Already exists, skipping: {output_path}")
            continue

        if this_year == 2005:
            months = range(6, 13)
        else:
            months = range(1, 13)

        agg_dfs: List[pl.DataFrame] = []

        for this_month in months:
            print("\tProcessing month:", this_month)

            data_dir = f"{RAW_DATA}/{filetype}/year={this_year}/month={this_month}/"
            parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))

            if not parquet_files:
                continue

            for this_file in tqdm(parquet_files, desc=f"{this_year}-{this_month:02d}", unit="file"):

                # Read only columns needed for this task
                if cols_to_keep is not None:
                    df = pl.read_parquet(this_file, columns=cols_to_keep)
                else:
                    # minimum needed columns
                    needed_cols = ["id", "author", "created_utc"]

                    # subreddit exists directly for comments and some submission years
                    if filetype == "comments" or this_year in [2015, 2016, 2017, 2018, 2019, 2020, 2022]:
                        needed_cols.append("subreddit")
                    else:
                        needed_cols.append("permalink")

                    df = pl.read_parquet(this_file, columns=needed_cols)

                if df.height == 0:
                    continue

                # Get subreddit column
                if filetype == "comments" or this_year in [2015, 2016, 2017, 2018, 2019, 2020, 2022]:
                    df = df.with_columns(
                        pl.col("subreddit")
                        .cast(pl.Utf8)
                        .str.to_lowercase()
                    )
                else:
                    # Prefer vectorized regex over map_elements when possible
                    df = df.with_columns(
                        pl.col("permalink").map_elements(
                            extract_subreddit_from_permalink,
                            return_dtype=pl.Utf8
                        ).alias("subreddit"))

                # Standardize core columns
                df = df.with_columns([
                    pl.col("id").cast(pl.Utf8),
                    pl.col("author").cast(pl.Utf8),
                    pl.col("created_utc").cast(pl.Datetime("ms")),
                ])

                # Optional but usually sensible
                df = df.filter(
                    pl.col("author").is_not_null()
                    & pl.col("subreddit").is_not_null()
                    & (pl.col("author") != "")
                    & (pl.col("subreddit") != "")
                    # uncomment if you want to exclude deleted users
                    # & (pl.col("author") != "[deleted]")
                )

                # File-level aggregation
                # If you trust each item id appears once, len() is faster than n_unique().
                df_aggr = (
                    df
                    .with_columns(
                        pl.lit(this_year).alias("year")
                    )
                    .group_by(["author", "subreddit", "year"])
                    .agg([
                        pl.len().alias(item_col),
                        pl.col("created_utc").min().alias("min_utc"),
                        pl.col("created_utc").max().alias("max_utc"),
                    ])
                )

                if df_aggr.height > 0:
                    agg_dfs.append(df_aggr)

        if not agg_dfs:
            print("No data found for year:", this_year)
            continue

        print("Combining file-level aggregates...")

        df_year_aggr = (
            pl.concat(agg_dfs, how="vertical", rechunk=True)
            .group_by(["author", "subreddit", "year"])
            .agg([
                pl.col(item_col).sum().alias(item_col),
                pl.col("min_utc").min().alias("min_utc"),
                pl.col("max_utc").max().alias("max_utc"),
            ])
            .sort(["subreddit", "author"])
        )

        print(df_year_aggr.head())
        print("n rows:", df_year_aggr.height)

        print("saving counts...")
        df_year_aggr.write_parquet(output_path, compression="zstd")

        print("saved:", output_path)

if __name__ == "__main__":

    YEARS = range(2005, 2020)
    # filetype = "submissions"
    FILETYPE="comments"
    OUTPUT_DIR = "/cinddata/reddit-ban-study/data/counts-raw/"
    # os.mkdir(OUTPUT_DIR, exist_ok=True)

    count_author_subreddit_year(years=YEARS, output_dir=OUTPUT_DIR, filetype=FILETYPE)

# (reddit-ban) yijingch@asc.upenn.edu@cindsrv01:~/workingdir/reddit-ban$
# python src/counting/activity-by-author-by-subreddit-by-year-polars.py