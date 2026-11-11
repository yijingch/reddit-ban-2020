import polars as pl
import pandas as pd
import numpy as np
import glob, os
import json 
from typing import Dict, List
import shutil  

from src.utils.functions import extract_subreddit_from_permalink  

import yaml
with open("src/configs.yml", "r") as configs:
    configs = yaml.safe_load(configs)

DATAPATH = configs["SERVER_DATA"]
ROOTPATH = configs["SERVER_WD"]
RAW_DATA = configs["RAW_DATA"]
print(configs)

def write_set_author_year_shards(
    subr_set_map: pl.DataFrame,
    years: List[int],
    filetype: str,
    tmp_dir: str,
    flush_every_n_files: int = 20,
):
    """
    Scan Reddit data without loading everything into memory.

    Output shards contain unique:

        set | author | year

    Therefore, if one user posts in multiple subreddits belonging
    to the same set, they are counted only once for that set/year.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    target_subrs = set(subr_set_map.get_column("subreddit").to_list())

    shard_buffer = []
    shard_id = 0

    for year in years:

        print(f"\nProcessing {filetype}, year {year}")

        months = range(6, 13) if year == 2005 else range(1, 13)

        for month in months:

            data_dir = f"{RAW_DATA}/{filetype}/year={year}/month={month}/"
            parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))

            if len(parquet_files) == 0:
                continue

            for i, this_file in enumerate(parquet_files):

                print(f"\t{year}-{month:02d}: " f"{i + 1}/{len(parquet_files)}")

                # Read only subreddit + author
                if (
                    filetype == "submissions"
                    and year in [
                        2005, 2006, 2007, 2008, 2009,
                        2010, 2011, 2012, 2013, 2014, 2021,
                    ]
                ):

                    df = pl.read_parquet(
                        this_file,
                        columns=["author", "permalink"],
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
                        columns=["subreddit", "author"],
                    )

                    df = df.with_columns(
                        pl.col("subreddit")
                        .cast(pl.Utf8)
                        .str.to_lowercase()
                    )

                # Keep only target subreddits
                df = (
                    df
                    .filter(
                        pl.col("subreddit").is_in(target_subrs)
                        & pl.col("author").is_not_null()
                        & ~pl.col("author").is_in(["[deleted]"])
                    )

                    # attach set membership
                    .join(subr_set_map, on="subreddit", how="inner")
                    .with_columns(pl.lit(year).cast(pl.Int32).alias("year"))

                    # IMPORTANT:
                    # collapse across subreddits within the set
                    .select(["set", "author", "year"])
                    .unique(["set", "author", "year"])
                )

                if df.height > 0:
                    shard_buffer.append(df)

                # periodically flush to disk
                if len(shard_buffer) >= flush_every_n_files:

                    out_df = (
                        pl.concat(
                            shard_buffer,
                            how="vertical",
                            rechunk=True,
                        )
                        .unique(["set", "author", "year"])
                    )

                    out_path = os.path.join(
                        tmp_dir,
                        f"{filetype}_set_author_year_"
                        f"{shard_id:05d}.parquet"
                    )

                    out_df.write_parquet(
                        out_path,
                        compression="snappy",
                    )

                    print(
                        f"\tWrote shard {shard_id}: "
                        f"{out_df.height:,} rows"
                    )

                    shard_buffer = []
                    shard_id += 1

    # Final flush
    if shard_buffer:

        out_df = (
            pl.concat(
                shard_buffer,
                how="vertical",
                rechunk=True,
            )
            .unique(["set", "author", "year"])
        )

        out_path = os.path.join(
            tmp_dir,
            f"{filetype}_set_author_year_"
            f"{shard_id:05d}.parquet"
        )

        out_df.write_parquet(
            out_path,
            compression="snappy",
        )

# Calculate new + cumulative unique users
def get_user_counts(shard_paths, suffix):
    """Count annual new users and cumulative unique users for each subreddit set.

    A user is counted as new in the earliest year they appear in a given set
    across the supplied shards. Users may be counted separately in multiple sets.
    """
    author_year = (
        pl.scan_parquet(shard_paths)
        .select(["set", "author", "year"])
        .unique()

        # first appearance of each user in each set
        .group_by(["set", "author"])
        .agg(pl.col("year").min().alias("first_year"))
        .collect(engine="streaming")
    )

    new_users = (
        author_year
        .group_by(["set", "first_year"])
        .agg(pl.len().alias(f"n_new_users_{suffix}"))
        .rename({"first_year": "year"})
    )

    set_names = list(subreddit_sets.keys())
    grid = (
        pl.DataFrame({"set": set_names})
        .join(pl.DataFrame({"year": years}), how="cross")
    )

    cumulative_users = (
        grid
        .join(
            new_users,
            on=["set", "year"],
            how="left",
        )
        .with_columns(
            pl.col(f"n_new_users_{suffix}").fill_null(0)   
        )
        .sort(["set", "year"])
        .with_columns(
            pl.col(f"n_new_users_{suffix}")                
            .cum_sum()
            .over("set")
            .alias(f"n_cumulative_users_{suffix}")         
        )
    )

    return cumulative_users   

if __name__ == "__main__":

    # ============================================================
    # Load subreddit sets
    # ============================================================
    print("Loading subreddit sets...")
    sets_df = pd.read_csv("../../data-public/subreddit_sets.csv")
    subreddit_sets = {name: set(group["subreddit"]) for name, group in sets_df.groupby("set")}

    print("processing for the following subreddit sets:")
    for set_name, subrs in subreddit_sets.items():
        print(f"\t{set_name}: {len(subrs)} subreddits")

    years = list(range(2005, 2020))

    tmp_dir = f"{DATAPATH}/data/tmp/subreddit_set_author_year"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # ============================================================
    # Build subreddit -> set mapping
    # ============================================================
    print("build subreddit -> set mapping...")
    mapping_rows = []

    for set_name, subrs in subreddit_sets.items():
        for subr in subrs:
            mapping_rows.append({
                "subreddit": subr.lower(),
                "set": set_name,
            })

    subr_set_map = pl.DataFrame(mapping_rows)
    print(subr_set_map.group_by("set").len())


    # ============================================================
    # Run for submissions + comments
    # ============================================================
    print("writing set author year temporary shards...")
    write_set_author_year_shards(
        subr_set_map=subr_set_map,
        years=years,
        filetype="submissions",
        tmp_dir=tmp_dir,
    )

    write_set_author_year_shards(
        subr_set_map=subr_set_map,
        years=years,
        filetype="comments",
        tmp_dir=tmp_dir,
    )

    # ============================================================
    # Submissions, comments, and posts
    # ============================================================
    print("counting from shards...")
    subm_shard_paths = glob.glob(os.path.join(tmp_dir, "submissions_set_author_year_*.parquet"))
    comm_shard_paths = glob.glob(os.path.join(tmp_dir, "comments_set_author_year_*.parquet"))
    all_shard_paths = subm_shard_paths + comm_shard_paths

    subm_users = get_user_counts(subm_shard_paths, suffix="subm")
    comm_users = get_user_counts(comm_shard_paths, suffix="comm")
    post_users = get_user_counts(all_shard_paths, suffix="post")

    # ============================================================
    # Combine results
    # ============================================================
    user_counts = (
        subm_users
        .join(comm_users,on=["set", "year"],how="left")
        .join(post_users,on=["set", "year"],how="left")
    )
    user_counts.write_parquet(
        f"{DATAPATH}/data/counts-raw/subreddit_set_user_counts_2005_2019.parquet",
        compression="snappy"
    )
    print("output saved!")

# (reddit-ban) yijingch@asc.upenn.edu@cindsrv01:~/workingdir/reddit-ban-2020$
# python src/preprocess/counting/acc-activity-author-by-subreddit-sets-polars.py
