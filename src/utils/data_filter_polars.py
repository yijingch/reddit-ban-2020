## ----------------------------------------------------------
## revamping everything in data_filter.py by transitioning 
## from pandas to polars
## ----------------------------------------------------------

import pandas as pd 
import numpy as np 
import polars as pl
from typing import List
import glob, os
import gc
import yaml
from src.utils.functions import extract_subreddit_from_permalink
from src.utils.data_loader_polars import load_filtered_data

cols_to_keep_map = {
    # "submissions": ["id","created_utc","author", "subreddit", "title", "selftext", "url"],
    "submissions": ["id", "author", "subreddit", "created_utc"], # lite version
    # "comments": ["id", "link_id", "parent_id", "created_utc", "author", "subreddit", "body"],
    "comments": ["id", "created_utc", "author", "subreddit"], # lite version
}

def filter_raw_data_by_subr(
        subrs: List[str],
        years: List[int],
        filetype:str = "submissions",
        output_fpath: str = "",
        raw_data_dir: str = "") -> pl.DataFrame:
    """Filter Reddit submission data by specified subreddits and years (Polars version),
    and save yearly results to parquet files. 
    raw_data_dir defaults to RAW_DATA from src/configs.yml.

    Args
    ----
    subrs : list[str]
        List of subreddit names (as lowercase strings) to retain in the filtered data.
    years : list[int]
        List of years (as integers) to process.
    output_fpath : str, optional (default = "")
        Output directory path where filtered yearly parquet files will be saved.
        Each file will be named 'submissions_filtered_by_subreddits_{year}.parquet'.
    """

    print(f"****** Filtering {filetype.capitalize()} by Subreddits (Polars) ******")
    print("- Number of subreddits:", len(subrs))
    print(f"- Time range of filtering: {min(years)}-{max(years)}")

    try: 
        cols_to_keep = cols_to_keep_map[filetype]
    except Exception as e:
        print("Invalid filetype!", e)

    if not raw_data_dir:
        with open("src/configs.yml", "r") as fh:
            raw_data_dir = yaml.safe_load(fh)["RAW_DATA"]
            
    all_year_dfs: List[pl.DataFrame] = []
    for this_year in years:
        print("\tProcessing year:", this_year)
        if this_year == 2005:
            months = range(6,13)
        else:
            months = range(1,13)
        year_dfs: List[pl.DataFrame] = []

        for this_month in months:
            print("\t\tProcessing month:", this_month)
            data_dir = f"{raw_data_dir}/{filetype}/year={this_year}/month={this_month}/"
            parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))

            month_dfs: List[pl.DataFrame] = []
            for i,this_file in enumerate(parquet_files):
                print(f"\t\t\tfile {i}, progress = {i/len(parquet_files)}")
                df = pl.read_parquet(this_file)
                if filetype == "comments" or this_year in [2015, 2016, 2017, 2018, 2019, 2020, 2022]:
                    df = df.with_columns(
                        pl.col("subreddit")
                        .cast(pl.Utf8)
                        .str.to_lowercase())
                elif this_year in [2005, 2006, 2007, 2008, 2009,
                                   2010, 2011, 2012, 2013, 2014, 2021]:
                    df = df.with_columns(
                        pl.col("permalink").map_elements(
                            extract_subreddit_from_permalink,
                            return_dtype=pl.Utf8
                        ).alias("subreddit"))
                
                df = df.select(cols_to_keep)
                df = df.filter(pl.col("subreddit").is_in(subrs))
                if filetype == "submissions":
                    df = df.with_columns([
                        pl.col("id").cast(pl.Utf8),
                        pl.col("created_utc").cast(pl.Datetime("ms")),
                        # pl.col("score").cast(pl.Int64),
                        # pl.col("num_comments").cast(pl.Int64),
                        # pl.col("is_self").cast(pl.Boolean)
                    ])
                elif filetype == "comments":
                    df = df.with_columns([
                        pl.col("id").cast(pl.Utf8),
                        pl.col("created_utc").cast(pl.Datetime("ms"))
                    ])
                if df.height > 0:
                    month_dfs.append(df)
            
            if month_dfs:
                df_month = pl.concat(month_dfs, how="vertical", rechunk=True)
                year_dfs.append(df_month)
        if year_dfs:
            df_year = pl.concat(year_dfs, how="vertical", rechunk=True)
        else:
            df_year = pl.DataFrame()
        
        if df_year.height > 0:
            all_year_dfs.append(df_year)

        if len(output_fpath) > 0:
            if df_year.height > 0:
                out_path = os.path.join(
                    output_fpath, 
                    f"{filetype}_filtered_by_subreddits_{this_year}.parquet")
                print("\tSaving year:", this_year)
                df_year.write_parquet(out_path, compression="snappy")
                print("\tSaved year:", this_year)
                print("\tNumber of rows:", df_year.height)
                print(df_year.head())
            else:
                print(f"Year {this_year} returns empty results!")
            print("\n")

    if all_year_dfs:
        df_all = pl.concat(all_year_dfs, how="vertical", rechunk=True)
    else:
        df_all = pl.DataFrame()

    return df_all


def filter_raw_data_by_user(
        users: List[str],
        years: List[int],
        filetype:str = "submissions",
        output_fpath: str = "", 
        raw_data_dir: str = "") -> pl.DataFrame:
    """Filter Reddit submission data by specified users and years (Polars version),
    and save yearly results to parquet files.
    raw_data_dir defaults to RAW_DATA from src/configs.yml.

    Args
    ----
    users : list[str]
        List of usernames to retain in the filtered data.
    years : list[int]
        List of years (as integers) to process. 
    output_fpath : str, optional (default = "")
        Output directory path where filtered yearly parquet files will be saved.
        Each file will be named 'submissions_filtered_by_subreddit_users_{year}.parquet'.
    """

    print(f"****** Filtering {filetype.capitalize()} by Users (Polars) ******")
    print("- Number of users:", len(users))
    print(f"- Time range of filtering: {min(years)}-{max(years)}")

    try: 
        cols_to_keep = cols_to_keep_map[filetype]
    except Exception as e:
        print("Invalid filetype!", e)

    all_year_dfs: List[pl.DataFrame] = []

    if not raw_data_dir:
        with open("src/configs.yml", "r") as fh:
            raw_data_dir = yaml.safe_load(fh)["RAW_DATA"]
            
    for this_year in years:
        print("\tProcessing year:", this_year)
        if this_year == 2005:
            months = range(6, 13)
        else:
            months = range(1, 13)
        year_dfs: List[pl.DataFrame] = []

        for this_month in months:
            print("\t\tProcessing month:", this_month)
            data_dir = f"{raw_data_dir}/{filetype}/year={this_year}/month={this_month}/"
            parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))

            month_dfs: List[pl.DataFrame] = []
            for i,this_file in enumerate(parquet_files):
                print(f"\t\t\tfile {i}, progress = {i/len(parquet_files)}")
                df = pl.read_parquet(this_file)
                if filetype == "comments" or this_year in [2015, 2016, 2017, 2018, 2019, 2020, 2022]:
                    df = df.with_columns(
                        pl.col("subreddit")
                        .cast(pl.Utf8)
                        .str.to_lowercase())
                elif this_year in [2005, 2006, 2007, 2008, 2009,
                                   2010, 2011, 2012, 2013, 2014, 2021]:
                    df = df.with_columns(
                        pl.col("permalink").map_elements(
                            extract_subreddit_from_permalink,
                            return_dtype=pl.Utf8
                        ).alias("subreddit"))
                    
                df = df.select(cols_to_keep)
                df = df.filter(pl.col("author").is_in(users))
                if filetype == "submissions":
                    df = df.with_columns([
                        pl.col("id").cast(pl.Utf8),
                        pl.col("created_utc").cast(pl.Datetime("ms")),
                        # pl.col("score").cast(pl.Int64),
                        # pl.col("num_comments").cast(pl.Int64),
                        # pl.col("is_self").cast(pl.Boolean),
                    ])
                elif filetype == "comments":
                    df = df.with_columns([
                        pl.col("id").cast(pl.Utf8),
                        pl.col("created_utc").cast(pl.Datetime("ms"))
                    ])
                if df.height > 0:
                    month_dfs.append(df)
            
            if month_dfs:
                df_month = pl.concat(month_dfs, how="vertical", rechunk=True)
                # df_month = df_month.with_columns(pl.col("id").cast(pl.Utf8))
                year_dfs.append(df_month)
        if year_dfs:
            df_year = pl.concat(year_dfs, how="vertical", rechunk=True)
        else:
            df_year = pl.DataFrame()
        
        if df_year.height > 0:
            all_year_dfs.append(df_year)

        if len(output_fpath) > 0:
            if df_year.height > 0:
                out_path = os.path.join(
                    output_fpath, 
                    f"{filetype}_filtered_by_subreddit_users_{this_year}.parquet"
                )
                print("\tSaving year:", this_year)
                df_year.write_parquet(out_path, compression="snappy")
                print("\tSaved year:", this_year)
                print("\tNumber of rows:", df_year.height)
                print(df_year.head())
            else:
                print(f"Year {this_year} returns empty results!")
            print("\n")
            # del df_year

    if all_year_dfs:
        df_all = pl.concat(all_year_dfs, how="vertical", rechunk=True)
    else:
        df_all = pl.DataFrame()

    return df_all


def filter_raw_data_by_subm_id(
        subm_ids_lf: pl.LazyFrame,
        years: List[int],
        filetype: str = "submissions",
        output_fpath: str = "",
        raw_data_dir: str = "") -> None:
    """Filter raw Reddit data by submission IDs and save yearly results.
    raw_data_dir defaults to RAW_DATA from src/configs.yml.

    For submissions, filters on `id`.
    For comments, filters on `link_id`, after removing the `t3_` prefix.
    """

    print(f"****** Filtering {filetype.capitalize()} by Submission IDs (Polars) ******")
    print(f"- Time range of filtering: {min(years)}-{max(years)}")

    if filetype not in cols_to_keep_map:
        raise ValueError(
            f"Invalid filetype: {filetype!r}. "
            "Expected 'submissions' or 'comments'."
        )
        
    if output_fpath:
        os.makedirs(output_fpath, exist_ok=True)

    cols_to_keep = cols_to_keep_map[filetype]
    
    if not raw_data_dir:
        with open("src/configs.yml", "r") as fh:
            raw_data_dir = yaml.safe_load(fh)["RAW_DATA"]

    for this_year in years:
        print("\tProcessing year:", this_year)

        if this_year == 2005:
            months = range(6, 13)
        else:
            months = range(1, 13)

        year_lfs: List[pl.LazyFrame] = []

        for this_month in months:
            print("\t\tProcessing month:", this_month)

            data_dir = (f"{raw_data_dir}/{filetype}/year={this_year}/month={this_month}/")
            parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))

            if not parquet_files: 
                print("\t\t\tNo parquet files found; skipping.") 
                continue

            for i, this_file in enumerate(parquet_files):
                print(f"\t\t\tfile {i}, progress = {i / len(parquet_files)}")

                df = pl.scan_parquet(this_file)

                if filetype == "comments" or this_year in [2015, 2016, 2017, 2018, 2019, 2020, 2022]:
                    df = df.with_columns(pl.col("subreddit").cast(pl.Utf8).str.to_lowercase())

                elif this_year in [2005, 2006, 2007, 2008, 2009,
                                   2010, 2011, 2012, 2013, 2014, 2021]:
                    df = df.with_columns(
                        pl.col("permalink").map_elements(
                            extract_subreddit_from_permalink,
                            return_dtype=pl.Utf8
                        ).alias("subreddit"))
                    
                df = df.select(cols_to_keep).with_columns(
                    pl.col("id").cast(pl.String),
                    pl.col("created_utc").cast(pl.Datetime("ms")),
                )
                if filetype == "submissions":
                    df = df.with_columns(pl.col("id").alias("_subm_id"))
                else:
                    df = df.with_columns(pl.col("link_id").cast(pl.String).str.strip_prefix("t3_").alias("_subm_id"))

                year_lfs.append(df)

        if not year_lfs:
            print(f"Year {this_year} has no input files.")
            continue

        df_year = pl.concat(year_lfs, how="vertical").join(subm_ids_lf, on="_subm_id", how="semi").drop("_subm_id")

        if output_fpath:
            out_path = os.path.join(output_fpath, f"{filetype}_filtered_by_submission_ids_{this_year}.parquet")

            print("\tSaving year:", this_year)
            df_year.sink_parquet(out_path, compression="snappy", engine="streaming")
            print("\tSaved year:", out_path)
            del df_year

    # if all_year_dfs:
    #     df_all = pl.concat(all_year_dfs, how="vertical", rechunk=True)
    # else:
    #     df_all = pl.DataFrame()

    # return df_all

def filter_prefiltered_data_by_subm_id(
        subm_ids_lf: pl.LazyFrame,
        prefiltered_path: str, 
        years: List[int],
        filetype: str = "submissions",
        output_fpath: str = "") -> None:
    """Filter yearly prefiltered Reddit data by submission ID and save it."""
    
    if filetype not in {"submissions", "comments"}:
        raise ValueError(
            f"Unsupported filetype {filetype!r}; "
            "expected 'submissions' or 'comments'."
        )

    if output_fpath:
        os.makedirs(output_fpath, exist_ok=True)
    
    for this_year in years:
        print("\tProcessing year:", this_year)
        df = load_filtered_data(
            years=[this_year], 
            data_dir=prefiltered_path, 
            filetype=filetype,
            filter_bots=False, 
            lazy=True,  # return LazyFrame to avoid materializing the yearly data
        )

        if filetype == "submissions":
            df_filtered = (df.with_columns(pl.col("id").cast(pl.String).alias("_subm_id"))
                            .join(subm_ids_lf, on="_subm_id", how="semi")
                            .drop("_subm_id"))

        else:
            df_filtered = (df.with_columns(pl.col("link_id").cast(pl.String).str.strip_prefix("t3_").alias("_subm_id"))
                            .join(subm_ids_lf, on="_subm_id", how="semi")
                            .drop("_subm_id"))

        out_path = os.path.join(output_fpath, f"{filetype}_filtered_by_submission_ids_{this_year}.parquet")
        df_filtered.sink_parquet(out_path, compression="snappy", engine="streaming")

        print("\tSaved:", out_path)