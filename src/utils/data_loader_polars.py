import os
from typing import List, Union
import polars as pl
from pathlib import Path

def load_filtered_data(
        years:List,
        data_dir:str,
        filetype:str = "submissions",
        filter_bots:bool = False,
        bots_dir:str = "",
        lazy: bool = False,
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Load filtered Reddit data for specified years.

    Parameters
    ----------
    years
        Years to load.
    data_dir
        Directory containing yearly Parquet files. The path must include
        either "from-user" or "from-subr".
    filetype
        Either "submissions" or "comments".
    filter_bots
        Whether to remove known bot accounts.
    bots_dir
        Directory containing yearly bot-author CSV files. Defaults to
        `data_dir`.
    lazy
        If True, return a LazyFrame using scan_parquet. Otherwise, return
        a collected DataFrame.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Concatenated data for the requested years.
    """

    if filetype not in {"submissions", "comments"}:
        raise ValueError(
            f"Unsupported filetype {filetype!r}; "
            "expected 'submissions' or 'comments'."
        )

    print(f"******************\nLoading {filetype.capitalize()} from {data_dir}...")

    data_dir = Path(data_dir)
    data_dir_str = str(data_dir)
    dfs = []

    for this_year in years:
        print("\tloading:", this_year)
        if "from-user" in data_dir_str:
            fpath = data_dir / f"{filetype}_filtered_by_subreddit_users_{this_year}.parquet"
        elif "from-subr" in data_dir_str:
            fpath = data_dir / f"{filetype}_filtered_by_subreddits_{this_year}.parquet"
        else:
            raise ValueError("data_dir must include either 'from-user' or 'from-subr'.")

        if not fpath.exists():
            print(f"\t[skip] file does not exist: {fpath}")
            continue

        if lazy:
            df = pl.scan_parquet(fpath)
        else:
            df = pl.read_parquet(fpath)
        
        df = df.with_columns(pl.col("created_utc").cast(pl.Datetime("ms")))
        dfs.append(df)

    if not dfs:
        return pl.LazyFrame() if lazy else pl.DataFrame()
    
    if lazy:
        df_all = pl.concat(dfs, how="vertical")
    else:
        df_all = pl.concat(dfs, how="vertical", rechunk=True)

    if filter_bots:
        print("\tfilter out bots...")
        if not bots_dir:
            bots_dir = str(data_dir)

        bot_dfs = []
        for this_year in years:
            bots_path = Path(bots_dir) / f"bots_author_{this_year}.csv"
            if os.path.exists(bots_path):
                try:
                    if lazy:
                        bots_df = pl.scan_csv(bots_path).select(pl.col("author").cast(pl.String))
                    else:
                        bots_df = pl.read_csv(bots_path).select(pl.col("author").cast(pl.String))
                    bot_dfs.append(bots_df)
                except Exception as e:
                    print(f"Failed to load bots dataframe for {this_year}: {e}")
            else:
                print(f"No bot file found for {this_year} at {bots_path}, skipping.")

        if bot_dfs:
            bot_authors = (pl.concat(bot_dfs, how="vertical").select("author").unique())
            df_all = df_all.with_columns(pl.col("author").cast(pl.String)).join(bot_authors, on="author", how="anti")
        else:
            print("No bot authors loaded; skipping bot filtering.")
    
    return df_all

    
