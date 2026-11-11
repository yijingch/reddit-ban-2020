"""Count shared users from author-subreddit-year activity summaries."""

import argparse
from pathlib import Path

import pandas as pd
import polars as pl
import yaml


COHORT_YEARS = tuple(range(2005, 2020))
ACTIVITY_YEARS = tuple(range(2005, 2020))
THRESHOLDS = (1, 2, 3, 5, 10, 20, 30, 50, 100)
# OUTPUT_NAME = "n_overlap_user_post_w_banned-2020-3274_thresholded-new-verify.parquet"
OUTPUT_NAME = "n_overlap_user_post_w_banned-2020-3274_thresholded.parquet"

def load_author_counts(count_dir, years=COHORT_YEARS):
    """Read both record types without loading unused date columns."""
    frames = []
    for year in years:
        for filetype, column in (("submissions", "n_subm"), ("comments", "n_comm")):
            path = Path(count_dir) / f"author_subreddit_count_{filetype}_minmax_utc_{year}.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"Missing author-year summary: {path}. Run step 1b first.")
            frames.append(
                pl.scan_parquet(path).select(
                    pl.col("subreddit").cast(pl.String),
                    pl.col("author").cast(pl.String),
                    pl.col(column).cast(pl.UInt64).alias("n_activity"),
                    pl.lit(year, dtype=pl.Int32).alias("year"),
                )
            )
    return pl.concat(frames)


def overlap_from_counts(counts, banned_subrs, activity_years=ACTIVITY_YEARS):
    """Rebuild the B-user cohort, then sum activity and count distinct shared users.

    Cohort selection includes 2005, matching the historical extraction window;
    overlap activity itself starts in 2006, matching the original script.
    """
    counts = counts.filter(
        pl.col("author").is_not_null()
        & (pl.col("author") != "[deleted]")
        & (~pl.col("subreddit").str.starts_with("u_"))
        & (pl.col("n_activity") > 0)
    )
    banned_subrs = list(banned_subrs)
    cohort = counts.filter(pl.col("subreddit").is_in(banned_subrs)).select("author").unique()
    history = counts.filter(pl.col("year").is_in(list(activity_years))).join(
        cohort, on="author", how="semi"
    )
    banned_activity = (
        history.filter(pl.col("subreddit").is_in(banned_subrs))
        .group_by("author")
        .agg(pl.col("n_activity").sum().alias("n_activity_in_banned_subrs"))
    )
    return (
        history.select("subreddit", "author").unique()
        .join(banned_activity, on="author", how="left")
        .with_columns(pl.col("n_activity_in_banned_subrs").fill_null(0))
        .group_by("subreddit")
        .agg(*[
            (pl.col("n_activity_in_banned_subrs") >= threshold)
            .sum().alias(f"u_overlap_min_post_ban_{threshold}")
            for threshold in THRESHOLDS
        ])
        .sort("u_overlap_min_post_ban_5", descending=True)
        .collect(engine="streaming")
    )


def load_banned_subreddits(data_dir):
    """Preserve the original status filter and first-submission birth screen."""
    banned_df = pd.read_csv(Path(data_dir) / "index/banned_subreddits_cleaned_251006_add_filter.csv")
    banned_df = banned_df[banned_df["active_filter"] == 1]
    count_dir = Path(data_dir) / "data/counts-raw"
    frames = []
    for year in range(2005, 2021):
        frame = pd.read_parquet(count_dir / f"count_submissions_n_item_n_author_minmax_utc_{year}.parquet")
        frames.append(frame[["subreddit", "min_date"]])
    counts = pd.concat(frames, ignore_index=True)
    counts["min_date"] = pd.to_datetime(counts["min_date"])
    counts = counts[counts["subreddit"].str.len() > 0]
    first_dates = counts.groupby("subreddit")["min_date"].min()
    born2020 = first_dates[first_dates > pd.Timestamp("2019-12-31")].index
    return set(banned_df["subreddit"]) - set(born2020)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/configs.yml")
    args = parser.parse_args()

    with open(args.config, "r") as fh:
        configs = yaml.safe_load(fh)
    data_dir = Path(configs["SERVER_DATA"])
    banned_subrs = load_banned_subreddits(data_dir)
    print("final list of banned subreddits:", len(banned_subrs))
    # final list of banned subreddits: 3274
    counts = load_author_counts(data_dir / "data/counts-raw")
    result = overlap_from_counts(counts, banned_subrs)

    output = data_dir / "data/counts-raw" / OUTPUT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output, compression="snappy")
    print("saved to:", output)


if __name__ == "__main__":
    main()
