#!/usr/bin/env python
"""Aggregate submissions + comments into one text blob per submission,
clean for moderation/toxicity scoring, and write a parquet file.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import polars as pl
from tqdm.auto import tqdm

# make `src` importable when run from anywhere inside the repo;
# adjust ROOT if you keep the script elsewhere
# ROOT = Path(__file__).resolve()
# for parent in ROOT.parents:
#     if (parent / "src").is_dir():
#         sys.path.insert(0, str(parent))
#         break

from src.utils.data_loader_polars import load_filtered_data  # noqa: E402

def parse_years(spec: str) -> list[int]:
    """'2005:2019' -> inclusive range; '2018,2019' -> list."""
    if ":" in spec:
        start, end = spec.split(":", 1)
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def clean_text_expr(colname: str) -> pl.Expr:
    return (
        pl.col(colname)
        .fill_null("")
        .cast(pl.Utf8)
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def aggregate_text_by_submission(subm: pl.DataFrame, comm: pl.DataFrame) -> pl.DataFrame:
    print("cleaning comments...")
    comm_by_subm = (
        comm
        .select(["id", "link_id", "parent_id", "created_utc", "body"])
        .with_columns([
            pl.col("link_id").str.replace(r"^t3_", "").alias("subm_id"),
            clean_text_expr("body").alias("body_clean"),
            pl.col("parent_id").str.starts_with("t3_").alias("is_top_level"),
        ])
        .filter(
            (pl.col("body_clean") != "") &
            (pl.col("body_clean") != "[deleted]") &
            (pl.col("body_clean") != "[removed]")
        )
        .sort(
            ["subm_id", "is_top_level", "created_utc"],
            descending=[False, True, False],  # top-level comments first
        )
        .group_by("subm_id", maintain_order=True)
        .agg(pl.col("body_clean").str.join("\n").alias("comments_text"))
    )

    print("cleaning submissions...")
    subm_text = (
        subm
        .with_columns([
            clean_text_expr("title").alias("title_clean"),
            clean_text_expr("selftext").alias("selftext_clean"),
        ])
        .with_columns([
            pl.when(pl.col("selftext_clean").is_in(["[deleted]", "[removed]"]))
            .then(pl.lit(""))
            .otherwise(pl.col("selftext_clean"))
            .alias("selftext_clean")
        ])
    )

    print("merging submissions and comments...")
    subm_comm = (
        subm_text
        .join(comm_by_subm, left_on="id", right_on="subm_id", how="left")
        .with_columns(pl.col("comments_text").fill_null(""))
        .with_columns(
            pl.concat_str(
                [
                    pl.col("title_clean"),
                    pl.lit("\n"),
                    pl.col("selftext_clean"),
                    pl.lit("\n"),
                    pl.col("comments_text"),
                ],
                separator="",
            ).alias("text")
        )
        .select(["id", "text"])
    )
    return subm_comm


def strip_urls(text: str) -> str:
    return re.sub(r"https?://\S+", "[URL]", text)


def strip_reddit_markdown(text: str) -> str:
    text = re.sub(r"^>.*$", "", text, flags=re.MULTILINE)        # blockquotes
    text = re.sub(r"~~(.+?)~~", r"\1", text)                     # strikethrough
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)                 # bold
    text = re.sub(r"\*(.+?)\*", r"\1", text)                     # italic
    text = re.sub(r"__(.+?)__", r"\1", text)                     # bold alt
    text = re.sub(r"_(.+?)_", r"\1", text)                       # italic alt
    text = re.sub(r"`(.+?)`", r"\1", text)                       # inline code
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)  # bullets
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE) # numbered lists
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)        # links -> label
    return text


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clean_for_moderation(text: str) -> str:
    return normalize_whitespace(strip_reddit_markdown(strip_urls(text)))


# ------------------------------------------------------------ subr filter

def load_subreddit_list(path: str, if_banned: Optional[int], subr_col: str) -> list[str]:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        if if_banned is not None:
            if "if_banned" not in df.columns:
                sys.exit(f"--if-banned given but no 'if_banned' column in {p}")
            df = df[df["if_banned"] == if_banned]
        return df[subr_col].dropna().tolist()
    # plain text: one subreddit per line
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description="Aggregate submissions+comments per submission and clean for toxicity scoring."
    )
    ap.add_argument("--subm-dir", required=True, help="folder with yearly submission files")
    ap.add_argument("--comm-dir", required=True, help="folder with yearly comment files")
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--input-years",
                   help="Only read parquet files whose filename year is in this range, e.g. 2005:2019.")
    ap.add_argument("--subr-file", default=None,
                    help="optional subreddit filter: .csv (uses --subr-col) or .txt (one per line)")
    ap.add_argument("--subr-col", default="subreddit", help="column name in --subr-file csv")
    ap.add_argument("--if-banned", type=int, choices=[0, 1], default=None,
                    help="if --subr-file is a match table, keep only rows with this if_banned value")
    ap.add_argument("--no-clean", action="store_true",
                    help="skip the (slow) markdown/URL cleaning step")
    ap.add_argument("--overwrite", action="store_true", help="overwrite --out if it exists")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        sys.exit(f"{out_path} exists; pass --overwrite to replace it.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    years = parse_years(args.input_years)

    subrs = None
    if args.subr_file:
        subrs = load_subreddit_list(args.subr_file, args.if_banned, args.subr_col)
        print(f"filtering to {len(subrs)} subreddits from {args.subr_file}")

    print("loading years:", years)
    print(f"loading submissions...")
    subm = load_filtered_data(years=years, data_dir=args.subm_dir,
                              filetype="submissions", filter_bots=False)
    print(f"loading comments...")
    comm = load_filtered_data(years=years, data_dir=args.comm_dir,
                              filetype="comments", filter_bots=False)

    if subrs is not None:
        subm = subm.filter(pl.col("subreddit").is_in(subrs))
        comm = comm.filter(pl.col("subreddit").is_in(subrs))

    print(f"submissions: {subm.height:,} rows | comments: {comm.height:,} rows")

    subm_comm = aggregate_text_by_submission(subm, comm)
    del subm, comm
    print(f"aggregated: {subm_comm.height:,} submissions")

    if not args.no_clean:
        texts = subm_comm["text"].to_list()
        cleaned = [clean_for_moderation(t) for t in tqdm(texts, desc="Cleaning text")]
        subm_comm = subm_comm.with_columns(pl.Series("text", cleaned))

    subm_comm.write_parquet(out_path, compression="snappy")
    print(f"wrote {out_path}")

if __name__ == "__main__":
    main()


"""

Examples
--------
# banned subreddits, filtered to the 3274 in a match file
python aggr_subm_for_toxicity_scoring.py \
    --subm-dir $DATA/filtered-data/submissions/subm-from-subr-banned-2020/ \
    --comm-dir $DATA/filtered-data/comments/comm-from-subr-banned-2020/ \
    --subr-file $DATA/data/samples/cem-batch4-new/matched_table_banned_3274_cem_X10N1_lifetime_d.csv \
    --if-banned 1 \
    --input-years 2005:2019 \
    --out $DATA/filtered-data/subm_comm_clean/subm-comm-from-subr-banned-2020-aggr-by-subm-2006-2019.parquet

# matched folder, no extra subreddit filter (folder already subset)
python aggr_subm_for_toxicity_scoring.py \
    --subm-dir $DATA/filtered-data/submissions/subm-from-subr-banned-2020-subset3274-match-lifetime_d-X10N1/ \
    --comm-dir $DATA/filtered-data/comments/comm-from-subr-banned-2020-subset3274-match-lifetime_d-X10N1/ \
    --input-years 2005:2019 \
    --out $DATA/filtered-data/subm_comm_clean/subm-comm-...-match-lifetime_d-X10N1-aggr-by-subm-2006-2019.parquet
"""