#!/usr/bin/env python
"""Aggregate Reddit submissions and comments into one text field per submission.

For each submission, the output combines:

    title + selftext + all linked comments

The script uses a two-pass, memory-bounded workflow:

1. Stream submissions and comments into hash buckets on disk. Rows belonging
   to the same submission are assigned to the same bucket.
2. Load one bucket at a time, aggregate its comments, join them to the
   submissions, and write one output parquet file.

This avoids loading the full Reddit dataset into memory. The output is a
folder containing ``part_*.parquet`` files with two columns: ``id`` and
``text``.

A completed pass 1 is marked with a sentinel file inside the bucket
directory. Re-running with ``--resume`` skips pass 1 when the sentinel is
present and skips any ``part_*.parquet`` files that were already written, so
an interrupted run can continue without redoing the expensive streaming pass.
"""

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Optional
from src.utils.text_helper.clean_text import normalized, clean_markdown
from src.utils.argparse_helper import parse_years, fill_path

import polars as pl
from tqdm.auto import tqdm
import yaml

# Sentinel written into the bucket directory once pass 1 has fully completed.
# Its absence means the buckets are missing or incomplete and must be rebuilt.
BUCKETING_DONE = "_bucketing_done"


# -----------------------------------------------------------------------------
# Input helpers
# -----------------------------------------------------------------------------

def extract_year(filename: str) -> Optional[int]:
    """Extract the data year from a parquet filename.

    Filenames may contain several four-digit numbers (for example a dataset
    label such as ``banned-2020`` alongside the file year), so a bare
    "first match" is unreliable. Preference order:

    1. A year attached to a Pushshift-style prefix, e.g. ``RS_2015`` or
       ``RC-2015``.
    2. Otherwise the *last* year-like number in the filename, since dataset
       labels tend to precede the per-file year.

    Parameters
    ----------
    filename
        Base name of the file, without directories.

    Returns
    -------
    int | None
        The detected year, or ``None`` if the filename contains no year.
    """
    match = re.search(r"(?:RS|RC)[_-]((?:19|20)\d{2})", filename)
    if match:
        return int(match.group(1))

    matches = re.findall(r"(?:19|20)\d{2}", filename)
    return int(matches[-1]) if matches else None


def find_files(folder: str, years: list[int]) -> list[str]:
    """Find parquet files whose filenames contain one of the requested years.

    The function recursively searches ``folder``, detects each file's year
    with :func:`extract_year`, and prints a per-year file count so that a
    wrong year-detection heuristic is visible before a long run starts.

    Parameters
    ----------
    folder
        Root directory containing parquet files.
    years
        Years that should be included.

    Returns
    -------
    list[str]
        Sorted paths to matching parquet files.

    Raises
    ------
    FileNotFoundError
        If no matching parquet files are found.
    """
    requested_years = set(years)
    files: list[str] = []
    per_year: dict[int, int] = {}

    for path in sorted(Path(folder).rglob("*.parquet")):
        year = extract_year(path.name)
        if year in requested_years:
            files.append(str(path))
            per_year[year] = per_year.get(year, 0) + 1

    if not files:
        raise FileNotFoundError(
            f"No parquet files for years {years} found under {folder}"
        )

    counts = ", ".join(f"{y}: {per_year[y]}" for y in sorted(per_year))
    print(f"  {len(files)} files under {folder} ({counts})")

    missing = requested_years - per_year.keys()
    if missing:
        print(f"  WARNING: no files found for years {sorted(missing)}")

    return files


def load_subreddits(path: Optional[str], column: str, if_banned: Optional[int]) -> Optional[list[str]]:
    """Load an optional list of subreddits used to filter the input data.

    A plain-text file should contain one subreddit per line. A CSV file should
    contain the subreddit names in ``column``. When ``if_banned`` is supplied,
    the CSV is additionally filtered using an ``if_banned`` column.

    Parameters
    ----------
    path
        Path to a CSV or plain-text file. ``None`` disables filtering.
    column
        Name of the subreddit column in a CSV file.
    if_banned
        Optional value, either 0 or 1, used to filter a CSV's ``if_banned``
        column.

    Returns
    -------
    list[str] | None
        Subreddit names, or ``None`` when no filter file is supplied.

    Raises
    ------
    ValueError
        If ``if_banned`` is supplied with a plain-text file (which has no
        ``if_banned`` column, so the filter could not be applied), or if the
        required columns are missing from a CSV file.
    """
    if path is None:
        if if_banned is not None:
            raise ValueError("--if-banned requires --subr-file")
        return None

    path_obj = Path(path)

    # Plain-text input: one subreddit name per non-empty line.
    if path_obj.suffix.lower() != ".csv":
        if if_banned is not None:
            raise ValueError(
                "--if-banned only applies to a CSV --subr-file; "
                f"{path_obj} is a plain-text list"
            )
        return [
            line.strip()
            for line in path_obj.read_text().splitlines()
            if line.strip()
        ]

    df = pl.read_csv(path_obj)

    if column not in df.columns:
        raise ValueError(f"Column {column!r} is not present in {path_obj}")

    if if_banned is not None:
        if "if_banned" not in df.columns:
            raise ValueError(
                f"--if-banned was provided, but {path_obj} has no "
                "'if_banned' column"
            )
        df = df.filter(pl.col("if_banned") == if_banned)

    return df.get_column(column).drop_nulls().cast(pl.Utf8).to_list()


# -----------------------------------------------------------------------------
# Pass 1: stream rows into hash buckets
# -----------------------------------------------------------------------------


def sink_buckets(lf: pl.LazyFrame, output: Path, key: str, n_buckets: int) -> None:
    """Partition a lazy dataset into parquet files based on ``hash(key)``.

    Hashing by submission ID ensures that a submission and all its comments
    are written to the same bucket. ``sink_parquet`` performs this operation
    without materializing the full dataset in memory. The bucket number is
    encoded in the directory name only (``include_key=False``), so it is not
    duplicated as a column inside the files.
    """
    bucketed = lf.with_columns(
        (pl.col(key).hash(seed=0) % n_buckets).alias("bucket")
    )

    bucketed.sink_parquet(
        pl.PartitionByKey(base_path=str(output), by="bucket", include_key=False),
        mkdir=True,
        compression="snappy",
    )


def submission_pipeline(files: list[str], subreddits: Optional[list[str]]) -> pl.LazyFrame:
    """Build the lazy cleaning pipeline for submission files.

    Only ``id``, ``title``, and ``selftext`` are retained. Deleted or removed
    selftext is replaced with an empty string before bucketing.
    """
    lf = pl.scan_parquet(files).select("id", "title", "selftext", "subreddit")

    if subreddits is not None:
        lf = lf.filter(pl.col("subreddit").is_in(subreddits))

    return (
        lf.with_columns(
            normalized("title").alias("title"),
            normalized("selftext").alias("selftext"),
        )
        .with_columns(
            pl.when(pl.col("selftext").is_in(["[deleted]", "[removed]"]))
            .then(pl.lit(""))
            .otherwise(pl.col("selftext"))
            .alias("selftext")
        )
        .select("id", "title", "selftext")
    )


def prepare_submissions(files: list[str], output: Path, n_buckets: int, subreddits: Optional[list[str]]) -> None:
    """Clean and partition submission rows by submission ID.

    Each input file is streamed and sunk separately into its own
    ``src_*`` subdirectory, so tqdm can report per-file progress during the
    long streaming pass. :func:`read_bucket` reunites the pieces later by
    globbing across all ``src_*`` directories for a given bucket.
    """
    for i, file in enumerate(
        tqdm(files, desc="Bucketing submissions", unit="file")
    ):
        sink_buckets(
            submission_pipeline([file], subreddits),
            output / f"src_{i:05d}",
            key="id",
            n_buckets=n_buckets,
        )


def comment_pipeline(files: list[str], subreddits: Optional[list[str]]) -> pl.LazyFrame:
    """Build the lazy cleaning pipeline for comment files.

    Reddit's ``link_id`` includes a ``t3_`` prefix, which is removed so it can
    be joined to submission ``id``. ``parent_id`` identifies whether a comment
    is top-level: a parent beginning with ``t3_`` refers directly to the
    submission, while ``t1_`` refers to another comment.
    """
    lf = pl.scan_parquet(files).select(
        "link_id", "parent_id", "created_utc", "body", "subreddit"
    )

    if subreddits is not None:
        lf = lf.filter(pl.col("subreddit").is_in(subreddits))

    return (
        lf.with_columns(
            pl.col("link_id").str.replace(r"^t3_", "").alias("submission_id"),
            pl.col("parent_id").str.starts_with("t3_").alias("top_level"),
            normalized("body").alias("body"),
        )
        # Exclude empty, deleted, and moderator-removed comments.
        .filter(~pl.col("body").is_in(["", "[deleted]", "[removed]"]))
        .select("submission_id", "top_level", "created_utc", "body")
    )


def prepare_comments(files: list[str], output: Path, n_buckets: int, subreddits: Optional[list[str]]) -> None:
    """Clean and partition comment rows by their linked submission ID.

    As with submissions, each input file is streamed and sunk into its own
    ``src_*`` subdirectory so that tqdm can report per-file progress.
    """
    for i, file in enumerate(
        tqdm(files, desc="Bucketing comments", unit="file")
    ):
        sink_buckets(
            comment_pipeline([file], subreddits),
            output / f"src_{i:05d}",
            key="submission_id",
            n_buckets=n_buckets,
        )


# -----------------------------------------------------------------------------
# Pass 2: aggregate one bucket at a time
# -----------------------------------------------------------------------------


def read_bucket(folder: Path, bucket: int) -> Optional[pl.DataFrame]:
    """Read one bucket, returning ``None`` when the bucket has no files.

    Pass 1 writes each input file's rows under its own ``src_*``
    subdirectory, so one logical bucket is the union of that bucket across
    all ``src_*`` directories.
    """
    files = sorted(folder.glob(f"src_*/bucket={bucket}/*.parquet"))
    return pl.read_parquet(files) if files else None


def aggregate(submissions: pl.DataFrame, comments: Optional[pl.DataFrame], clean: bool) -> pl.DataFrame:
    """Combine each submission with all comments assigned to it.

    Comments are ordered with top-level comments first. Within top-level and
    nested-comment groups, comments are ordered chronologically. The resulting
    output has one row per submission and two columns: ``id`` and ``text``.

    Parameters
    ----------
    submissions
        Submission rows from one hash bucket.
    comments
        Comment rows from the same bucket, or ``None`` if no comments exist.
    clean
        Whether to apply Markdown and URL cleaning to the final text.
    """
    if comments is None:
        # An empty frame with the expected join schema handles submissions that
        # have no comments without requiring a separate aggregation branch.
        grouped_comments = pl.DataFrame(
            schema={"submission_id": pl.Utf8, "comments": pl.Utf8}
        )
    else:
        grouped_comments = (
            comments
            .sort(
                ["submission_id", "top_level", "created_utc"],
                descending=[False, True, False],
            )
            .group_by("submission_id", maintain_order=True)
            .agg(pl.col("body").str.join("\n").alias("comments"))
        )

    out = (
        submissions
        .join(
            grouped_comments,
            left_on="id",
            right_on="submission_id",
            how="left",
        )
        .with_columns(pl.col("comments").fill_null(""))
        .with_columns(
            # Preserve a simple title/selftext/comments structure in each text.
            pl.concat_str(
                ["title", "selftext", "comments"],
                separator="\n",
            ).alias("text")
        )
        .select("id", "text")
    )

    if clean:
        out = out.with_columns(clean_markdown(pl.col("text")).alias("text"))

    return out


def write_part(result: pl.DataFrame, part_path: Path) -> None:
    """Write one output part atomically.

    The frame is written to a temporary file and then renamed, so a part file
    either exists in full or not at all. This keeps ``--resume`` safe: a part
    interrupted mid-write is never mistaken for a completed one.
    """
    tmp_path = part_path.with_suffix(part_path.suffix + ".tmp")
    result.write_parquet(tmp_path, compression="snappy")
    os.replace(tmp_path, part_path)


# -----------------------------------------------------------------------------
# Command-line pipeline
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(description=("Aggregate Reddit submission titles, selftext, and comments into one text field per submission."))

    parser.add_argument("--subm-dir", required=True, help="Directory containing yearly submission parquet files.")
    parser.add_argument("--comm-dir", required=True, help="Directory containing yearly comment parquet files.")
    parser.add_argument("--out-dir", required=True, help="Output directory for part_*.parquet files.")
    parser.add_argument("--years", required=True, help="Input years, for example 2005:2019 or 2018,2019.")
    parser.add_argument("--buckets", type=int, default=256, help="Number of hash buckets. Increase this to reduce peak memory.")
    parser.add_argument("--tmp-dir", help="Temporary bucket directory. Defaults to <out-dir>_buckets.")
    parser.add_argument("--resume", action="store_true", help=(
            "Continue an interrupted run: reuse the bucket directory if "
            "pass 1 completed, and skip output parts that already exist. "
            "Must be run with the same --buckets value and Polars version "
            "as the original run."))
    parser.add_argument("--keep-tmp", action="store_true", help="Keep the bucket directory after a successful run.")
    parser.add_argument("--subr-file", help="Optional CSV or text file containing subreddits to retain.")
    parser.add_argument("--subr-col", default="subreddit", help="Subreddit column in --subr-file when it is a CSV.")
    parser.add_argument("--if-banned", type=int, choices=[0, 1], help="Optional filter on an 'if_banned' CSV column.")
    parser.add_argument("--no-clean", action="store_true", help="Do not remove Markdown or replace URLs in the final text.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the output directory if it already exists.")
    parser.add_argument("--covs")
    parser.add_argument("--X")
    parser.add_argument("--N")
    parser.add_argument("--config", default="src/configs.yml")

    return parser


def main() -> None:
    """Parse arguments and run the two-pass aggregation pipeline."""
    args = build_parser().parse_args()

    with open(args.config) as fh:
        configs = yaml.safe_load(fh)
    subs = {
        "DATAPATH": configs["SERVER_DATA"],
        "ROOTPATH": configs["SERVER_WD"],
        "covs": args.covs,
        "X": args.X,
        "N": args.N,
    }

    if args.buckets < 1:
        raise ValueError("--buckets must be at least 1")
    
    subm_dir = fill_path(args.subm_dir, subs)
    comm_dir = fill_path(args.comm_dir, subs)
    out_dir = Path(fill_path(args.out_dir, subs))
    # Derive the default from the directory *name* so a trailing slash in
    # --out-dir does not produce a hidden "<out-dir>/_buckets" sibling.
    tmp_dir = (
        Path(fill_path(args.tmp_dir, subs))
        if args.tmp_dir
        else out_dir.parent / (out_dir.name + "_buckets")
    )
    sentinel = tmp_dir / BUCKETING_DONE
    subr_file = (
        fill_path(args.subr_file, subs)
        if args.subr_file
        else None)

    # Avoid accidentally replacing a completed output dataset. --resume is
    # allowed to write into an existing directory (it skips finished parts).
    if out_dir.exists():
        if args.overwrite:
            shutil.rmtree(out_dir)
        elif not args.resume:
            raise FileExistsError(f"{out_dir} exists; use --overwrite or --resume")

    print(f"Submission input: {subm_dir}")
    print(f"Comment input:    {comm_dir}")
    print(f"Output:           {out_dir}")
    print(f"Temporary data:   {tmp_dir}")

    # Pass 1 can be skipped only if a previous run finished bucketing, which
    # is marked by the sentinel file. A partially written bucket directory
    # (no sentinel) is always rebuilt from scratch.
    run_pass1 = not (args.resume and sentinel.exists())

    years = parse_years(args.years)
    subreddits = load_subreddits(
        subr_file,
        args.subr_col,
        args.if_banned,
    )

    if run_pass1:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        print("Pass 1/2: bucketing submissions")
        prepare_submissions(
            find_files(subm_dir, years),
            tmp_dir / "submissions",
            args.buckets,
            subreddits,
        )

        print("Pass 1/2: bucketing comments")
        prepare_comments(
            find_files(comm_dir, years),
            tmp_dir / "comments",
            args.buckets,
            subreddits,
        )

        sentinel.touch()
    else:
        print(f"Pass 1/2: reusing buckets in {tmp_dir}")

    print("Pass 2/2: aggregating buckets")
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    skipped = 0

    for bucket in tqdm(range(args.buckets), desc="Aggregating"):
        part_path = out_dir / f"part_{bucket:04d}.parquet"

        if args.resume and part_path.exists():
            skipped += 1
            continue

        submissions = read_bucket(tmp_dir / "submissions", bucket)

        # A bucket without submissions cannot contribute output rows.
        if submissions is None:
            continue

        comments = read_bucket(tmp_dir / "comments", bucket)
        result = aggregate(
            submissions,
            comments,
            clean=not args.no_clean,
        )

        write_part(result, part_path)
        total += result.height

    # Bucket files are intermediate data and are no longer needed unless the
    # user wants to rerun pass 2 later (for example with different cleaning).
    if not args.keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if skipped:
        print(f"Skipped {skipped} already-written parts (--resume)")
    print(f"Wrote {total:,} new submissions to {out_dir}")


if __name__ == "__main__":
    main()


"""
Examples:

## for testing: 
python src/preprocess/labeling/aggr_subm_for_toxicity_scoring_bucketed.py \
    --subm-dir "{DATAPATH}/filtered-data-new/submissions/subm-from-subr-banned-2020-subset3274-neighbors-only/" \
    --comm-dir "{DATAPATH}/filtered-data-new/comments/comm-from-subr-banned-2020-subset3274-neighbors-only/" \
    --years 2005 \
    --out-dir "{DATAPATH}/filtered-data-new/subm_comm_clean/test-aggr-2005" \
    --buckets 8 \
    --overwrite

# for neighborhood of banned subreddits
python src/preprocess/labeling/aggr_subm_for_toxicity_scoring_bucketed.py \
    --subm-dir "{DATAPATH}/filtered-data-new/submissions/subm-from-subr-banned-2020-subset3274-neighbors-only/" \
    --comm-dir "{DATAPATH}/filtered-data-new/comments/comm-from-subr-banned-2020-subset3274-neighbors-only/" \
    --years 2005:2019 \
    --out-dir "{DATAPATH}/filtered-data-new/subm_comm_clean/subm-comm-from-subr-banned-2020-subset3274-neighbors-only-aggr-by-subm-2005-2019" \
    --buckets 256
"""