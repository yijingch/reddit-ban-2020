"""
Score toxicity (Detoxify) over cleaned submission/comment text.

Input can be either:
  - a single .parquet file, or
  - a directory containing many .parquet files (part_0000.parquet, ...).

Optionally join against a filtered submissions dir (to restrict to a matched
subset and/or attach created_utc for per-week sampling).

 * Caching behavior

 | Behavior                              | Single file, no --per-file               | --per-file                                 |
 |---------------------------------------|------------------------------------------|--------------------------------------------|
 | Input loading                         | Loads the single parquet file            | Loads one parquet file at a time           |
 | Incremental shard caching             | Yes                                      | Yes                                        |
 | Resume already scored IDs             | Yes                                      | Yes                                        |
 | Final output                          | One consolidated parquet                 | One consolidated parquet per input file    |
 | Completion marker                     | No                                       | Yes, one .done file per completed file     |
 | Skip completed work before loading    | No                                       | Yes, when cache and marker both exist      |
 | Interrupted-file recovery             | Resumes from cache and shards            | Resumes from cache and shards              |
 | Completed rerun                       | Reloads input, then finds 0 rows left    | Skips the file before loading              |
 | Output naming                         | Uses the specified output parquet path   | Adds -toxicity.parquet to each input stem  |

"""

from __future__ import annotations
from pathlib import Path

import argparse
import glob
import os

import polars as pl
import yaml

from src.utils.text_helper.toxicity_scoring_detoxify import score_toxicity_detoxify
from src.utils.data_loader_polars import load_filtered_data
from src.utils.argparse_helper import parse_years, fill_path

# ---------------------------------------------------------------- helpers --

def collect_parquet_files(path: str) -> list[str]:
    """Resolve a single parquet file or a directory of parquet files."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
        if not files:
            raise ValueError(f"No .parquet files found in directory: {path}")
        return files
    if os.path.isfile(path) and path.lower().endswith(".parquet"):
        return [path]
    raise ValueError(f"--input must be a .parquet file or a directory of them: {path}")


def sample_submissions_per_subreddit_week(
    df: pl.DataFrame,
    window_length: str = "1w",
    n: int = 10,
    seed: int = 42,
) -> pl.DataFrame:
    """Randomly keep up to n rows per (subreddit, week)."""
    return (
        df.with_columns(
            pl.col("created_utc").cast(pl.Date).dt.truncate(window_length).alias("week")
        )
        .with_columns(
            pl.int_range(pl.len(), dtype=pl.UInt32)
            .shuffle(seed=seed)
            .over(["subreddit", "week"])
            .alias("_rank")
        )
        .filter(pl.col("_rank") < n)
        .drop(["_rank", "week"])
    )


def derive_cache_path(output: str, input_file: str, per_file: bool) -> str:
    """Single mode: output is the cache file. Per-file mode: output is a dir."""
    if per_file:
        stem = os.path.splitext(os.path.basename(input_file))[0]
        return os.path.join(output, f"{stem}-toxicity.parquet")
    if output.lower().endswith(".parquet"):
        return output
    stem = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(output, f"{stem}-toxicity.parquet")


# completion markers distinguish fully completed files from partial caches
def completion_marker_path(cache_path: str) -> str:
    """Return the marker path written only after scoring finishes successfully."""
    return f"{cache_path}.done"


def write_completion_marker(marker_path: str) -> None:
    """Atomically write a small completion marker."""
    marker = Path(marker_path)
    tmp_marker = marker.with_suffix(marker.suffix + ".tmp")
    tmp_marker.write_text("completed\n")
    tmp_marker.replace(marker)


# verify that every input ID has a real toxicity result
def count_unscored_ids(df: pl.DataFrame, cache_path: str) -> int:
    """Count unique input IDs without a non-null toxicity score in the cache."""
    cache = Path(cache_path)
    if not cache.exists():
        return df.select("id").unique().height

    scored = (
        pl.scan_parquet(cache)
        .filter(pl.col("toxicity").is_not_null())
        .select("id")
        .unique()
        .collect()
    )

    return (
        df.select("id")
        .unique()
        .join(scored, on="id", how="anti")
        .height
    )

# ------------------------------------------------------------------- main --

def main() -> None:
    p = argparse.ArgumentParser(description="Score toxicity with Detoxify over cleaned text parquet(s).")
    p.add_argument("--input", required=True, help="A single .parquet file OR a directory of .parquet files. Supports {DATAPATH},{ROOTPATH},{covs},{X},{N}.")
    p.add_argument("--output", required=True, help="Cache parquet path (single mode) or output dir (per-file mode). Same placeholders as --input.")
    p.add_argument("--per-file", action="store_true", help="Process each input parquet separately, writing one cache per file (memory-friendly for big dirs).")

    # optional join with submissions metadata (subset filter + created_utc for sampling)
    p.add_argument("--subm-dir", help="Optional filtered submissions dir; used to attach subreddit and created_utc for per-week sampling.")
    p.add_argument("--subm-years", help="Years to read from --subm-dir, e.g. 2006:2019.")

    # optional per-week sampling
    p.add_argument("--sample-per-week", type=int, default=None, help="If set, keep up to N rows per (subreddit, week). Requires subreddit+created_utc (via input or --subm-dir).")
    p.add_argument("--window", default="1w", help="Truncation window for sampling (default: 1w).")
    p.add_argument("--seed", type=int, default=42)

    # scoring params
    p.add_argument("--model", default="multilingual", help="Detoxify model name (default: multilingual).")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-chars", type=int, default=10_000)
    p.add_argument("--flush-every", type=int, default=20, help="Flush cache every N batches.")
    p.add_argument("--device", default=None, help="e.g. cuda, cuda:1, cpu. Default: auto.")

    # misc
    p.add_argument("--head", type=int, default=None, help="Only score the first N rows (for testing).")
    p.add_argument("--covs")
    p.add_argument("--X")
    p.add_argument("--N")
    p.add_argument("--config", default="src/configs.yml")
    args = p.parse_args()

    # --subm-dir is only meaningful in the sampling pipeline
    if args.subm_dir and args.sample_per_week is None:
        p.error("--subm-dir should only be used together with --sample-per-week.")
    if args.sample_per_week is not None and args.sample_per_week <= 0:
        p.error("--sample-per-week must be greater than 0.")
    if args.sample_per_week is not None and args.subm_dir and not args.subm_years:
        p.error("--subm-years is required when --subm-dir is used for sampling.")

    with open(args.config) as fh:
        configs = yaml.safe_load(fh)
    subs = {
        "DATAPATH": configs["SERVER_DATA"],
        "ROOTPATH": configs["SERVER_WD"],
        "covs": args.covs,
        "X": args.X,
        "N": args.N,
    }
    input_path = fill_path(args.input, subs)
    output_path = fill_path(args.output, subs)
    subm_dir = fill_path(args.subm_dir, subs) if args.subm_dir else None

    files = collect_parquet_files(input_path)
    print(f"input: {input_path} ({len(files)} parquet file(s))")
    print(f"output: {output_path}")
    print(f"model: {args.model} | batch_size: {args.batch_size} | device: {args.device or 'auto'}")

    # load submissions metadata once, if requested
    subm_index = None
    if args.sample_per_week is not None and subm_dir:
        subm_years = parse_years(args.subm_years)
        join_cols = ["id", "created_utc", "subreddit"]
        print(f"loading submissions index from {subm_dir} ({subm_years[0]}-{subm_years[-1]})...")
        subm_index = load_filtered_data(years=subm_years, data_dir=subm_dir, filetype="submissions", filter_bots=False ).select(join_cols)
        print(f"  {subm_index.height:,} submissions loaded")

    # output dir(s)
    out_dir = output_path if (args.per_file or not output_path.lower().endswith(".parquet")) \
        else os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # group files: per-file mode processes one at a time, otherwise all at once
    groups = [[f] for f in files] if args.per_file else [files]

    for i, group in enumerate(groups):
        print(f"------------------ chunk {i + 1}/{len(groups)} ------------------")

        cache_path = derive_cache_path(output_path, group[0], args.per_file)
        marker_path = completion_marker_path(cache_path)  # NEW!
        print(f"cache: {cache_path}")

        # skip only files known to have completed successfully.
        # A cache without a marker is treated as partial and will be resumed
        # by score_toxicity_detoxify().
        if args.per_file and os.path.exists(marker_path):
            if os.path.exists(cache_path):
                print(f"completed marker found; skipping: {marker_path}")
                continue
            print(
                f"warning: marker exists but cache is missing; reprocessing: {cache_path}"
            )
            os.remove(marker_path)

        df = pl.read_parquet(group)
        print(f"loaded {df.height:,} rows from {len(group)} file(s)")

        if args.sample_per_week is not None:
            if subm_index is not None:
                before_join = df.height
                df = df.join(subm_index, on="id", how="inner", validate="m:1")
                print(f"after join with submissions index: {df.height:,}/{before_join:,} rows retained")

            if not {"subreddit", "created_utc"}.issubset(df.columns):
                raise ValueError("Sampling needs 'subreddit' and 'created_utc' columns "
                                 "(provide them in the input or via --subm-dir).")
            
            df = sample_submissions_per_subreddit_week(
                df, window_length=args.window, n=args.sample_per_week, seed=args.seed
            )
            print(f"after sampling ({args.sample_per_week}/subreddit/{args.window}): {df.height:,} rows")

        if args.head is not None:
            df = df.head(args.head)
            print(f"head mode: scoring first {df.height:,} rows only")

        score_toxicity_detoxify(
            df=df,
            cache_path=cache_path,
            model_name=args.model,
            batch_size=args.batch_size,
            max_chars=args.max_chars,
            flush_every_n_batches=args.flush_every,
            device=args.device,
        )

        # write a marker only when every input ID has a real score.
        # Failed batches are stored as nulls by the scoring helper, so they must
        # remain unmarked and be retried on the next run.
        if args.per_file:
            n_unscored = count_unscored_ids(df, cache_path)

            if n_unscored == 0:
                write_completion_marker(marker_path)
                print(f"completion marker written: {marker_path}")
            else:
                print(
                    f"not marking complete: {n_unscored:,} unique IDs "
                    "still have no toxicity score"
                )

        del df


if __name__ == "__main__":
    main()

"""
Examples:

# small test run for sanity check
FIRST_FILE=$(find \
  /cinddata/reddit-ban-study/filtered-data-new/subm_comm_clean/subm-comm-from-subm-by-user-banned-2020-subset3274-neighbors-only-aggr-by-subm-2005-2019/ \
  -maxdepth 1 -type f -name '*.parquet' \
  | sort \
  | head -n 1)

echo "Testing with: $FIRST_FILE"

python src/preprocess/labeling/score_toxicity_detoxify.py \
    --input "$FIRST_FILE" \
    --per-file \
    --batch-size 128 \
    --device cuda:0 \
    --output '{DATAPATH}/output/toxicity/subset3274/test-per-file/'

At batch size 128, the GPU was already compute-saturated. Increasing to 256 or 512 mainly increases the amount of work queued per step and memory usage.

# single cleaned parquet, score every row directly
python src/preprocess/labeling/score_toxicity_detoxify.py \
    --covs lifetime_d --X 10 --N 1 \
    --input '{DATAPATH}/filtered-data/subm_comm_clean/subm-comm-from-subr-banned-2020-subset3274-match-{covs}-X{X}N{N}-aggr-by-subm-2006-2019.parquet' \
    --output '{DATAPATH}/output/toxicity/subset3274/detoxify-multilingual-all-subm-perweek/subm-comm-from-subr-banned-2020-subset3274-match-{covs}-X{X}N{N}-by-subm-toxicity.parquet'

# directory of many big parquets, processed one file at a time (memory-friendly)
python src/preprocess/labeling/score_toxicity_detoxify.py \
    --input '{DATAPATH}/filtered-data-new/subm_comm_clean/subm-comm-from-subm-by-user-banned-2020-subset3274-neighbors-only-aggr-by-subm-2005-2019/' \
    --per-file \
    --batch-size 128 \
    --output '{DATAPATH}/output/toxicity/subset3274/detoxify-multilingual-all-subm-perweek/subm-comm-from-subm-by-user-banned-2020-subset3274-neighbors-only/'

# sampled run: 10 submissions per subreddit-week (needs subreddit + created_utc)
python src/preprocess/labeling/score_toxicity_detoxify.py \
    --input '{DATAPATH}/filtered-data/subm_comm_clean/subm-comm-from-subr-banned-2020-aggr-by-subm-2006-2019.parquet' \
    --subm-dir '{DATAPATH}/filtered-data/submissions/subm-from-subr-banned-2020/' \
    --subm-years 2006:2019 \
    --sample-per-week 10 \
    --output '{DATAPATH}/output/toxicity/detoxify-multilingual-10-subm-perweek/banned-by-subm-toxicity.parquet'

# quick smoke test on 200 rows, on a specific GPU
python src/preprocess/labeling/score_toxicity_detoxify.py \
    --input '{DATAPATH}/filtered-data/subm_comm_clean/subm-comm-from-subr-banned-2020-aggr-by-subm-2006-2019.parquet' \
    --head 200 --device cuda:0 \
    --output '{DATAPATH}/output/toxicity/test-tmp/test.parquet'
"""