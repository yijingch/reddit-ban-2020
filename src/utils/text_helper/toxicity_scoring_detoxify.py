import gc
import time
import polars as pl
from pathlib import Path
from tqdm.auto import tqdm
import torch
from detoxify import Detoxify
from typing import Optional
import torch.quantization

DETOXIFY_COLS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
]


def _get_shards(cache_path: Path) -> list[Path]:
    return sorted(cache_path.parent.glob(f"{cache_path.stem}_shard_*.parquet"))


def _load_scored_ids(cache_path: Path) -> set:
    dfs = []
    if cache_path.exists():
        dfs.append(pl.read_parquet(cache_path))
    for shard in _get_shards(cache_path):
        dfs.append(pl.read_parquet(shard))
    if not dfs:
        return set()

    df = pl.concat(dfs, how="diagonal_relaxed")

    # only rows with a real toxicity score count as finished
    df_done = df.filter(pl.col("toxicity").is_not_null())
    return set(df_done["id"].to_list())


def _flush(pending: list[dict], cache_path: Path):
    if not pending:
        return
    shard_idx = len(_get_shards(cache_path))
    shard_path = cache_path.parent / f"{cache_path.stem}_shard_{shard_idx:05d}.parquet"
    pl.DataFrame(pending).write_parquet(shard_path)


def consolidate_cache(cache_path: str = "detoxify_cache.parquet") -> pl.DataFrame:
    cache_path = Path(cache_path)
    shards = _get_shards(cache_path)

    dfs = []
    if cache_path.exists():
        dfs.append(pl.read_parquet(cache_path))
    for shard in shards:
        dfs.append(pl.read_parquet(shard))

    if not dfs:
        return pl.DataFrame()

    df = pl.concat(dfs, how="diagonal_relaxed")

    # prefer rows with actual results over placeholders
    df = (
        df.with_columns(
            pl.col("toxicity").is_not_null().cast(pl.Int8).alias("_has_result")
        )
        .sort(["id", "_has_result"])
        .group_by("id", maintain_order=True)
        .tail(1)
        .drop("_has_result")
    )

    # make sure all expected cols exist
    needed = ["id"] + DETOXIFY_COLS
    for c in needed:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).alias(c))

    df = df.select(needed)
    df.write_parquet(cache_path)

    for shard in shards:
        shard.unlink()

    print(f"Consolidated {len(shards)} shards into {cache_path} ({len(df):,} rows total).")
    return df


def _empty_result(id_: str) -> dict:
    return {"id": id_, **{c: None for c in DETOXIFY_COLS}}


def _parse_batch(batch_ids: list[str], preds: dict) -> list[dict]:
    """
    preds is the dict returned by Detoxify.predict(batch_texts)
    e.g. {
        "toxicity": [...],
        "severe_toxicity": [...],
        ...
    }
    """
    out = []

    available_cols = list(preds.keys())
    n = len(batch_ids)

    for i in range(n):
        row = {"id": batch_ids[i]}
        for c in DETOXIFY_COLS:
            if c in available_cols:
                row[c] = preds[c][i]
            else:
                row[c] = None
        out.append(row)

    return out


def score_toxicity_detoxify(
    df: pl.DataFrame,
    cache_path: str = "detoxify_cache.parquet",
    model_name: str = "unbiased",     # good default for English toxicity scoring
    batch_size: int = 128,
    max_chars: int = 2_000,
    flush_every_n_batches: int = 10,
    device: Optional[str] = None,        # None -> auto
    show_device: bool = True,
    num_threads: int = 18,    
    quantization_speedup: bool = False,
) -> None:
    """
    Score toxicity locally using Detoxify.

    Parameters
    ----------
    df                    : must have columns "id" and "text"
    cache_path            : base parquet path; shards written during scoring,
                            consolidated into this file at the end
    model_name            : 'original', 'unbiased', 'multilingual',
                            'original-small', 'unbiased-small'
    batch_size            : texts per forward pass
    max_chars             : truncate texts before sending
    flush_every_n_batches : write a shard every N batches
    device                : 'cuda', 'cpu', 'mps', or None for auto
    """

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if device == "cpu":
        torch.set_num_threads(num_threads)

    if show_device:
        print(f"Loading Detoxify model '{model_name}' on device={device} ...")

    model = Detoxify(model_name, device=device)
    model.model.eval()

    if device == "cuda":
        model.model = model.model.half()
        model.model.eval()

    # INT8 quantization on CPU; ~2-3x faster inference, no quality calibration needed
    if device == "cpu" and quantization_speedup:
        model.model = torch.quantization.quantize_dynamic(
            model.model, {torch.nn.Linear}, dtype=torch.qint8
        )
        model.model.eval()

    cache_path = Path(cache_path)

    scored_ids = _load_scored_ids(cache_path)
    print(f"Resuming: {len(scored_ids):,} already scored, skipping.")
    # CHANGED: normalize once
    if not isinstance(scored_ids, set):
        scored_ids = set(scored_ids)
    scored_ids_list = list(scored_ids)

    # todo = df.filter(~pl.col("id").is_in(scored_ids))
    # n_todo = todo.height
    # instead of building a full `todo = df.filter(...)`
    # Count remaining rows slice-by-slice instead.
    count_slice_size = max(batch_size * 20, 10_000)
    n_todo = 0
    for slice_df in df.iter_slices(count_slice_size):
        n_todo += slice_df.filter(~pl.col("id").is_in(scored_ids_list)).height

    todo = df.filter(~pl.col("id").is_in(list(scored_ids)))
    n_todo = todo.height
    print(f"Remaining: {n_todo:,} rows to score.")

    if n_todo == 0:
        consolidate_cache(str(cache_path))
        return

    pending = []
    kept_batches = 0  # CHANGED: only count non-empty batches

    try:
        with tqdm(total=n_todo, desc=f"Scoring toxicity ({model_name})") as pbar:
            for batch_idx, slice_df in enumerate(df.iter_slices(batch_size)):
                batch = slice_df.filter(~pl.col("id").is_in(scored_ids_list))
                # CHANGED: skip empty batches
                if batch.height == 0:
                    continue

                batch_ids = batch["id"].to_list()
                batch_texts = [((t or "")[:max_chars]) for t in batch["text"].to_list()]

                try:
                    # preds = model.predict(batch_texts)
                    # pending.extend(_parse_batch(batch_ids, preds))
                    preds = model.predict(batch_texts)
                    pending.extend(_parse_batch(batch_ids, preds))

                except RuntimeError as e:
                    # common for CUDA OOM
                    print(f"RuntimeError on batch {batch_idx}: {e}")
                    print("Recording nulls for this batch and continuing.")
                    for id_ in batch_ids:
                        pending.append(_empty_result(id_))

                    if device == "cuda":
                        torch.cuda.empty_cache()
                    elif device == "mps":
                        torch.mps.empty_cache() 

                except Exception as e:
                    print(f"Error on batch {batch_idx}: {e}")
                    print("Recording nulls for this batch and continuing.")
                    for id_ in batch_ids:
                        pending.append(_empty_result(id_))

                # if device == "mps":
                #     torch.mps.synchronize()   # wait for GPU work to actually finish
                pbar.update(len(batch_ids))
                kept_batches += 1

                if kept_batches % flush_every_n_batches == 0:
                    _flush(pending, cache_path)
                    pending = []

                    # optional memory hygiene for long runs
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    elif device == "mps":
                        torch.mps.empty_cache() 

    finally:
        _flush(pending, cache_path)
        consolidate_cache(str(cache_path))
        print(f"Progress saved to {cache_path}")