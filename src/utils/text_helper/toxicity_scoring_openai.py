import time
import random 
import polars as pl
from pathlib import Path
from tqdm.auto import tqdm
from openai import OpenAI, RateLimitError

CATEGORY_COLS = [
    "flagged",
    "harassment", "harassment_threatening",
    "hate", "hate_threatening",
    "self_harm", "self_harm_instructions", "self_harm_intent",
    "sexual", "sexual_minors",
    "violence", "violence_graphic",
    "illicit", "illicit_violent",
]

def _parse_result(id_: str, result) -> dict:
    s = result.category_scores
    return {
        "id": id_,
        "flagged": result.flagged,
        "harassment": s.harassment,
        "harassment_threatening": s.harassment_threatening,
        "hate": s.hate,
        "hate_threatening": s.hate_threatening,
        "self_harm": s.self_harm,
        "self_harm_instructions": s.self_harm_instructions,
        "self_harm_intent": s.self_harm_intent,
        "sexual": s.sexual,
        "sexual_minors": s.sexual_minors,
        "violence": s.violence,
        "violence_graphic": s.violence_graphic,
        "illicit":s.illicit,
        "illicit_violent":s.illicit_violent,
    }

def _empty_result(id_: str) -> dict:
    return {"id": id_, "flagged": None, **{c: None for c in CATEGORY_COLS[1:]}}

def _get_shards(cache_path: Path) -> list[Path]:
    return sorted(cache_path.parent.glob(f"{cache_path.stem}_shard_*.parquet"))

# def _load_scored_ids(cache_path: Path) -> set:
#     dfs = []
#     if cache_path.exists():
#         dfs.append(pl.read_parquet(cache_path))
#     for shard in _get_shards(cache_path):
#         dfs.append(pl.read_parquet(shard))
#     if not dfs:
#         return set()
#     return set(pl.concat(dfs)["id"].to_list())

def _load_scored_ids(cache_path: Path) -> set:
    dfs = []
    if cache_path.exists():
        dfs.append(pl.read_parquet(cache_path))
    for shard in _get_shards(cache_path):
        dfs.append(pl.read_parquet(shard))
    if not dfs:
        return set()
    df = pl.concat(dfs)
    # only count rows with an actual moderation result as completed
    df_done = df.filter(pl.col("flagged").is_not_null())

    return set(df_done["id"].to_list())

def _flush(pending: list[dict], cache_path: Path):
    if not pending:
        return
    shard_idx = len(_get_shards(cache_path))
    shard_path = cache_path.parent / f"{cache_path.stem}_shard_{shard_idx:05d}.parquet"
    pl.DataFrame(pending).write_parquet(shard_path)

# def consolidate_cache(cache_path: str = "toxicity_cache.parquet") -> pl.DataFrame:
#     cache_path = Path(cache_path)
#     shards = _get_shards(cache_path)
#     dfs = []
#     if cache_path.exists():
#         dfs.append(pl.read_parquet(cache_path))
#     for shard in shards:
#         dfs.append(pl.read_parquet(shard))
#     if not dfs:
#         return pl.DataFrame()
#     df = pl.concat(dfs)
#     df.write_parquet(cache_path)
#     for shard in shards:
#         shard.unlink()
#     print(f"Consolidated {len(shards)} shards into {cache_path} ({len(df):,} rows total).")
#     return df

def consolidate_cache(cache_path: str = "toxicity_cache.parquet") -> pl.DataFrame:
    cache_path = Path(cache_path)
    shards = _get_shards(cache_path)
    dfs = []
    if cache_path.exists():
        dfs.append(pl.read_parquet(cache_path))
    for shard in shards:
        dfs.append(pl.read_parquet(shard))
    if not dfs:
        return pl.DataFrame()

    df = pl.concat(dfs)
    # prefer rows with real results over null-placeholder rows
    df = (
        df.with_columns(
            pl.col("flagged").is_not_null().cast(pl.Int8).alias("_has_result")
        )
        .sort(["id", "_has_result"])
        .group_by("id", maintain_order=True)
        .tail(1)
        .drop("_has_result")
    )

    df.write_parquet(cache_path)
    for shard in shards:
        shard.unlink()

    print(f"Consolidated {len(shards)} shards into {cache_path} ({len(df):,} rows total).")
    return df

class DailyLimitExceeded(Exception):
    pass

def _is_probably_daily_limit_error(e: Exception) -> bool:
    """
    Inspect the message only after we've already identified a RateLimitError.
    This is still heuristic, but less fragile than classifying all exceptions
    via string matching.
    """
    msg = str(e).lower()
    return any(term in msg for term in ["daily", "per day", "day", "rpd", "tpd", "quota"])


# def _is_rate_limit_error(e: Exception) -> bool:
#     return "429" in str(e)

def score_toxicity(
    df: pl.DataFrame,
    cache_path: str = "toxicity_cache.parquet",
    model: str = "omni-moderation-latest",
    batch_size: int = 50,          # keep 100 so ~987,770 rows fits within 10k RPD
    requests_per_minute: int = 10,  # samller to avoid bursts
    max_chars: int = 10_000,
    flush_every_n_batches: int = 10,
    max_retries: int = 10,
) -> None:
    """
    Score toxicity for each row in df using the OpenAI moderation endpoint.

    Parameters
    ----------
    df                      : must have columns "id" (str) and "text" (str)
    cache_path              : base parquet path; shards written during scoring,
                              consolidated into this file at the end
    model                   : moderation model to use
    batch_size              : texts per request (100 fits ~987K rows in ~9,878
                              requests, just under the 10,000 RPD cap)
    requests_per_minute     : effective rate (set to 40% of ceiling to avoid bursts)
    max_chars               : truncate texts before sending
    flush_every_n_batches   : write a new shard every N batches
    max_retries             : exponential backoff retries on failure

    Stops gracefully when daily RPD limit is exhausted — safe to re-run
    next day to continue from where it left off.
    Results saved to cache_path. Resume-safe: re-running skips already-scored rows.
    """

    client = OpenAI()
    cache_path = Path(cache_path)

    # use monotonic clock for pacing; avoids issues if system clock changes
    min_interval = 60.0 / requests_per_minute
    next_allowed_time = time.monotonic()

    scored_ids = _load_scored_ids(cache_path)
    print(f"Resuming: {len(scored_ids):,} already scored, skipping.")

    todo = df.filter(~pl.col("id").is_in(scored_ids))
    n_todo = len(todo)
    print(f"Remaining: {n_todo:,} rows to score.")

    if n_todo == 0:
        consolidate_cache(str(cache_path))
        return

    ids = todo["id"].to_list()
    texts = [(t or "")[:max_chars] for t in todo["text"].to_list()]

    pending = []

    try:
        with tqdm(total=n_todo, desc="Scoring toxicity") as pbar:
            for batch_idx, i in enumerate(range(0, n_todo, batch_size)):
                batch_ids   = ids[i:i + batch_size]
                batch_texts = texts[i:i + batch_size]

                # request with exponential backoff
                success = False
                for attempt in range(max_retries):
                    # enforce pacing before every attempt, not just based on last success
                    now = time.monotonic()
                    if now < next_allowed_time:
                        time.sleep(next_allowed_time - now)

                    try:
                        response = client.moderations.create(
                            model=model,
                            input=batch_texts,
                        )
                        next_allowed_time = time.monotonic() + min_interval

                        for id_, result in zip(batch_ids, response.results):
                            pending.append(_parse_result(id_, result))

                        success = True
                        break

                    except RateLimitError as e:

                        # # FOR DEBUG: show the full error and any rate-limit/reset information
                        # tqdm.write(f"Response body: {e.response.text}")
                        # for key, value in e.response.headers.items():
                        #     if key.lower() == "retry-after" or key.lower().startswith("x-ratelimit"):
                        #         tqdm.write(f"{key}: {value}")

                        if _is_probably_daily_limit_error(e):
                            raise DailyLimitExceeded(str(e))
                        
                        # exponential backoff with jitter, capped
                        wait_time = min(15.0, (2 ** attempt) + random.uniform(0, 1))
                        # after a failed attempt, still push out next allowed time
                        next_allowed_time = time.monotonic() + wait_time

                        tqdm.write(
                            f"[attempt {attempt+1}/{max_retries}] "
                            f"Rate limit on batch {batch_idx} — waiting {wait_time:.1f}s"
                        )
                        time.sleep(wait_time)

                    except Exception as e:
                        # separate non-rate-limit transient failures
                        wait_time = min(30.0, (2 ** attempt) + random.uniform(0, 1))
                        next_allowed_time = time.monotonic() + wait_time

                        tqdm.write(
                            f"[attempt {attempt+1}/{max_retries}] "
                            f"Error on batch {batch_idx}: {e} — retrying in {wait_time:.1f}s"
                        )
                        time.sleep(wait_time)

                if not success:
                    tqdm.write(f"Failed after {max_retries} retries on batch {batch_idx}, recording nulls.")
                    for id_ in batch_ids:
                        pending.append(_empty_result(id_))

                pbar.update(len(batch_ids))

                if (batch_idx + 1) % flush_every_n_batches == 0:
                    _flush(pending, cache_path)
                    pending = []

    except DailyLimitExceeded as e:
        tqdm.write(f"\nDaily limit reached: {e}")
        tqdm.write("Saving progress — re-run tomorrow to continue.")

    finally:
        # always flush and consolidate whatever was scored before stopping
        _flush(pending, cache_path)
        consolidate_cache(str(cache_path))
        print(f"Progress saved to {cache_path}")