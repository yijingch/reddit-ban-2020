import glob, os
import pandas as pd 
import numpy as np 
import polars as pl
from typing import List
from collections import Counter
from src.utils.data_loader_polars import load_filtered_data
from src.utils.text_helper.toxicity_scoring_openai import score_toxicity
import yaml

if __name__ == "__main__":

    print("1. setting up environment...")
    with open("secrets/key.yml") as f:
        secrets = yaml.safe_load(f)
    os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]

    with open("src/configs.yml", "r") as configs:
        configs = yaml.safe_load(configs)

    DATAPATH = configs["SERVER_DATA"]
    ROOTPATH = configs["SERVER_WD"]
    print(configs)

    X0 = 10; N0 = 1
    X1 = 20; N1 = 10
    X2 = "inf"; N2 = 1
    covs1 = "lifetime_d"
    covs2 = "lifetime_d_n_post_author_till2019_log"

    print("2. fetching samples...")

    clean_text_dir = f"{DATAPATH}/filtered-data/subm_comm_clean/"
    cleaned_text_banned = pl.read_parquet(f"{clean_text_dir}subm-comm-from-subr-banned-2020-aggr-by-subm-2006-2019.parquet")
    cleaned_text_matched1_indp = pl.read_parquet(f"{clean_text_dir}subm-comm-from-subr-banned-2020-subset3274-match-{covs1}-X{X0}N{N0}-aggr-by-subm-2006-2019.parquet")
    cleaned_text_matched2_indp = pl.read_parquet(f"{clean_text_dir}subm-comm-from-subr-banned-2020-subset3274-match-{covs2}-X{X1}N{N1}-aggr-by-subm-2006-2019.parquet")
    cleaned_text_matched2_conn = pl.read_parquet(f"{clean_text_dir}subm-comm-from-subr-banned-2020-subset3274-match-{covs2}-X{X2}N{N2}-aggr-by-subm-2006-2019.parquet")
    cleaned_text_random = pl.read_parquet(f"{clean_text_dir}subm-comm-from-subr-banned-2020-subset3274-random-aggr-by-subm-2006-2019.parquet")

    n_per_sample = 5_000
    sample = pl.DataFrame()
    for df in [
        cleaned_text_banned, 
        cleaned_text_matched1_indp, 
        cleaned_text_matched2_indp, 
        cleaned_text_matched2_conn, 
        cleaned_text_random]:
        sample = pl.concat([sample, df.sample(n=n_per_sample, seed=42)])

    print(f"\ttotal samples: {sample.shape[0]}")

    print("3. scoring toxicity...")

    model = "omni-moderation-latest"
    save_dir = f"{DATAPATH}/output/toxicity/openai-eval/{model}"
    score_toxicity(
        # df=sample1.head(200), # for testing
        df = sample,
        model = model,
        cache_path = f"{save_dir}/subm-comm-from-subr-5groups-random-{n_per_sample}-toxicity.parquet",
        batch_size = 1, 
        requests_per_minute = 5,
        flush_every_n_batches = 2,
    )

# (reddit-ban) yijingch@asc.upenn.edu@cindsrv01:~/workingdir/reddit-ban-2020$
# python src/preprocess/labeling/score_toxicity_openai_eval.py