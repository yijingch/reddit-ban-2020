# Preprocessing guide: from raw Reddit data to analysis inputs

This guide provides a step-by-step documentation of operations in preparing the data for `src/analysis`. 



## Pipeline and run order

```text
Converted Reddit archive (monthly Parquet files)
  0. Environment setup
  1. Count raw activity
  2. Identify banned communities
  3. Count cumulative distinct participants for matching
  4. Extract user histories and measure overlap
  5. Prepare covariates and construct comparison sets
  6. Prepare within-set and neighboring-community records
  7. Aggregate document and score toxicity
  8. Build co-posting networks
  9. Handoff to analysis
```


## 0. Environment setup

### Working directory and configuration

Install and activate the environment using the root README's [installation instructions](../../README.md#installation-and-reproduction). Set three folders** in [`src/configs.yml`](../configs.yml):

| Setting | Purpose |
| --- | --- |
| `RAW_DATA` | Converted archive containing `submissions/` and `comments/`; defaults to `/mnt/RedditData` |
| `SERVER_DATA` | Project data workspace for counts, extracts, sample tables, and scores |
| `SERVER_WD` | Repository root |


The reference [`environment.yml`](../../environment.yml) records Python 3.9.23 and Linux/CUDA packages; recreating it on another platform may require a platform-specific environment. R requirements are listed in step 5.

### Raw archive

The scripts start from an **already converted Parquet archive**. Archive download and JSON-to-Parquet conversion are not implemented in this folder. With `RAW_DATA: /mnt/RedditData`, the expected raw layout is:

```text
/mnt/RedditData/submissions/year=YYYY/month=M/*.parquet
/mnt/RedditData/comments/year=YYYY/month=M/*.parquet
```

CIND server currently has data from June 2015 to December 2022. Historical analyses use activity through 2019; the bann-subreddit identification step also needs 2020–2022 coverage. 


## 1. Count raw activity

### 1a. Subreddit-year summaries

**Script:** [`src/counting/activity-author-by-subreddit-by-year-polars.py`](counting/activity-author-by-subreddit-by-year-polars.py).

**Input:** raw year/month Parquet files. 

**Operation:** count distinct IDs and authors per subreddit-year, record first/last activity dates within that yer, and optionally combine submissions and comments. Joint `n_author` counts their union of authors.

**Output:** `data/counts-raw/`, with the following files for each year:

| Product | `YEARS` | `COUNT_MODE` | `FILETYPE` | Filename | Main count columns |
| --- | --- | --- | --- | --- | --- |
| Submission summaries | `range(2005, 2023)` | `"separate"` | `"submissions"` | `count_submissions_n_item_n_author_minmax_utc_YYYY.parquet` | `n_subm`, `n_subm_author` |
| Comment summaries | `range(2005, 2023)` | `"separate"` | `"comments"` | `count_comments_n_item_n_author_minmax_utc_YYYY.parquet` | `n_comm`, `n_comm_author` |
| Historical joint summaries and 2020 eligibility | `range(2005, 2021)` | `"joint"` | Ignored | `count_joint_n_item_n_author_minmax_utc_YYYY.parquet` | Separate counts and union `n_author` |

Each row records a unique subreddit-year.

```bash
mkdir -p /absolute/path/to/project-data/data/counts-raw
python src/preprocess/counting/activity-author-by-subreddit-by-year-polars.py
```

### 1b. Author-subreddit-year summaries

**Script:** [`src/preprocess/counting/activity-by-author-by-subreddit-by-year-polars.py`](counting/activity-by-author-by-subreddit-by-year-polars.py).

**Input:** raw year/month Parquet files.

**Operation:** count each author's records per subreddit-year and retain first/last timestamps. Counts are summed across shards. Null/empty authors and communities are removed.

**Output:** saved in `data/counts-raw/`, with one file per year.

| Product | `YEARS` | `FILETYPE` | Filename | Count column |
| --- | --- | --- | --- | --- |
| Submission activity | `range(2005, 2020)` | `"submissions"` | `author_subreddit_count_submissions_minmax_utc_YYYY.parquet` | `n_subm` |
| Comment activity | `range(2005, 2020)` | `"comments"` | `author_subreddit_count_comments_minmax_utc_YYYY.parquet` | `n_comm` |

Each row records a unique author-subreddit-year.

```bash
python src/preprocess/counting/activity-by-author-by-subreddit-by-year-polars.py
```


## 2. Identify banned communities

This stage was carried out locally, so its paths refer to local directories and do not follow the server directory structure. The collection code may stop working as Reddit changes its restrictions on URL requests. The cleaned community memberships are included in the released subreddit sets, so this stage can be skipped when using those sets.

### 2a. Submission-only inactivity screen and status classification

**Notebook:** [`src/preprocess/collecting/251001-identify-banned-subreddits.ipynb`](collecting/251001-identify-banned-subreddits.ipynb).

**Input:** 2020–2022 submission summaries and saved homepage-response JSON. Current reads use repository `data/counts/` and `index/`, independently of `SERVER_DATA`.

**Operation:** exclude `u_` profiles; require ≥10 submissions and ≥5 submission authors in 2020; remove communities with later submissions; classify saved messages; apply `MANUL_CORRECT_MAP` from [`label_subreddit.py`](../utils/label_subreddit.py); extract ban reasons.

**Output:** saved in `data/counts-raw/`, with one file per year.

### 2b. Refine inactivity using both submissions and comments

**Notebook:** [`src/preprocess/collecting/260327-2nd-round-cleaning.ipynb`](collecting/260327-2nd-round-cleaning.ipynb).

**Input:** 2020–2022 submission/comment summaries in repository `data/counts-raw/`; initial banned CSV and saved responses in repository `index/`. Adjust count paths if the products live under `SERVER_DATA`.

**Operation:** revisit the initial banned list using **both submission and comment activity in 2021–2022**. Remove communities with either type of activity, including those with no later submissions but continued comments that passed step 2a. Mark the remaining communities—those with **neither submissions nor comments in 2021–2022**—with `active_filter == 1`. 

**Output:** repository `index/banned_subreddits_cleaned_251006_add_filter.csv`, retaining the initial fields and adding `active_filter`. Downstream selection keeps `active_filter == 1` and excludes 2020-born communities.


## 3. Count cumulative distinct participants for matching

**Script:** [`src/preprocess/counting/acc-activity-author-by-subreddit-by-year-polars.py`](counting/acc-activity-author-by-subreddit-by-year-polars.py).

**Input:** raw submissions, plus raw comments for the combined count; `COUNT_DIR/count_submissions_n_item_n_author_minmax_utc_{TARGET_YEAR}.parquet` from step 1a.

**Operation:** select communities with ≥10 submissions and ≥5 submission authors in the target year, then count cumulative distinct authors from 2005 through 2020. `[deleted]` authors are excluded; returning participants count once cumulatively.

**Output:** saved in `data/counts-raw/`, with one file per mode.  Filenames below use `label = "2020active"` and `N = len(all_subrs)`.

| Product | `INCLUDE_COMMENTS` | `YEARS` | Filename | Main columns |
| --- | --- | --- | --- | --- |
| Submission authors only | `False` | `range(2005, TARGET_YEAR + 1)` | `2020active_subreddits_N_acc_n_subm_author.parquet` | `subreddit`, `acc_subm_author2005` … `acc_subm_author2020` |
| Submission/comment author union | `True` (default) | `range(2005, TARGET_YEAR + 1)` | `2020active_subreddits_N_acc_n_post_author.parquet` | `subreddit`, `acc_post_author2005` … `acc_post_author2020` |

Set `INCLUDE_COMMENTS = False` for submissions only, or keep `True` for the combined count. Only the combined mode reads comments and uses scratch shards under `data/tmp/subreddit_author_year/`; these are reset before running and removed afterward when `CLEAN_UP_SHARDS = True`.

```bash
python src/preprocess/counting/acc-activity-author-by-subreddit-by-year-polars.py
```


## 4. Extract user histories and measure overlap

Commands below use the released `banned` set. For a rebuild, replace the membership input with a `set,subreddit` CSV selected using `active_filter == 1` and positive historical lifespan from step 5a. Keep B membership consistent across stages.

### 4a. Extract records in B

**Script:** [`src/preprocess/filtering/filter_raw_data_by_subr_cli.py`](filtering/filter_raw_data_by_subr_cli.py).

**Input:** raw Parquet files and repository `data-public/subreddit_sets.csv`.

**Operation:** select B communities and retrieve their submissions/comments for 2005–2019.

**Output:** `filtered-data-new/{submissions,comments}/`, in the subdirectories shown below. The CLI creates output folders.

| Product | `--filetype` | Filename |
| --- | --- | --- |
| B submissions | `submissions` | `submissions_filtered_by_subreddits_YYYY.parquet` |
| B comments | `comments` | `comments_filtered_by_subreddits_YYYY.parquet` |

```bash
for kind in submissions comments; do
  python src/preprocess/filtering/filter_raw_data_by_subr_cli.py \
    --input data-public/subreddit_sets.csv --where "set == 'banned'" \
    --filetype "$kind" --years 2005:2019 \
    --output "{DATAPATH}/filtered-data-new/$kind/${kind:0:4}-from-subr-banned-2020-subset3274/"
done
```


### 4b. Retrieve platform-wide histories of B users

**Script:** [`src/preprocess/filtering/filter_raw_data_by_user_cli.py`](filtering/filter_raw_data_by_user_cli.py).

**Input:** both B record directories from step 4a and raw Parquet files.

**Operation:** union submission/comment authors, exclude `[deleted]`, and retrieve their platform-wide records for 2005–2019.

**Output:** `filtered-data-new/{submissions,comments}/...from-user-banned-2020-subset3274-lite/`, containing `{filetype}_filtered_by_subreddit_users_YYYY.parquet`. Output directories are created automatically; when marking `-lite` is a label the filter does not retain title or textbody columns.

```bash
for kind in submissions comments; do
  python src/preprocess/filtering/filter_raw_data_by_user_cli.py \
    --input \
      "{DATAPATH}/filtered-data-new/submissions/subm-from-subr-banned-2020-subset3274/" \
      "{DATAPATH}/filtered-data-new/comments/comm-from-subr-banned-2020-subset3274/" \
    --input-years 2005:2019 --filetype "$kind" --years 2005:2019 \
    --output "{DATAPATH}/filtered-data-new/$kind/${kind:0:4}-from-user-banned-2020-subset3274-lite/"
done
```

### 4c. Count user overlap with B

**Script:** [`src/preprocess/counting/threshold_user_overlap.py`](counting/threshold_user_overlap.py).

**Input:** refined banned CSV in `index/`; step 1a's 2006–2020 submission summaries and step 1b's `author_subreddit_count_{submissions,comments}_minmax_utc_YYYY.parquet` for **2005–2019**, both in `data/counts-raw/`.

**Operation:** obtain the set of banned subreddits, count distinct shared users per candidate at varying $P_{min}$ threshold. Null/`[deleted]` authors and `u_` pages are excluded. 

**Output:** `data/counts-raw/n_overlap_user_post_w_banned-2020-3274_thresholded.parquet`, with `subreddit` and `u_overlap_min_post_ban_{threshold}` columns.

Run from the repository root; configuration is read directly from `src/configs.yml` (`--config` overrides it), and the output directory is created automatically:

```bash
python src/preprocess/counting/threshold_user_overlap.
# sanity check with filtered data
python test/test_threshold_user_overlap.py
```

## 5. Prepare covariates and construct comparison sets

### 5a. Prepare the matching table

**Notebook:** [`src/preprocess/matching/01-prepare-subr-to-match.ipynb`](matching/01-prepare-subr-to-match.ipynb).

**Input:** 2020 submission counts, 2006–2020 joint counts, and cumulative participant counts from `data/counts-raw/`; refined banned CSV from `index/`; overlap table from `data/subreddits_metadata/`. Author-year counts support a population-size diagnostic.

**Operation:** select 2020-active communities (≥10 submissions, ≥5 submission authors, excluding `u_` pages); compute lifespan (first activity to 2019/12/31); exclude subreddits born in 2020; merge tables and build covariates and overlap; popularity is `log1p(total users through 2019)`.

**Output:**

| Product | Folder | Filename / fields |
| --- | --- | --- |
| Full matching table | `data/samples/` | `to_match_table_selfid_banned_3274_260528.csv`; membership, lifespan, activity, popularity, and overlap fields |
| Pool-size plots | Repository `output/appendix/` | `matching-pool-sizes-Amax-Pmin.{png,pdf}` |

### 5b. Run CEM on the selected subreddits

**Script:** [`src/preprocess/matching/02-match-samples-cem-multibatch4.Rmd`](matching/02-match-samples-cem-multibatch4.Rmd).

**Input:** step 5a's preparation CSV in `data/samples/`.

**Operation:** restrict controls by overlap, coarsen covariates, and select one-to-one CEM matches using Mahalanobis distance within strata; `set.seed(42)`. `N` (`P_min`) is minimum user activity across B; `X` (`A_max`) limits qualifying shared users per control community.

**Output:** `data/samples/cem-batch4-new/`, using prefix `matched_table_banned_3274_cem_`:

| Set | `X` | `N` | Covariates | Cutpoints | Filename suffix |
| --- | --- | --- | --- | --- | --- |
| `matched1_indp` | `10` | `1` | `lifetime_d` | 5%-95% in 5% steps | `X10N1_lifetime_d.csv` |
| `matched2_indp` | `20` | `10` | `lifetime_d`, `n_post_author_till2019_log` | 10%–90% in 10% steps for both covariates | `X20N10_lifetime_d_n_post_author_till2019_log.csv` |
| `matched2_conn` | `Inf` | `1` | `lifetime_d`, `n_post_author_till2019_log` | 10%–90% in 10% steps for both covariates | `XinfN1_lifetime_d_n_post_author_till2019_log.csv` |

This step was also done locally with paths referring to local directories. Requires R packages `cem`, `MatchIt`, `cobalt`, `dplyr`, `ggplot2`, `tcltk`; rendering adds `rmarkdown`/`knitr`. The optional [pool-overlap diagnostic](matching/03-evaluate-cov-overlap-ban-pool.Rmd) adds `tidyr`/`ggnewscale`.


### 5c. Freeze subreddit set and release in `public-data` folder

**Notebook:** [`src/analysis/00-load-subreddit-sets.ipynb`](../analysis/00-load-subreddit-sets.ipynb).

**Input:** preparation CSV and three matching CSVs from `data/samples/`.

**Operation:** read from matched tables and gather five sets of subreddits.

**Output:** repository `data-public/subreddit_sets.csv`, with `set,subreddit` and 3,274 distinct communities per set.


## 6. Prepare within-set and neighboring-community records

### 6a. Extract activity within each set

**Script:** [`src/preprocess/filtering/filter_raw_data_by_subr_cli.py`](filtering/filter_raw_data_by_subr_cli.py).

**Input:** raw Parquet files and public/rebuilt membership CSV. Use step 0's full-text profile for subsequent scoring.

**Operation:** retrieve all submissions/comments in each fixed set for 2005–2019; basically repeating 4a for other four sets of subreddits for comparison.

**Output:** `filtered-data/{submissions,comments}/{subm,comm}-from-subr-{label}/`, containing `{filetype}_filtered_by_subreddits_YYYY.parquet`. Directories are created automatically.

| `--where` selection | `{label}` used in output paths and steps 7a/7c |
| --- | --- |
| `set == 'banned'` | `banned-2020` |
| `set == 'matched1_indp'` | `banned-2020-subset3274-match-lifetime_d-X10N1` |
| `set == 'matched2_indp'` | `banned-2020-subset3274-match-lifetime_d_n_post_author_till2019_log-X20N10` |
| `set == 'matched2_conn'` | `banned-2020-subset3274-match-lifetime_d_n_post_author_till2019_log-XinfN1` |
| `set == 'random'` | `banned-2020-subset3274-random` |

```bash
# example for matched1_indp
for kind in submissions comments; do
  python src/preprocess/filtering/filter_raw_data_by_subr_cli.py \
    --input data-public/subreddit_sets.csv --where "set == 'matched1_indp'" \
    --filetype "$kind" --years 2005:2019 \
    --output "{DATAPATH}/filtered-data/$kind/${kind:0:4}-from-subr-banned-2020/"
done
```

### 6b. Count distinct users across whole sets

**Script:** [`src/preprocess/counting/acc-activity-author-by-subreddit-sets-polars.py`](counting/acc-activity-author-by-subreddit-sets-polars.py).

**Input:** raw data and released subreddit sets.

**Operation:** deduplicate `(set,author,year)`, identify first appearance, and count new/cumulative users for submissions, comments, and their union. Multiple communities within a set count once per user; `[deleted]` authors are excluded.

**Output:** `data/counts-raw/`, with `set`, `year`, `n_new_users_{subm,comm,post}`, and `n_cumulative_users_{subm,comm,post}`. Scratch: `data/tmp/subreddit_set_author_year/`, reset automatically.

| Product | `years` | Membership | Filename |
| --- | --- | --- | --- |
| Cumulative user count for 5 sets of subreddits | `list(range(2005, 2020))` | Public membership CSV | `subreddit_set_user_counts_2005_2019.parquet` |

```bash
python src/preprocess/counting/acc-activity-author-by-subreddit-sets-polars.py
```


### 6c. Extract all records in neighboring communities

**Script:** [`src/preprocess/filtering/filter_raw_data_by_subr_cli.py`](filtering/filter_raw_data_by_subr_cli.py).

**Input:** banned-subreddit user submission and comment histories from step 4b (`filtered-data-new/...from-user-banned-2020-subset3274-lite/`), banned subreddit set from `data-public/subreddit_sets.csv`, and raw Parquet files. Use step 0's full-text profile for the raw extraction.

**Operation:** take the union of communities appearing in banned-subreddit users' submissions and comments during 2005–2019; exclude banned subreddits and `u_` profile pages; retrieve all submissions and comments in the remaining subreddits for 2005–2019. These neighborhood extracts include records by all authors, including people outside the banned subreddits' user cohort.

**Output:**

| Product | Folder | Filename |
| --- | --- | --- |
| Neighborhood submissions | `filtered-data-new/submissions/subm-from-subr-banned-2020-subset3274-neighbors-only/` | `submissions_filtered_by_subreddits_YYYY.parquet` |
| Neighborhood comments | `filtered-data-new/comments/comm-from-subr-banned-2020-subset3274-neighbors-only/` | `comments_filtered_by_subreddits_YYYY.parquet` |

```bash
for kind in submissions comments; do
  python src/preprocess/filtering/filter_raw_data_by_subr_cli.py \
    --input \
      "{DATAPATH}/filtered-data-new/submissions/subm-from-user-banned-2020-subset3274-lite/" \
      "{DATAPATH}/filtered-data-new/comments/comm-from-user-banned-2020-subset3274-lite/" \
    --input-years 2005:2019 \
    --exclude data-public/subreddit_sets.csv --exclude-where "set == 'banned'" \
    --filetype "$kind" --years 2005:2019 \
    --output "{DATAPATH}/filtered-data-new/$kind/${kind:0:4}-from-subr-banned-2020-subset3274-neighbors-only/"
done
```


## 7. Aggregate document and score toxicity

### 7a. Create a clean document file per subreddit set

**Script:** [`src/preprocess/labeling/aggr_subm_for_toxicity_scoring.py`](labeling/aggr_subm_for_toxicity_scoring.py).

**Input:** full submissions/comments from step 4a and 6a, for 2006–2019.

**Operation:** join comments after removing `t3_` from submission links; remove empty/deleted/removed comments and blank submission-body placeholders; order top-level comments before nested comments, chronologically within each group; concatenate title, body, and comments; clean Markdown/whitespace and URLs.

**Output:** `filtered-data/subm_comm_clean/subm-comm-from-subr-{label}-aggr-by-subm-2006-2019.parquet`, with one `id,text` row per submission. The parent directory is created automatically.

```bash
# example here for the banned set
python src/preprocess/labeling/aggr_subm_for_toxicity_scoring.py \
  --subm-dir /absolute/path/to/project-data/filtered-data/submissions/subm-from-subr-banned-2020/ \
  --comm-dir /absolute/path/to/project-data/filtered-data/comments/comm-from-subr-banned-2020/ \
  --input-years 2006:2019 \
  --out /absolute/path/to/project-data/filtered-data/subm_comm_clean/subm-comm-from-subr-banned-2020-aggr-by-subm-2006-2019.parquet
```


### 7b. Bucketed alternative for larger inputs (e.g., submissions in connected subreddits)

**Script:** [`src/preprocess/labeling/aggr_subm_for_toxicity_scoring_bucketed.py`](labeling/aggr_subm_for_toxicity_scoring_bucketed.py).

**Input:** full neighborhood submissions/comments from step 6c, for **2005–2019**.

**Operation:** stream records into submission-ID hash buckets, then assemble/clean threads one bucket at a time using the same aggregation procedure in step 7a. 

**Output:**

| Product | Folder / setting | Files / columns |
| --- | --- | --- |
| Cleaned parts | `filtered-data/subm_comm_clean/banned-2020-thread-parts/` | `part_*.parquet`; `id,text` |
| Scratch buckets | Sibling `<output-directory-name>_buckets`, or `--tmp-dir` | Removed after success unless `--keep-tmp` |

```bash
python src/preprocess/labeling/aggr_subm_for_toxicity_scoring_bucketed.py \
  --subm-dir "{DATAPATH}/filtered-data-new/submissions/subm-from-subr-banned-2020-subset3274-neighbors-only/" \
  --comm-dir "{DATAPATH}/filtered-data-new/comments/comm-from-subr-banned-2020-subset3274-neighbors-only/" \
  --years 2005:2019 \
  --out-dir "{DATAPATH}/filtered-data-new/subm_comm_clean/subm-comm-from-subr-banned-2020-subset3274-neighbors-only-aggr-by-subm-2005-2019" \
  --buckets 256
```


### 7c. Assign toxicity score to cleaned documents

**Script:** [`src/preprocess/labeling/score_toxicity_detoxify.py`](labeling/score_toxicity_detoxify.py).

**Input:** cleaned `id,text` from step 7a for the five fixed sets, or neighborhood `part_*.parquet` files from step 7b. Use `--per-file` for the neighborhood directory to process one part at a time.

**Operation:** score all supplied threads with Detoxify's multilingual model, truncating to 10,000 characters before model tokenization. Community analysis subsequently samples up to 50 submissions per subreddit-week with seed 42; `--sample-per-week` is a separate optional scoring mode.

**Output:** under `output/toxicity/subset3274/detoxify-multilingual-all-subm-perweek/`:

| Product | File / directory |
| --- | --- |
| Within-set scores | `subm-comm-from-subr-{label}-by-subm-toxicity.parquet`, plus intermediate cache shards |
| Banned neighborhood scores | `subm-comm-from-subr-banned-2020-subset3274-neighbors-only/part_0000-toxicity.parquet`, etc., and completion markers |

Columns: `id`, `toxicity`, `severe_toxicity`, `obscene`, `threat`, `insult`, `identity_attack`; unavailable results may be null.


For the within-set scoring, repeat the first command for all five labels. Use the second command for the neighborhood. Output directories are created automatically. Device selection is automatic (`--device cpu` or a CUDA device overrides it). Caches use submission IDs; choose a new output location when text/model settings change.

```bash
# the example here scores the banned subreddits' documents
python src/preprocess/labeling/score_toxicity_detoxify.py \
  --input "{DATAPATH}/filtered-data/subm_comm_clean/subm-comm-from-subr-banned-2020-aggr-by-subm-2006-2019.parquet" \
  --output "{DATAPATH}/output/toxicity/subset3274/detoxify-multilingual-all-subm-perweek/subm-comm-from-subr-banned-2020-by-subm-toxicity.parquet" \
  --model multilingual --batch-size 128 --max-chars 10000
```

Neighborhood scoring uses the full 2005–2019 threads assembled in step 7b:

```bash
python src/preprocess/labeling/score_toxicity_detoxify.py \
  --input "{DATAPATH}/filtered-data-new/subm_comm_clean/subm-comm-from-subr-banned-2020-subset3274-neighbors-only-aggr-by-subm-2005-2019/" \
  --per-file --model multilingual --batch-size 128 --max-chars 10000 \
  --output "{DATAPATH}/output/toxicity/subset3274/detoxify-multilingual-all-subm-perweek/subm-comm-from-subr-banned-2020-subset3274-neighbors-only/"
```

Rerun the same command to resume unfinished parts. A part is marked complete only when every input ID has non-null primary toxicity.


### 7d: compare Detoxify and OpenAI scores

**Script / notebook:** [`src/preprocess/labeling/score_toxicity_openai_eval.py`](labeling/score_toxicity_openai_eval.py), then [`evaluate-detoxify-openai.ipynb`](labeling/evaluate-detoxify-openai.ipynb).

**Input:** five cleaned single-file thread products from step 7a, Detoxify scores, matching tables, and repository `secrets/key.yml` containing `OPENAI_API_KEY`.

**Operation:** sample 5,000 threads per set, call `omni-moderation-latest`, and compare scores using Spearman correlations. This comparison is included in the appendix.

**Output:**
| Product | Folder | Filename |
| --- | --- | --- |
| OpenAI scores | `output/toxicity/openai-eval/omni-moderation-latest/` | `subm-comm-from-subr-5groups-random-5000-toxicity.parquet` |
| Correlation figures | Repository `output/appendix/` | `score_correlation_detoxify_openai.{pdf,png}` |

```bash
python src/preprocess/labeling/score_toxicity_openai_eval.py
```


## 8. Build co-posting networks

### 8a. Construct annual subreddit networks

**Script:** [`src/preprocess/network_generation/networks.py`](network_generation/networks.py).

**Input:** monthly raw submission and comment Parquet files containing `author` and `subreddit`.

<!-- Notes: `I` is the numbered shard index, starting at zero. This module reads subreddit names directly; it does not extract them from submission permalinks. -->

**Operation:** build monthly subreddit-to-user sets, then take their union across months to construct annual networks. Remove missing authors/subreddits, `u_` profile pages, `[deleted]`, AutoModerator, and usernames matching the script's bot-name pattern. 
* Each graph is undirected. Nodes represent subreddits; an edge connects two subreddits when they share at least one retained author during the year. Edge weights count distinct shared authors. Vertex properties record the subreddit name (`name`) and number of retained authors (`size`); the edge property is `weight`.
* For the combined networks, submissions and comments are pooled before applying the activity threshold.
* The threshold is **monthly**, not annual. An author enters a subreddit's annual user set if they qualify in at least one month.
* Networks cover all retained communities. Banned and comparison communities are selected subsequently in the analysis.


**Output:**

| Product | Location | Contents |
| --- | --- | --- |
| Full yearly networks | Selected network output root, under `full/` (`threshold=0`) | `YYYY.gt.gz`|
| Filtered yearly networks | Selected network output root, under `filtered/` (`threshold=2`) | `YYYY_2.gt.gz` |

**Configuration:** this module imports a separate Python `config` module, rather than `src/configs.yml`. That configuration is not included in this folder and must be supplied before execution. It requires:

- `raw_comments`, `raw_submissions`
- `comments`, `comments_path`, `submissions_path`
- `combined_path`, `combined_f_path`
- `cs_f_networks_path`


To run the code, use:
```bash
python - <<'PY'
from pathlib import Path
import config
from src.preprocess.network_generation import networks

NETWORK_ROOT = Path("/absolute/path/to/network-work/comments_submissions/networks")

for threshold in (0, 2):
    networks.process_years(start=2006, end=2019, threshold=threshold)

    monthly_root = Path(config.combined_path)
    if threshold:
        monthly_root = monthly_root / "filtered"

    output_dir = NETWORK_ROOT / ("filtered" if threshold else "full")
    output_dir.mkdir(parents=True, exist_ok=True)

    networks.get_networks(
        name="_2" if threshold else "",
        data_path=str(monthly_root) + "/",
        start=2006,
        end=2019,
        networks_path=str(output_dir) + "/",
    )
PY
```

Set `NETWORK_ROOT` to the location used by the analysis notebooks. The script skips unreadable shards and missing monthly pickle files, so check input coverage before interpreting an annual graph.


## 9. Handoff to analysis

**Notebooks:** [`src/analysis/`](../analysis/), opened with that folder as the working directory.

**Input:** the products below

**Operation:** load memberships, join records/scores, estimate outcomes, and generate paper panels.

**Output:** the main figures mapped in the [root README](../../README.md#main-figures).

| Notebook | Required products / remaining alignment |
| --- | --- |
| `01a-descriptives` | Matching tables, joint counts, author-year counts, and step 6b's user counts. |
| `01b-networks` | Public membership CSV and four annual metric families from step 8 |
| `02a-growth` | Public membership CSV and five sets' community records from step 6a |
| `02b-ego-networks` | Public membership CSV and filtered graphs; adjust the hard-coded network root |
| `02c-gatekeepers` | Public membership CSV and step 4b's `filtered-data-new/...from-user...subset3274-lite` histories |
| `03a-toxicty-trends` | Public membership CSV, community extracts, cleaned single files, and scores from steps 6–7 |
| `03b-toxicity-user-change`, `03c-toxicity-stacked-did` | Public membership CSV, step 4b's user histories, within-B metadata/scores, and neighborhood metadata/scores from steps 6c and 7c. |

