# Online Harm Proliferated for Years Prior to Content Moderation Takedowns


Project repository for the paper "Online Harm Proliferated for Years Prior to Content Moderation Takedowns" by Yijing Chen, Lluís Danús, Yphtach Lelkes and Sandra González-Bailón.


## Overall pipeline

```text
Pushshift Reddit archive -> year/month-partitioned Parquet
    -> count activity by subreddit (sets)
    -> identify banned subreddits (B) and extract historical activity
    -> measure user overlap and build comparison sets
    -> aggregate submissions and score toxicity
    -> main analysis
```

| Goal | Key operation and audit trail |
| --- | --- |
| **1. Count activity** | [`src/preprocess/counting/`](src/preprocess/counting/) produces subreddit-year and author-subreddit-year counts for submissions/comments, with first/last activity dates. Cumulative users are deduplicated across years and across communities within each set. |
| **2. Identify $B$** | [`src/preprocess/collecting/`](src/preprocess/collecting/) screens 2020 activity, checks subsequent inactivity through 2022, and classifies saved homepage-status responses with manual corrections. Exclude communities born in 2020. |
| **3. Build baselines** | [`src/preprocess/matching/`](src/preprocess/matching/) considers lifespan, cumulative participants, and shared-user counts in the matching process; applies coarsened exact matching (CEM) in R; and defines an unmatched random comparison. |
| **4. Extract records** | [`src/preprocess/filtering/`](src/preprocess/filtering/) filters submissions or comments by subreddit, author, or submission ID. |
| **5. Measure toxicity** | [`src/preprocess/labeling/`](src/preprocess/labeling/) joins submission title/body to comments, cleans text, and scores threads with Detoxify's multilingual model or OpenAI's moderation endpoint; and checks output alignement across models. |
| **6. Estimate and plot** | [`src/analysis/`](src/analysis/) produces the panels mapped below. Supplementary checks and exploratory alternatives coexist with the main sections. |


## Main figures

| Paper panels | Notebook | Main measurement |
| --- | --- | --- |
| Fig. 1 | [00: literature review](src/analysis/00-lit-review-vis.ipynb) | -- |
| Fig. 2A–B | [01a: descriptives](src/analysis/01a-descriptives.ipynb) | Cumulative activity; Kaplan–Meier retention with a 300-day inactivity window |
| Fig. 2D | [01b: networks](src/analysis/01b-networks.ipynb) | 2019 weighted strength and PageRank distributions |
| Fig. 3A–C | [02a: growth](src/analysis/02a-growth.ipynb) | Monthly activity; sensitivity to excluding r/The_Donald |
| Fig. 3D | [02b: ego network](src/analysis/02b-ego-networks.ipynb) | Annual counts of all/banned neighbors of r/The_Donald |
| Fig. 3E | [02c: gatekeepers](src/analysis/02c-gatekeepers.ipynb) | Annual Poisson Pearson residuals for first-entry prominence; 1,000 user bootstraps |
| Fig. 4A | [03a: toxicity trends](src/analysis/03a-toxicty-trends.ipynb) | Weekly means over scored threads, sampling up to 50 submissions per subreddit-week |
| Fig. 4B | [03b: user changes](src/analysis/03b-toxicity-user-level-change.ipynb) | Paired mean toxicity before/after first entry, nominal five-year windows; 5,000 user bootstraps |
| Fig. 4C | [03c: stacked DiD](src/analysis/03c-toxicity-stacked-did.ipynb) | Effects of having sustained engagement in $B$ |

Fig. 2C's network schematic is not generated here.


## Dataset and availabilities

We release the list of banned subreddits and comparison sets in `data-public/subreddi_sets.csv`. 
[TODO: discuss whether/how we want to release other preprocessed datasets such as activity counts and toxicity scores.]


## Preprocessing pipeline

```text
Raw Reddit data
  -> Count activity and identify banned subreddits
  -> Build matched and random comparison sets
  -> Extract community records and user histories
  -> Prepare cumulative / set counts, networks, and toxicity scores
```

For a step-by-step guide, please go to [preprocessing README](src/preprocess/README.md). 

## Data availability and privacy concern

### Source data availability

We obtained the historical [Pushshift Reddit archive](https://arxiv.org/abs/2001.08435) by 2024, through its public listings on Academic Torrents. In summer 2026, the archive maintainer reported taking down those listings at Reddit's request; see [takedown announcement](https://www.reddit.com/r/pushshift/comments/1v50ved/).

Ou server stores the raw Reddit submission and comment archive during **2005–2022**. Historical analyses primarily use activity through 2019; identifying banned communities also uses activity from 2020–2022. 

According to Pushshift’s [data-access policy](https://api.pushshift.io/signup), we do not redistribute the bulk raw archive is in this repository. Instead, we provide:

- **Community membership:** [data-public/subreddit_sets.csv](data-public/subreddit_sets.csv) contains the complete banned and baseline subreddit sets used in the analysis.
- **Matching tables:** [data-public/snippets/samples/cem-batch4-new/](data-public/snippets/samples/cem-batch4-new/) contains community covariates and matching assignments for the CEM specifications.
- **Illustrative data excerpts:** [data-public/snippets/counts-raw/](data-public/snippets/counts-raw/) contains randomly sampled activity and participant-count tables, including subreddit-level statistics, author–subreddit–year statistics, and set-level user counts. 
  - *The accompanying [snippet-manifest.csv](data-public/snippets/counts-raw/snippet-manifest.csv) records source and excerpt sizes and the columns pseudonymized.*

These materials support inspection and reuse of the study’s community definitions, data structures, and analytical methods. Reproducing the full analysis additionally requires the raw archive and intermediate inputs described in the [preprocessing guide](src/preprocess/README.md); the illustrative excerpts alone are insufficient to reproduce the paper’s results.

### Data privacy

The released subreddit-set CSV contains subreddit names and their set membership, with no individual user records. In the illustrative data snippets, usernames are replaced with placeholders and personally identifiable information is removed.

Raw data and intermediate datasets are stored on a server at the Center for Information Networks and Democracy at the University of Pennsylvania. Access is restricted to authorized researchers.


## Installation and reproduction

With the exported `environment.yml` at the repository root:

```bash
conda env create --name reddit-ban-2020 --file environment.yml
conda activate reddit-ban-2020
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```
