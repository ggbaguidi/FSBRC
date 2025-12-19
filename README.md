# FSBRC — Farm to Feed Shopping Basket Recommender (Zindi)

This repository contains a **CPU-only, 16GB-RAM-friendly** baseline solution for the [Farm to Feed Shopping Basket Recommendation Challenge](https://zindi.africa/competitions/farm-to-feed-shopping-basket-recommendation-challenge) on Zindi.

The task is to predict, for each `(customer, SKU, week)` in **Test**, both:

- **Purchase probability** in the next **1 week** and **2 weeks**
- **Expected quantity** in the next **1 week** and **2 weeks**

It trains 4 LightGBM models (2 classifiers + 2 regressors) and produces a submission CSV in the format expected by Zindi.

---

## 1) Problem summary

Each row is a weekly record for a `(customer_id, product_unit_variant_id, week_start)` pair.

Targets (train only):

- `Target_purchase_next_1w` (binary 0/1)
- `Target_qty_next_1w` (float ≥ 0)
- `Target_purchase_next_2w` (binary 0/1)
- `Target_qty_next_2w` (float ≥ 0)

Submission columns:

- `ID`
- `Target_purchase_next_1w`
- `Target_qty_next_1w`
- `Target_purchase_next_2w`
- `Target_qty_next_2w`

Evaluation (leaderboard): **50% AUC + 50% MAE**.

Important rule: **No manual thresholding** (no fixed cut-off like “if p > 0.3 then buy”). We always output probabilities.

---

## 2) Repository layout

- `data/`
  - `Train.csv`, `Test.csv`, `SampleSubmission.csv`
  - `customer_data.csv`, `sku_data.csv`
  - `variable_description.pdf`
- `src/`
  - `profile_data.py` — quick dataset profiling (counts, ranges, cardinalities)
  - `features.py` — feature engineering for train and test
  - `train.py` — model training + time-split validation + saving artifacts
  - `predict.py` — inference + submission generation
  - `cache_features.py` — optional Parquet caching for faster iteration
- `artifacts/` (generated)
  - `lgb_purchase_1w.txt`, `lgb_purchase_2w.txt`
  - `lgb_qty_1w.txt`, `lgb_qty_2w.txt`
  - optional: `lgb_qty_cond_1w.txt`, `lgb_qty_cond_2w.txt`
  - `meta.json`, `report.json`
  - optional cache: `features_train.parquet`

---

## 3) What this solution does (exactly)

### 3.1 Feature engineering (`src/features.py`)

This solution builds **history-only** features that avoid leakage and work with Test.

Key point: `Test.csv` does **not** contain same-week transactional behavior columns like `qty_this_week`, `spend_this_week`, etc. Therefore:

- In training, we create history features using **lagged / rolling** stats.
- In inference, we attach the **latest available history** for the same `(customer, SKU)` before the test week.

#### A) Customer×SKU history features
All computed per `(customer_id, product_unit_variant_id)` in time order:

- Lags (previous week only):
  - `lag1_qty`, `lag1_orders`, `lag1_spend`, `lag1_purchased`
- Rolling sums on past weeks (all shifted by 1 week):
  - `roll{2,4,8,16}_qty_sum`
  - `roll{2,4,8,16}_purch_cnt`
  - `roll{2,4,8,16}_spend_sum`
- Recency:
  - `last_purchase_week_index` (last week index where purchased)
  - `weeks_since_last_purchase`

**Leakage control**: all history-based behavior features use `.shift(1)` so the model never uses the current week’s behavior to predict next week.

#### B) Customer-week aggregates (shifted)
Aggregated per `(customer_id, week_start)` and shifted by 1 week:

- total qty/spend/orders across all SKUs for that customer
- number of SKUs purchased that week

#### C) SKU-week aggregates (shifted)
Aggregated per `(product_unit_variant_id, week_start)` and shifted by 1 week:

- SKU total qty/spend
- number of customer purchases
- number of active customers for the SKU

#### D) Static / slowly changing
- `customer_age_weeks` computed from `customer_created_at` and `week_start`

### 3.2 Test-time “point-in-time” join

`build_test_features()` uses a **point-in-time (as-of) join**:

- For each test row `(customer, sku, test_week)`, it finds the **latest history row** with the same `(customer, sku)` and `history_week <= test_week`.
- The history timestamp is kept as `week_start_hist`.
- A gap feature is computed:
  - `weeks_since_hist = (test_week - week_start_hist) / 7`

Then we recompute:

- `week_index` from **test** `week_start`
- `weeks_since_last_purchase` using **test** `week_index - last_purchase_week_index`

This makes time features consistent even when Test weeks are in the future.

### 3.3 Modeling (`src/train.py`)

We train 4 models with LightGBM:

- Purchase probability (classification):
  - `lgb_purchase_1w.txt` predicts `Target_purchase_next_1w`
  - `lgb_purchase_2w.txt` predicts `Target_purchase_next_2w`
  - Uses `objective=binary`, `metric=auc`
  - Uses `scale_pos_weight = (#neg / #pos)` to handle class imbalance

- Quantity (regression for MAE):
  - `lgb_qty_1w.txt` predicts `Target_qty_next_1w`
  - `lgb_qty_2w.txt` predicts `Target_qty_next_2w`
  - Uses `objective=regression_l1`, `metric=l1` (MAE-friendly)

Validation:

- Time split: last `--val-weeks` unique weeks are validation.
- The script writes:
  - `artifacts/report.json` (metrics)
  - `artifacts/meta.json` (feature names + categorical columns)

### 3.4 Inference + submission (`src/predict.py`)

- Builds test features using `build_test_features()`.
- Aligns feature columns to `meta.json` (important for consistent LightGBM input).
- Predicts:
  - probabilities are clipped to `(1e-6, 1-1e-6)` for numerical safety
  - quantities are clipped to `>= 0`
- Writes a submission CSV.

---

## 4) Setup

Create and activate a Python environment (venv recommended), then install dependencies:

- `polars`
- `pyarrow`
- `lightgbm`
- `scikit-learn`
- `pandas`, `numpy`

(Exact versions depend on your system; any modern versions should work.)

---

## 5) Quickstart

### 5.1 Profile the dataset (optional)

Run:

- `python src/profile_data.py`

This prints row counts, date ranges, key cardinalities, and null rates.

### 5.2 Train models

Run:

- `python src/train.py --val-weeks 6`

Optional (better validation + small ensemble):

- `python src/train.py --cv-folds 3 --val-weeks 6`

This trains 3 rolling time folds and saves `lgb_*_fold*.txt` models; `predict.py` will automatically average them.

Outputs:

- LightGBM model files in `artifacts/`
- `artifacts/meta.json` and `artifacts/report.json`

Optional:

- `--two-stage-qty` trains conditional quantity models as well (kept for experimentation).
- `--qty-objective` lets you experiment with alternative LightGBM objectives for quantity (default: `regression_l1`).

### 5.3 Generate a submission

Run:

- `python src/predict.py --out submission.csv`

This writes `submission.csv` in the repo root by default.

### 5.4 End-to-end (train -> predict -> submission)

Minimal (rebuild features from `data/Train.csv` each time):

- `python src/train.py --val-weeks 6`
- `python src/predict.py --out submission.csv`

Faster iteration (recommended):

- `python src/cache_features.py`
- `python src/train.py --use-cache --val-weeks 6`
- `python src/predict.py --out submission_cached.csv`

Notes:

- `predict.py` expects `artifacts/meta.json` and the model files in `artifacts/` to exist (created by `train.py`).
- If `artifacts/features_train.parquet` exists, `predict.py` will automatically use it as the history source.

---

## 6) Faster iteration with caching (recommended)

Feature generation from CSV can be the slowest step on CPU. To speed up repeated experiments:

1) Build caches once:

- `python src/cache_features.py`

This writes:

- `artifacts/features_train.parquet`

2) Train using the cache:

- `python src/train.py --use-cache`

3) Predict using the cache:

- `python src/predict.py --out submission_cached.csv`

`predict.py` will automatically use `artifacts/features_train.parquet` if present.

---

## 7) Notes on correctness and leakage

- All history features are shifted by 1 week.
- Test-time features are built using a backward as-of join: only history with `week_start <= test_week` is used.
- No probability thresholding is applied.

---

## 8) Common troubleshooting

- **Polars panic / rolling on Int8**: We cast `purchased_this_week` to `Int32` for rolling sums.
- **LightGBM + pandas Arrow dtypes error**: We convert Polars to pandas with NumPy dtypes (`use_pyarrow_extension_array=False`).
- **Missing columns in Test (e.g., selling_price)**: Feature casting is guarded; absent columns are skipped.

---

## License

This code is provided as-is for competition use.
