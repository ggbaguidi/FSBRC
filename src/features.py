from __future__ import annotations

import dataclasses
import pathlib
from typing import Iterable

import numpy as np
import polars as pl


@dataclasses.dataclass(frozen=True)
class FeatureConfig:
    # Customer×SKU history windows (in weeks)
    windows: tuple[int, ...] = (2, 4, 8, 16)


STATIC_COLS = [
    "customer_id",
    "product_unit_variant_id",
    "product_id",
    "product_grade_variant_id",
    "grade_name",
    "unit_name",
    "selling_price",
    "customer_category",
    "customer_status",
    "customer_created_at",
]

BEHAV_COLS = [
    "qty_this_week",
    "num_orders_week",
    "spend_this_week",
    "purchased_this_week",
]

TARGET_COLS = [
    "Target_purchase_next_1w",
    "Target_qty_next_1w",
    "Target_purchase_next_2w",
    "Target_qty_next_2w",
]


def scan_train(path: pathlib.Path) -> pl.LazyFrame:
    return pl.scan_csv(
        path,
        infer_schema_length=5_000,
        ignore_errors=True,
        try_parse_dates=True,
        low_memory=True,
    )


def scan_test(path: pathlib.Path) -> pl.LazyFrame:
    return pl.scan_csv(
        path,
        infer_schema_length=5_000,
        ignore_errors=True,
        try_parse_dates=True,
        low_memory=True,
    )


def _week_index_expr(col: str = "week_start") -> pl.Expr:
    # Integer week index to make time deltas cheap.
    # week_start is a Monday; we convert to days since epoch and divide by 7.
    return (pl.col(col).cast(pl.Date).cast(pl.Int32) // 7).alias("week_index")


def _customer_age_weeks() -> pl.Expr:
    # customer_created_at may be datetime string; we cast to Datetime then to Date.
    created = pl.col("customer_created_at").cast(pl.Datetime, strict=False).dt.date()
    ws = pl.col("week_start").cast(pl.Date)
    return ((ws - created).dt.total_days() / 7.0).alias("customer_age_weeks")


def _add_customer_week_aggregates(lf: pl.LazyFrame, cfg: FeatureConfig) -> pl.LazyFrame:
    # Aggregates per (customer_id, week_start). These are based on that week; we will shift by 1
    # to avoid using same-week behavior (not present in Test).
    cust_week = (
        lf.group_by(["customer_id", "week_start"], maintain_order=True)
        .agg(
            pl.col("qty_this_week").sum().alias("cust_qty_sum_w"),
            pl.col("spend_this_week").sum().alias("cust_spend_sum_w"),
            pl.col("num_orders_week").sum().alias("cust_orders_sum_w"),
            pl.col("purchased_this_week").sum().alias("cust_sku_purchased_cnt_w"),
        )
        .sort(["customer_id", "week_start"])
        .with_columns(_week_index_expr())
        .with_columns(
            [
                pl.col("cust_qty_sum_w").shift(1).over("customer_id").alias("cust_qty_sum_lag1"),
                pl.col("cust_spend_sum_w").shift(1).over("customer_id").alias("cust_spend_sum_lag1"),
                pl.col("cust_orders_sum_w").shift(1).over("customer_id").alias("cust_orders_sum_lag1"),
                pl.col("cust_sku_purchased_cnt_w").shift(1)
                .over("customer_id")
                .alias("cust_sku_purchased_cnt_lag1"),
            ]
        )
    )

    out = lf.join(cust_week.select(["customer_id", "week_start"] + [c for c in cust_week.collect_schema().names() if c.endswith("lag1")]),
                  on=["customer_id", "week_start"], how="left")
    return out


def _add_sku_week_aggregates(lf: pl.LazyFrame, cfg: FeatureConfig) -> pl.LazyFrame:
    sku_week = (
        lf.group_by(["product_unit_variant_id", "week_start"], maintain_order=True)
        .agg(
            pl.col("qty_this_week").sum().alias("sku_qty_sum_w"),
            pl.col("spend_this_week").sum().alias("sku_spend_sum_w"),
            pl.col("purchased_this_week").sum().alias("sku_cust_purchase_cnt_w"),
            pl.col("customer_id").n_unique().alias("sku_cust_active_cnt_w"),
        )
        .sort(["product_unit_variant_id", "week_start"])
        .with_columns(
            [
                pl.col("sku_qty_sum_w").shift(1).over("product_unit_variant_id").alias("sku_qty_sum_lag1"),
                pl.col("sku_spend_sum_w").shift(1).over("product_unit_variant_id").alias("sku_spend_sum_lag1"),
                pl.col("sku_cust_purchase_cnt_w").shift(1)
                .over("product_unit_variant_id")
                .alias("sku_cust_purchase_cnt_lag1"),
                pl.col("sku_cust_active_cnt_w").shift(1)
                .over("product_unit_variant_id")
                .alias("sku_cust_active_cnt_lag1"),
            ]
        )
    )
    out = lf.join(
        sku_week.select(["product_unit_variant_id", "week_start"] + [c for c in sku_week.collect_schema().names() if c.endswith("lag1")]),
        on=["product_unit_variant_id", "week_start"],
        how="left",
    )
    return out


def _add_cust_sku_history_features(lf: pl.LazyFrame, cfg: FeatureConfig) -> pl.LazyFrame:
    # These features are built from behavior columns, but ALWAYS shifted by 1 week so that
    # the model never uses same-week behavior (not present in Test).
    key = ["customer_id", "product_unit_variant_id"]

    lf = lf.sort(key + ["week_start"])

    # Lag-1 features
    lf = lf.with_columns(
        [
            pl.col("qty_this_week").shift(1).over(key).alias("lag1_qty"),
            pl.col("num_orders_week").shift(1).over(key).alias("lag1_orders"),
            pl.col("spend_this_week").shift(1).over(key).alias("lag1_spend"),
            pl.col("purchased_this_week").shift(1).over(key).alias("lag1_purchased"),
        ]
    )

    # Rolling windows on shifted series (only past weeks)
    for w in cfg.windows:
        lf = lf.with_columns(
            [
                pl.col("qty_this_week")
                .shift(1)
                .rolling_sum(window_size=w, min_periods=1)
                .over(key)
                .alias(f"roll{w}_qty_sum"),
                pl.col("purchased_this_week")
                .cast(pl.Int32)
                .shift(1)
                .rolling_sum(window_size=w, min_periods=1)
                .over(key)
                .alias(f"roll{w}_purch_cnt"),
                pl.col("spend_this_week")
                .shift(1)
                .rolling_sum(window_size=w, min_periods=1)
                .over(key)
                .alias(f"roll{w}_spend_sum"),
            ]
        )

    # Recency in weeks since last purchase (based on past weeks)
    lf = lf.with_columns(_week_index_expr())
    last_purchase_week = (
        pl.when(pl.col("purchased_this_week") == 1)
        .then(pl.col("week_index"))
        .otherwise(None)
        .forward_fill()
        .shift(1)
        .over(key)
        .alias("last_purchase_week_index")
    )
    lf = lf.with_columns(last_purchase_week)
    lf = lf.with_columns(
        (
            (pl.col("week_index") - pl.col("last_purchase_week_index"))
            .cast(pl.Float32)
            .fill_null(999.0)
            .alias("weeks_since_last_purchase")
        )
    )

    return lf


def _safe_ratio(num: pl.Expr, den: pl.Expr, *, eps: float = 1e-3, name: str) -> pl.Expr:
    """Safe ratio feature: returns 0 when denominator is null/<=0.

    We intentionally keep this simple and deterministic for LightGBM.
    """

    den_f = den.cast(pl.Float32).fill_null(0.0)
    num_f = num.cast(pl.Float32).fill_null(0.0)
    return (
        pl.when(den_f > 0)
        .then(num_f / (den_f + pl.lit(eps, dtype=pl.Float32)))
        .otherwise(pl.lit(0.0, dtype=pl.Float32))
        .alias(name)
    )


def _add_affinity_ratio_features(lf: pl.LazyFrame, cfg: FeatureConfig) -> pl.LazyFrame:
    """Add customer-SKU affinity ratios and simple rates.

    These are computed only from already leakage-safe inputs (lagged / shifted aggregates).
    """

    cols = set(lf.collect_schema().names())
    out_exprs: list[pl.Expr] = []

    # Shares vs customer previous-week totals
    if {"lag1_qty", "cust_qty_sum_lag1"}.issubset(cols):
        out_exprs.append(_safe_ratio(pl.col("lag1_qty"), pl.col("cust_qty_sum_lag1"), name="aff_lag1_qty_cust_share"))
    if {"lag1_spend", "cust_spend_sum_lag1"}.issubset(cols):
        out_exprs.append(
            _safe_ratio(pl.col("lag1_spend"), pl.col("cust_spend_sum_lag1"), name="aff_lag1_spend_cust_share")
        )

    # Shares vs SKU previous-week totals
    if {"lag1_qty", "sku_qty_sum_lag1"}.issubset(cols):
        out_exprs.append(_safe_ratio(pl.col("lag1_qty"), pl.col("sku_qty_sum_lag1"), name="aff_lag1_qty_sku_share"))
    if {"lag1_spend", "sku_spend_sum_lag1"}.issubset(cols):
        out_exprs.append(_safe_ratio(pl.col("lag1_spend"), pl.col("sku_spend_sum_lag1"), name="aff_lag1_spend_sku_share"))

    # Rolling purchase rate per window (counts are already shifted)
    for w in cfg.windows:
        c_cnt = f"roll{w}_purch_cnt"
        if c_cnt in cols:
            out_exprs.append((pl.col(c_cnt).cast(pl.Float32) / pl.lit(float(w), dtype=pl.Float32)).alias(f"roll{w}_purch_rate"))

    # Rolling quantity shares vs customer / sku totals (using previous-week totals as stabilizers)
    for w in cfg.windows:
        c_qty = f"roll{w}_qty_sum"
        if c_qty in cols and "cust_qty_sum_lag1" in cols:
            out_exprs.append(_safe_ratio(pl.col(c_qty), pl.col("cust_qty_sum_lag1"), name=f"aff_roll{w}_qty_cust_share"))
        if c_qty in cols and "sku_qty_sum_lag1" in cols:
            out_exprs.append(_safe_ratio(pl.col(c_qty), pl.col("sku_qty_sum_lag1"), name=f"aff_roll{w}_qty_sku_share"))

    if out_exprs:
        lf = lf.with_columns(out_exprs)
    return lf


def _cast_and_basic(lf: pl.LazyFrame, keep_targets: bool) -> pl.LazyFrame:
    cols: list[str] = ["ID", "week_start"] + STATIC_COLS + BEHAV_COLS
    if keep_targets:
        cols += TARGET_COLS

    cols = [c for c in cols if c in lf.collect_schema().names()]
    lf = lf.select(cols)

    present = set(lf.collect_schema().names())

    # Basic casts
    cast_exprs: list[pl.Expr] = []
    if "week_start" in present:
        cast_exprs.append(pl.col("week_start").cast(pl.Date, strict=False))
    if "customer_id" in present:
        cast_exprs.append(pl.col("customer_id").cast(pl.Int32, strict=False))
    if "product_unit_variant_id" in present:
        cast_exprs.append(pl.col("product_unit_variant_id").cast(pl.Int32, strict=False))
    if "product_id" in present:
        cast_exprs.append(pl.col("product_id").cast(pl.Int32, strict=False))
    if "product_grade_variant_id" in present:
        cast_exprs.append(pl.col("product_grade_variant_id").cast(pl.Int32, strict=False))
    if "selling_price" in present:
        cast_exprs.append(pl.col("selling_price").cast(pl.Float32, strict=False))
    if "qty_this_week" in present:
        cast_exprs.append(pl.col("qty_this_week").cast(pl.Float32, strict=False))
    if "num_orders_week" in present:
        cast_exprs.append(pl.col("num_orders_week").cast(pl.Float32, strict=False))
    if "spend_this_week" in present:
        cast_exprs.append(pl.col("spend_this_week").cast(pl.Float32, strict=False))
    if "purchased_this_week" in present:
        # Keep as Int32 because Polars rolling window ops are not implemented for Int8.
        cast_exprs.append(pl.col("purchased_this_week").cast(pl.Int32, strict=False))

    if cast_exprs:
        lf = lf.with_columns(cast_exprs)

    if "customer_created_at" in present and "week_start" in present:
        lf = lf.with_columns(_customer_age_weeks())
    return lf


def build_train_features(train_lf: pl.LazyFrame, cfg: FeatureConfig) -> pl.LazyFrame:
    lf = _cast_and_basic(train_lf, keep_targets=True)
    lf = _add_cust_sku_history_features(lf, cfg)
    lf = _add_customer_week_aggregates(lf, cfg)
    lf = _add_sku_week_aggregates(lf, cfg)
    lf = _add_affinity_ratio_features(lf, cfg)

    # Cleanup: remove columns that won't exist in Test (same-week behavior)
    # Keep lag/rolling features.
    drop_now = [c for c in BEHAV_COLS if c in lf.collect_schema().names()]
    lf = lf.drop(drop_now)

    return lf


def build_test_features(
    train_hist_lf: pl.LazyFrame,
    test_lf: pl.LazyFrame,
    cfg: FeatureConfig,
) -> pl.LazyFrame:
    """Build features for Test using only history from Train.

    Since Test rows do not include same-week behavior fields, we create history features from Train
    and join the *latest available* history per (customer_id, product_unit_variant_id). Then we add
    a time-gap feature between that last history week and the requested test week.
    """

    # `train_hist_lf` can be either:
    #  - raw Train.csv scan (has BEHAV_COLS) => we must featurize first
    #  - cached/featurized history (no BEHAV_COLS) => use as-is
    train_cols = set(train_hist_lf.collect_schema().names())
    has_raw_behavior = any(c in train_cols for c in BEHAV_COLS)
    hist = build_train_features(train_hist_lf, cfg) if has_raw_behavior else train_hist_lf

    # Keys
    key = ["customer_id", "product_unit_variant_id"]
    required = set(key + ["week_start"])
    hist_cols_set = set(hist.collect_schema().names())
    if not required.issubset(hist_cols_set):
        raise ValueError(
            "train_hist_lf must include customer_id, product_unit_variant_id, and week_start. "
            "If you pass a cached feature table, ensure it still contains those keys."
        )

    # Select only the columns we actually want to carry from history.
    base_hist_cols = ["customer_id", "product_unit_variant_id", "week_start"]
    hist_cols = hist.collect_schema().names()
    extra_hist_cols = [
        c
        for c in hist_cols
        if c not in TARGET_COLS
        and c not in base_hist_cols
        # These are time-dependent and should be recomputed at the test week.
        and c not in ("week_index", "weeks_since_last_purchase")
    ]

    # Keep the history timestamp under a separate name so we can compute gaps.
    hist_state = (
        hist.select(base_hist_cols + extra_hist_cols)
        .with_columns(pl.col("week_start").alias("week_start_hist"))
        .drop("week_start")
        .sort(key + ["week_start_hist"])
    )

    test_base = _cast_and_basic(test_lf, keep_targets=False).drop(
        [c for c in BEHAV_COLS if c in test_lf.collect_schema().names()]
    )
    test_base = test_base.sort(key + ["week_start"])

    # Point-in-time join: for each (customer, sku, test_week), attach the latest history row
    # with week_start <= test_week.
    joined = test_base.join_asof(
        hist_state,
        left_on="week_start",
        right_on="week_start_hist",
        by=key,
        strategy="backward",
        suffix="_hist",
    )

    # weeks since last history week (if no history, set big)
    joined = joined.with_columns(
        ((pl.col("week_start") - pl.col("week_start_hist")).dt.total_days() / 7.0)
        .cast(pl.Float32)
        .fill_null(999.0)
        .alias("weeks_since_hist")
    )

    # Recompute week_index for the test week (consistent with training)
    joined = joined.with_columns(_week_index_expr())

    # Recompute recency based on last_purchase_week_index (from history) and the test week_index.
    if "last_purchase_week_index" in joined.collect_schema().names():
        joined = joined.with_columns(
            (
                (pl.col("week_index") - pl.col("last_purchase_week_index"))
                .cast(pl.Float32)
                .fill_null(999.0)
                .alias("weeks_since_last_purchase")
            )
        )

    # Replace missing history numeric features with 0
    schema = joined.collect_schema()
    num_cols: list[str] = []
    for name, dtype in schema.items():
        if name in ("ID", "week_start", "week_start_hist", "grade_name", "unit_name", "customer_category", "customer_status"):
            continue
        if dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            num_cols.append(name)

    joined = joined.with_columns([pl.col(c).fill_null(0) for c in num_cols])

    # Add ratio/rate features (must happen after joins so required columns are present)
    joined = _add_affinity_ratio_features(joined, cfg)
    return joined


def to_pandas_for_lgbm(df: pl.DataFrame, categorical: Iterable[str]):
    """Convert to pandas for LightGBM; returns X, feature_names, categorical_feature_indices, pandas_df."""
    # LightGBM (as of 4.x) does not accept pandas Arrow extension dtypes like `int32[pyarrow]`.
    # Use NumPy-backed dtypes.
    pdf = df.to_pandas(use_pyarrow_extension_array=False)

    # Ensure categoricals are pandas category dtype
    cat_cols = [c for c in categorical if c in pdf.columns]
    for c in cat_cols:
        pdf[c] = pdf[c].astype("category")

    # Drop non-features
    # Note: we drop raw customer_created_at; we keep `customer_age_weeks` which is numeric and safer.
    drop_cols = ["ID", "week_start", "week_start_hist", "customer_created_at"] + TARGET_COLS
    drop_cols = [c for c in drop_cols if c in pdf.columns]
    feature_df = pdf.drop(columns=drop_cols)

    # Also drop any remaining *_hist columns (these are join artifacts and can be object dtypes).
    hist_cols = [c for c in feature_df.columns if c.endswith("_hist")]
    if hist_cols:
        feature_df = feature_df.drop(columns=hist_cols)

    feature_names = list(feature_df.columns)
    cat_idx = [feature_names.index(c) for c in cat_cols if c in feature_names]

    return feature_df, feature_names, cat_idx, pdf
