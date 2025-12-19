from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import asdict

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, roc_auc_score

from features import FeatureConfig, build_train_features, scan_train, to_pandas_for_lgbm


ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"


CATEGORICAL_COLS = [
    "grade_name",
    "unit_name",
    "customer_category",
    "customer_status",
]


def _time_split(df: pl.DataFrame, val_weeks: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Use last N unique weeks for validation.
    weeks = df.select(pl.col("week_start").unique().sort()).to_series().to_list()
    if len(weeks) <= val_weeks + 2:
        raise ValueError(f"Not enough weeks for time split: {len(weeks)}")
    cut = weeks[-val_weeks]
    train_df = df.filter(pl.col("week_start") < cut)
    val_df = df.filter(pl.col("week_start") >= cut)
    return train_df, val_df


def _train_lgbm(
    X_train,
    y_train,
    X_val,
    y_val,
    feature_names: list[str],
    cat_idx: list[int],
    objective: str,
    metric: str,
    seed: int,
    extra_params: dict | None = None,
) -> tuple[lgb.Booster, dict]:
    params = {
        "objective": objective,
        "metric": metric,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 255,
        "verbosity": -1,
        "seed": seed,
        "num_threads": 4,
    }
    if extra_params:
        params.update(extra_params)

    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=feature_names,
        categorical_feature=cat_idx if cat_idx else "auto",
        free_raw_data=False,
    )

    has_val = X_val is not None and len(y_val) > 0
    if has_val:
        dval = lgb.Dataset(
            X_val,
            label=y_val,
            feature_name=feature_names,
            categorical_feature=cat_idx if cat_idx else "auto",
            free_raw_data=False,
        )
        booster = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            num_boost_round=5000,
            callbacks=[lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(period=100)],
        )

        pred_val = booster.predict(X_val)
        if objective == "binary":
            score = float(roc_auc_score(y_val, pred_val))
            return booster, {"val_auc": score, "best_iteration": booster.best_iteration}

        score = float(mean_absolute_error(y_val, pred_val))
        return booster, {"val_mae": score, "best_iteration": booster.best_iteration}

    # No validation set available: train without early stopping
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        callbacks=[lgb.log_evaluation(period=100)],
    )
    return booster, {"note": "trained_without_validation", "best_iteration": booster.current_iteration()}


def _scale_pos_weight(y: np.ndarray) -> float:
    # pos_weight = n_negative / n_positive
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    if pos <= 0:
        return 1.0
    return max(1.0, neg / pos)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-weeks", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--two-stage-qty",
        action="store_true",
        help="Train extra quantity models conditional on purchase and combine with purchase probability.",
    )
    ap.add_argument(
        "--use-cache",
        action="store_true",
        help="If artifacts/features_train.parquet exists, load it instead of rebuilding features from Train.csv.",
    )
    args = ap.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)

    cfg = FeatureConfig()

    cache_path = ARTIFACTS / "features_train.parquet"
    if args.use_cache and cache_path.exists():
        feat_df = pl.scan_parquet(cache_path).collect(streaming=True)
    else:
        train_lf = scan_train(DATA / "Train.csv")
        feat_lf = build_train_features(train_lf, cfg)
        # Materialize once. Keep only needed cols to reduce RAM.
        feat_df = feat_lf.collect(streaming=True)

    train_df, val_df = _time_split(feat_df, val_weeks=args.val_weeks)

    # Convert each to pandas
    X_train, feature_names, cat_idx, _ = to_pandas_for_lgbm(train_df, categorical=CATEGORICAL_COLS)
    X_val, _, _, _ = to_pandas_for_lgbm(val_df, categorical=CATEGORICAL_COLS)

    report: dict[str, dict] = {"config": asdict(cfg), "val_weeks": args.val_weeks}

    # 1w purchase
    ytr = train_df["Target_purchase_next_1w"].to_numpy()
    yva = val_df["Target_purchase_next_1w"].to_numpy()
    spw_1w = _scale_pos_weight(ytr)
    m1, r1 = _train_lgbm(
        X_train,
        ytr,
        X_val,
        yva,
        feature_names,
        cat_idx,
        objective="binary",
        metric="auc",
        seed=args.seed,
        extra_params={"scale_pos_weight": spw_1w},
    )
    m1.save_model(str(ARTIFACTS / "lgb_purchase_1w.txt"))
    report["purchase_1w"] = {**r1, "scale_pos_weight": spw_1w}

    # 2w purchase
    ytr = train_df["Target_purchase_next_2w"].to_numpy()
    yva = val_df["Target_purchase_next_2w"].to_numpy()
    spw_2w = _scale_pos_weight(ytr)
    m2, r2 = _train_lgbm(
        X_train,
        ytr,
        X_val,
        yva,
        feature_names,
        cat_idx,
        objective="binary",
        metric="auc",
        seed=args.seed + 1,
        extra_params={"scale_pos_weight": spw_2w},
    )
    m2.save_model(str(ARTIFACTS / "lgb_purchase_2w.txt"))
    report["purchase_2w"] = {**r2, "scale_pos_weight": spw_2w}

    # 1w qty (MAE-friendly objective)
    ytr = train_df["Target_qty_next_1w"].to_numpy()
    yva = val_df["Target_qty_next_1w"].to_numpy()
    m3, r3 = _train_lgbm(
        X_train,
        ytr,
        X_val,
        yva,
        feature_names,
        cat_idx,
        objective="regression_l1",
        metric="l1",
        seed=args.seed + 2,
    )
    m3.save_model(str(ARTIFACTS / "lgb_qty_1w.txt"))
    report["qty_1w"] = r3

    # 2w qty
    ytr = train_df["Target_qty_next_2w"].to_numpy()
    yva = val_df["Target_qty_next_2w"].to_numpy()
    m4, r4 = _train_lgbm(
        X_train,
        ytr,
        X_val,
        yva,
        feature_names,
        cat_idx,
        objective="regression_l1",
        metric="l1",
        seed=args.seed + 3,
    )
    m4.save_model(str(ARTIFACTS / "lgb_qty_2w.txt"))
    report["qty_2w"] = r4

    # Optional: two-stage quantity modeling
    #   q_hat = p_hat * q_hat_cond
    # where q_hat_cond is trained on rows with purchase==1, using log1p(target_qty).
    two_stage_enabled = bool(args.two_stage_qty)
    use_two_stage_1w = False
    use_two_stage_2w = False
    if two_stage_enabled:
        report["qty_two_stage"] = {}

        # Precompute purchase predictions on validation for combination
        p1_val = m1.predict(X_val)
        p2_val = m2.predict(X_val)

        # 1w conditional quantity model
        ytr_p = train_df["Target_purchase_next_1w"].to_numpy()
        yva_qty = val_df["Target_qty_next_1w"].to_numpy()
        mask_tr = ytr_p == 1
        if np.any(mask_tr):
            ytr_qty = train_df["Target_qty_next_1w"].to_numpy()[mask_tr]
            Xtr_pos = X_train.iloc[np.nonzero(mask_tr)[0]]
            ytr_log = np.log1p(ytr_qty)

            # For validation, we evaluate on the full val set after combining with p_hat.
            # But the conditional model is trained only on positive purchases.
            mask_va = (val_df["Target_purchase_next_1w"].to_numpy() == 1)
            Xva_pos = X_val.iloc[np.nonzero(mask_va)[0]] if np.any(mask_va) else X_val.iloc[:0]
            yva_pos_log = np.log1p(yva_qty[mask_va]) if np.any(mask_va) else np.array([])

            m3c, r3c = _train_lgbm(
                Xtr_pos,
                ytr_log,
                Xva_pos,
                yva_pos_log,
                feature_names,
                cat_idx,
                objective="regression_l1",
                metric="l1",
                seed=args.seed + 12,
            )
            m3c.save_model(str(ARTIFACTS / "lgb_qty_cond_1w.txt"))
            q1c_val = np.expm1(m3c.predict(X_val))
            q1_two_stage = np.clip(p1_val * q1c_val, 0.0, None)
            mae_two_stage_1w = float(mean_absolute_error(yva_qty, q1_two_stage))
            report["qty_two_stage"]["qty_1w"] = {
                **r3c,
                "val_mae_two_stage": mae_two_stage_1w,
            }
            # Choose best method based on overall MAE on the same validation split
            use_two_stage_1w = mae_two_stage_1w < float(report["qty_1w"]["val_mae"])

        # 2w conditional quantity model
        ytr_p = train_df["Target_purchase_next_2w"].to_numpy()
        yva_qty = val_df["Target_qty_next_2w"].to_numpy()
        mask_tr = ytr_p == 1
        if np.any(mask_tr):
            ytr_qty = train_df["Target_qty_next_2w"].to_numpy()[mask_tr]
            Xtr_pos = X_train.iloc[np.nonzero(mask_tr)[0]]
            ytr_log = np.log1p(ytr_qty)

            mask_va = (val_df["Target_purchase_next_2w"].to_numpy() == 1)
            Xva_pos = X_val.iloc[np.nonzero(mask_va)[0]] if np.any(mask_va) else X_val.iloc[:0]
            yva_pos_log = np.log1p(yva_qty[mask_va]) if np.any(mask_va) else np.array([])

            m4c, r4c = _train_lgbm(
                Xtr_pos,
                ytr_log,
                Xva_pos,
                yva_pos_log,
                feature_names,
                cat_idx,
                objective="regression_l1",
                metric="l1",
                seed=args.seed + 13,
            )
            m4c.save_model(str(ARTIFACTS / "lgb_qty_cond_2w.txt"))
            q2c_val = np.expm1(m4c.predict(X_val))
            q2_two_stage = np.clip(p2_val * q2c_val, 0.0, None)
            mae_two_stage_2w = float(mean_absolute_error(yva_qty, q2_two_stage))
            report["qty_two_stage"]["qty_2w"] = {
                **r4c,
                "val_mae_two_stage": mae_two_stage_2w,
            }
            use_two_stage_2w = mae_two_stage_2w < float(report["qty_2w"]["val_mae"])

    # Save metadata for inference
    meta = {
        "feature_names": feature_names,
        "categorical_cols": CATEGORICAL_COLS,
        "config": asdict(cfg),
        "qty_two_stage_trained": bool(args.two_stage_qty),
        # Automatically pick the better approach per horizon.
        "qty_use_two_stage_1w": use_two_stage_1w,
        "qty_use_two_stage_2w": use_two_stage_2w,
        "qty_cond_log1p": True,
    }
    (ARTIFACTS / "meta.json").write_text(json.dumps(meta, indent=2))
    (ARTIFACTS / "report.json").write_text(json.dumps(report, indent=2))

    print("\n## Validation report")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
