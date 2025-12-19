from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import asdict
from typing import Any
import ast

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


def _rolling_week_blocks(weeks: np.ndarray, *, val_weeks: int, cv_folds: int) -> list[np.ndarray]:
    """Return validation week blocks for rolling time CV.

    Fold 0 is the most recent validation block, then fold 1 is the block before that, etc.
    Each block has exactly `val_weeks` unique weeks.
    """

    if cv_folds <= 0:
        raise ValueError("cv_folds must be >= 1")
    if val_weeks <= 0:
        raise ValueError("val_weeks must be >= 1")

    weeks = np.array(weeks)
    if weeks.ndim != 1:
        raise ValueError("weeks must be a 1D array")
    if len(weeks) <= val_weeks + 2:
        raise ValueError(f"Not enough weeks for time split: {len(weeks)}")

    blocks: list[np.ndarray] = []
    for fold in range(cv_folds):
        start = -(fold + 1) * val_weeks
        end = -fold * val_weeks if fold > 0 else None
        block = weeks[start:end]
        if len(block) != val_weeks:
            raise ValueError(
                f"Cannot create fold {fold}: expected {val_weeks} weeks but got {len(block)}. "
                f"Try reducing --cv-folds or --val-weeks."
            )
        blocks.append(block)

    # Ensure blocks are strictly ordered back in time
    for i in range(1, len(blocks)):
        if not (blocks[i][-1] < blocks[i - 1][0]):
            raise ValueError("Rolling CV blocks overlap or are not ordered; check week extraction.")
    return blocks


def _get_sorted_unique_weeks(pdf) -> np.ndarray:
    # Robustly coerce week_start to numpy datetime64[D] and return sorted unique weeks.
    week_arr = np.array(pdf["week_start"].values, dtype="datetime64[D]")
    return np.sort(np.unique(week_arr))


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


def _load_params_override(raw: str | None) -> dict[str, Any]:
    """Load a JSON dict from either a file path or an inline JSON string.

    Examples:
      --qty-params '{"objective":"huber","alpha":0.8}'
      --purchase-params artifacts/purchase_params.json
    """

    if not raw:
        return {}

    p = pathlib.Path(raw)
    def _parse_json(s: str) -> dict[str, Any]:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise ValueError("Params override must be a JSON object (dict)")
        return obj

    if p.exists() and p.is_file():
        data = _parse_json(p.read_text())
        return data

    # Try strict JSON first.
    try:
        return _parse_json(raw)
    except json.JSONDecodeError:
        # Common shell mistake: users pass {\"k\": 1} instead of {"k": 1}.
        # Retry by un-escaping quotes.
        try:
            fixed = raw.replace('\\"', '"')
            return _parse_json(fixed)
        except json.JSONDecodeError:
            # As a last resort, accept Python dict syntax.
            try:
                obj = ast.literal_eval(raw)
            except Exception as e:
                raise ValueError(
                    "Could not parse params override. Use valid JSON, e.g. "
                    "--purchase-params '{\"num_leaves\":127,\"min_data_in_leaf\":100}'."
                ) from e
            if not isinstance(obj, dict):
                raise ValueError("Params override must be a dict")
            return obj


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
    ap.add_argument(
        "--cv-folds",
        type=int,
        default=1,
        help="Number of rolling time folds to train and (optionally) ensemble. 1 keeps the original single-split behavior.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--qty-objective",
        type=str,
        default="regression_l1",
        help="LightGBM objective for quantity models (default: regression_l1). Examples: regression, huber, tweedie, poisson.",
    )
    ap.add_argument(
        "--purchase-params",
        type=str,
        default=None,
        help="JSON dict (inline or file path) to override LightGBM params for purchase models.",
    )
    ap.add_argument(
        "--qty-params",
        type=str,
        default=None,
        help="JSON dict (inline or file path) to override LightGBM params for quantity models.",
    )
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

    if args.cv_folds > 1 and args.two_stage_qty:
        raise ValueError("--two-stage-qty is not supported with --cv-folds > 1 (too many extra models).")

    ARTIFACTS.mkdir(exist_ok=True)

    cfg = FeatureConfig()

    purchase_params_override = _load_params_override(args.purchase_params)
    qty_params_override = _load_params_override(args.qty_params)

    cache_path = ARTIFACTS / "features_train.parquet"
    if args.use_cache and cache_path.exists():
        feat_df = pl.scan_parquet(cache_path).collect(streaming=True)
    else:
        train_lf = scan_train(DATA / "Train.csv")
        feat_lf = build_train_features(train_lf, cfg)
        # Materialize once. Keep only needed cols to reduce RAM.
        feat_df = feat_lf.collect(streaming=True)

    # Convert once to pandas; then do all splits as masks on the same arrays.
    X_all, feature_names, cat_idx, pdf_all = to_pandas_for_lgbm(feat_df, categorical=CATEGORICAL_COLS)
    weeks_sorted = _get_sorted_unique_weeks(pdf_all)
    val_blocks = _rolling_week_blocks(weeks_sorted, val_weeks=args.val_weeks, cv_folds=args.cv_folds)
    week_arr = np.array(pdf_all["week_start"].values, dtype="datetime64[D]")

    report: dict[str, dict] = {
        "config": asdict(cfg),
        "val_weeks": args.val_weeks,
        "cv_folds": args.cv_folds,
        "qty_objective": args.qty_objective,
        "purchase_params_override": purchase_params_override,
        "qty_params_override": qty_params_override,
    }

    model_files: dict[str, list[str]] = {"purchase_1w": [], "purchase_2w": [], "qty_1w": [], "qty_2w": []}

    def _train_over_folds(target_col: str, *, model_key: str, objective: str, metric: str, seed0: int, extra_params_fn=None):
        fold_metrics: list[float] = []
        fold_best_iters: list[int] = []
        fold_paths: list[str] = []

        for fold, block in enumerate(val_blocks):
            # Validation weeks are exactly `block`; training is strictly before the first val week.
            val_mask = np.isin(week_arr, block)
            train_mask = week_arr < block[0]

            X_train = X_all.loc[train_mask]
            X_val = X_all.loc[val_mask]
            y_train = np.asarray(pdf_all[target_col].values)[train_mask]
            y_val = np.asarray(pdf_all[target_col].values)[val_mask]

            extra_params = extra_params_fn(y_train) if extra_params_fn else None
            booster, fold_report = _train_lgbm(
                X_train,
                y_train,
                X_val,
                y_val,
                feature_names,
                cat_idx,
                objective=objective,
                metric=metric,
                seed=seed0 + fold,
                extra_params=extra_params,
            )

            # Persist each fold model (enables optional ensemble at predict time)
            out_name = f"lgb_{model_key}_fold{fold}.txt" if args.cv_folds > 1 else f"lgb_{model_key}.txt"
            out_path = ARTIFACTS / out_name
            booster.save_model(str(out_path))
            fold_paths.append(out_name)

            if "val_auc" in fold_report:
                fold_metrics.append(float(fold_report["val_auc"]))
            elif "val_mae" in fold_report:
                fold_metrics.append(float(fold_report["val_mae"]))
            fold_best_iters.append(int(fold_report.get("best_iteration", 0) or 0))

        # Aggregate
        agg = {
            "fold_metrics": fold_metrics,
            "fold_best_iterations": fold_best_iters,
            "mean": float(np.mean(fold_metrics)) if fold_metrics else None,
            "std": float(np.std(fold_metrics)) if fold_metrics else None,
        }
        report[model_key] = agg
        model_files[model_key] = fold_paths

    # Purchase models (classification)
    _train_over_folds(
        "Target_purchase_next_1w",
        model_key="purchase_1w",
        objective="binary",
        metric="auc",
        seed0=args.seed,
        extra_params_fn=lambda y: {**purchase_params_override, "scale_pos_weight": _scale_pos_weight(y)},
    )
    _train_over_folds(
        "Target_purchase_next_2w",
        model_key="purchase_2w",
        objective="binary",
        metric="auc",
        seed0=args.seed + 10,
        extra_params_fn=lambda y: {**purchase_params_override, "scale_pos_weight": _scale_pos_weight(y)},
    )

    # Quantity models
    _train_over_folds(
        "Target_qty_next_1w",
        model_key="qty_1w",
        objective=args.qty_objective,
        metric="l1",
        seed0=args.seed + 20,
        extra_params_fn=(lambda _y: qty_params_override) if qty_params_override else None,
    )
    _train_over_folds(
        "Target_qty_next_2w",
        model_key="qty_2w",
        objective=args.qty_objective,
        metric="l1",
        seed0=args.seed + 30,
        extra_params_fn=(lambda _y: qty_params_override) if qty_params_override else None,
    )

    # Optional: two-stage quantity modeling
    #   q_hat = p_hat * q_hat_cond
    # where q_hat_cond is trained on rows with purchase==1, using log1p(target_qty).
    two_stage_enabled = bool(args.two_stage_qty)
    use_two_stage_1w = False
    use_two_stage_2w = False
    if two_stage_enabled:
        report["qty_two_stage"] = {}

        # For two-stage mode we keep the original single split behavior.
        train_df, val_df = _time_split(feat_df, val_weeks=args.val_weeks)
        X_train, _, _, _ = to_pandas_for_lgbm(train_df, categorical=CATEGORICAL_COLS)
        X_val, _, _, _ = to_pandas_for_lgbm(val_df, categorical=CATEGORICAL_COLS)

        # Load (single) purchase models from the most recent fold (fold0 when cv_folds>1 is disabled here)
        m1 = lgb.Booster(model_file=str(ARTIFACTS / "lgb_purchase_1w.txt"))
        m2 = lgb.Booster(model_file=str(ARTIFACTS / "lgb_purchase_2w.txt"))

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
        "model_files": model_files,
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
