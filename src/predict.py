from __future__ import annotations

import argparse
import json
import pathlib

import lightgbm as lgb
import numpy as np
import polars as pl

from features import FeatureConfig, build_test_features, scan_test, scan_train, to_pandas_for_lgbm


ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"


def _clip_proba(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 1e-6, 1 - 1e-6)


def _load_model_ensemble(meta: dict, *, key: str, default_filename: str) -> list[lgb.Booster]:
    model_files = (meta.get("model_files") or {}).get(key)
    if isinstance(model_files, list) and len(model_files) > 0:
        return [lgb.Booster(model_file=str(ARTIFACTS / fn)) for fn in model_files]
    return [lgb.Booster(model_file=str(ARTIFACTS / default_filename))]


def _predict_mean(models: list[lgb.Booster], X) -> np.ndarray:
    if len(models) == 1:
        return models[0].predict(X)
    preds = [m.predict(X) for m in models]
    return np.mean(np.vstack(preds), axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(ROOT / "submission.csv"))
    args = ap.parse_args()

    meta = json.loads((ARTIFACTS / "meta.json").read_text())
    cfg = FeatureConfig(**meta.get("config", {}))
    categorical_cols = meta.get("categorical_cols", [])
    feature_names_expected = meta.get("feature_names")
    qty_use_two_stage_1w = bool(meta.get("qty_use_two_stage_1w", False))
    qty_use_two_stage_2w = bool(meta.get("qty_use_two_stage_2w", False))
    qty_cond_log1p = bool(meta.get("qty_cond_log1p", True))

    # Load models (single model or ensemble across folds)
    m_p1_list = _load_model_ensemble(meta, key="purchase_1w", default_filename="lgb_purchase_1w.txt")
    m_p2_list = _load_model_ensemble(meta, key="purchase_2w", default_filename="lgb_purchase_2w.txt")
    m_q1_list = _load_model_ensemble(meta, key="qty_1w", default_filename="lgb_qty_1w.txt")
    m_q2_list = _load_model_ensemble(meta, key="qty_2w", default_filename="lgb_qty_2w.txt")

    # Optional conditional-qty models (two-stage)
    cond_1w_path = ARTIFACTS / "lgb_qty_cond_1w.txt"
    cond_2w_path = ARTIFACTS / "lgb_qty_cond_2w.txt"
    m_q1c = lgb.Booster(model_file=str(cond_1w_path)) if cond_1w_path.exists() else None
    m_q2c = lgb.Booster(model_file=str(cond_2w_path)) if cond_2w_path.exists() else None

    test_lf = scan_test(DATA / "Test.csv")

    # If we have cached features, use them; otherwise build from Train.csv.
    # Note: cache_features.py always writes features_train.parquet; history can be derived from it.
    train_cache = ARTIFACTS / "features_train.parquet"
    hist_cache = ARTIFACTS / "features_hist.parquet"
    if hist_cache.exists():
        train_hist_lf = pl.scan_parquet(hist_cache)
    elif train_cache.exists():
        train_hist_lf = pl.scan_parquet(train_cache).drop(
            [
                "Target_purchase_next_1w",
                "Target_qty_next_1w",
                "Target_purchase_next_2w",
                "Target_qty_next_2w",
            ]
        )
    else:
        train_hist_lf = scan_train(DATA / "Train.csv")

    feat_lf = build_test_features(train_hist_lf, test_lf, cfg)
    feat_df = feat_lf.collect(streaming=True)

    X, feature_names, cat_idx, _ = to_pandas_for_lgbm(feat_df, categorical=categorical_cols)

    # Align features to the exact training schema
    if feature_names_expected:
        X = X.reindex(columns=feature_names_expected, fill_value=0)
        # Restore categorical dtype after reindex
        for c in categorical_cols:
            if c in X.columns:
                X[c] = X[c].astype("category")

    # Predict
    p1 = _clip_proba(_predict_mean(m_p1_list, X))
    p2 = _clip_proba(_predict_mean(m_p2_list, X))

    # Quantity
    can_two_stage = (m_q1c is not None) and (m_q2c is not None)
    if can_two_stage and (qty_use_two_stage_1w or qty_use_two_stage_2w):
        q1c = m_q1c.predict(X)
        q2c = m_q2c.predict(X)
        if qty_cond_log1p:
            q1c = np.expm1(q1c)
            q2c = np.expm1(q2c)
        q1_two = np.clip(p1 * q1c, 0.0, None)
        q2_two = np.clip(p2 * q2c, 0.0, None)

        q1_dir = np.clip(_predict_mean(m_q1_list, X), 0.0, None)
        q2_dir = np.clip(_predict_mean(m_q2_list, X), 0.0, None)

        q1 = q1_two if qty_use_two_stage_1w else q1_dir
        q2 = q2_two if qty_use_two_stage_2w else q2_dir
    else:
        q1 = np.clip(_predict_mean(m_q1_list, X), 0.0, None)
        q2 = np.clip(_predict_mean(m_q2_list, X), 0.0, None)

    sub = pl.DataFrame(
        {
            "ID": feat_df["ID"],
            "Target_purchase_next_1w": p1,
            "Target_qty_next_1w": q1,
            "Target_purchase_next_2w": p2,
            "Target_qty_next_2w": q2,
        }
    )

    out_path = pathlib.Path(args.out)
    sub.write_csv(out_path)
    print(f"Wrote {out_path} with {sub.height} rows")


if __name__ == "__main__":
    main()
