from __future__ import annotations

import argparse
import json
import pathlib

import polars as pl

from features import FeatureConfig, build_train_features, scan_train


ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-train",
        type=str,
        default=str(ARTIFACTS / "features_train.parquet"),
        help="Where to write the full training feature table (includes targets).",
    )
    ap.add_argument(
        "--out-hist",
        type=str,
        default=str(ARTIFACTS / "features_hist.parquet"),
        help="Where to write the history feature table (targets dropped) for inference joins.",
    )
    args = ap.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)

    cfg = FeatureConfig()
    train_lf = scan_train(DATA / "Train.csv")
    feat_lf = build_train_features(train_lf, cfg)

    out_train = pathlib.Path(args.out_train)
    out_hist = pathlib.Path(args.out_hist)

    def _write_parquet(lf: pl.LazyFrame, path: pathlib.Path) -> None:
        # Polars' sink_parquet is not always available in the standard engine.
        # Prefer sink_parquet (streaming) when supported; otherwise fall back to
        # a streaming collect + write_parquet.
        try:
            lf.sink_parquet(path, compression="zstd", engine="streaming")
            return
        except Exception:
            df = lf.collect(streaming=True)
            df.write_parquet(path, compression="zstd")

    # Write full training features (with targets)
    _write_parquet(feat_lf, out_train)

    # Write inference history table (drop targets)
    target_cols = [
        "Target_purchase_next_1w",
        "Target_qty_next_1w",
        "Target_purchase_next_2w",
        "Target_qty_next_2w",
    ]
    hist_lf = feat_lf.drop([c for c in target_cols if c in feat_lf.collect_schema().names()])
    _write_parquet(hist_lf, out_hist)

    meta = {
        "config": {"windows": list(cfg.windows)},
        "out_train": str(out_train),
        "out_hist": str(out_hist),
    }
    (ARTIFACTS / "features_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote: {out_train}")
    print(f"Wrote: {out_hist}")


if __name__ == "__main__":
    main()
