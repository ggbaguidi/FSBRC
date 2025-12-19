from __future__ import annotations

import argparse
import pathlib

import numpy as np
import polars as pl


PROBA_COLS = ["Target_purchase_next_1w", "Target_purchase_next_2w"]
QTY_COLS = ["Target_qty_next_1w", "Target_qty_next_2w"]
ALL_COLS = ["ID", *PROBA_COLS, *QTY_COLS]


def _clip_proba(x: pl.Expr) -> pl.Expr:
    return x.clip(1e-6, 1 - 1e-6)


def main() -> None:
    ap = argparse.ArgumentParser(description="Blend two submission CSVs by averaging predictions (by ID).")
    ap.add_argument("--a", required=True, help="Path to first submission CSV")
    ap.add_argument("--b", required=True, help="Path to second submission CSV")
    ap.add_argument("--out", required=True, help="Output path")
    ap.add_argument("--w", type=float, default=0.5, help="Weight for A: out = w*A + (1-w)*B")
    args = ap.parse_args()

    if not (0.0 <= args.w <= 1.0):
        raise ValueError("--w must be between 0 and 1")

    a_path = pathlib.Path(args.a)
    b_path = pathlib.Path(args.b)
    out_path = pathlib.Path(args.out)

    a = pl.read_csv(a_path)
    b = pl.read_csv(b_path)

    missing_a = [c for c in ALL_COLS if c not in a.columns]
    missing_b = [c for c in ALL_COLS if c not in b.columns]
    if missing_a:
        raise ValueError(f"Submission A is missing columns: {missing_a}")
    if missing_b:
        raise ValueError(f"Submission B is missing columns: {missing_b}")

    a = a.select(ALL_COLS)
    b = b.select(ALL_COLS)

    # Inner join to ensure IDs align; validate sizes afterwards.
    j = a.join(b, on="ID", how="inner", suffix="_b")
    if j.height != a.height or j.height != b.height:
        # Provide a helpful hint
        a_only = a.select("ID").join(b.select("ID"), on="ID", how="anti").height
        b_only = b.select("ID").join(a.select("ID"), on="ID", how="anti").height
        raise ValueError(
            "Submissions do not contain the same set of IDs. "
            f"Only-in-A: {a_only}, only-in-B: {b_only}."
        )

    w = float(args.w)

    exprs: list[pl.Expr] = [pl.col("ID")]
    for c in PROBA_COLS:
        exprs.append(_clip_proba(pl.lit(w) * pl.col(c) + pl.lit(1.0 - w) * pl.col(f"{c}_b")).alias(c))
    for c in QTY_COLS:
        exprs.append((pl.lit(w) * pl.col(c) + pl.lit(1.0 - w) * pl.col(f"{c}_b")).clip(0.0, None).alias(c))

    out = j.select(exprs)

    # Basic sanity: probabilities in [0,1], quantities non-negative
    # (avoid expensive full scans; just ensure dtypes are numeric)
    for c in PROBA_COLS + QTY_COLS:
        if out[c].dtype not in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
            out = out.with_columns(pl.col(c).cast(pl.Float64, strict=False))

    out.write_csv(out_path)
    print(f"Wrote blended submission: {out_path} ({out.height} rows), w={w}")


if __name__ == "__main__":
    main()
