from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def _flatten(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten a single experiment record into a tabular row."""

    out: dict[str, Any] = {}
    for k, v in record.items():
        if k == "args" and isinstance(v, dict):
            for ak, av in v.items():
                out[f"arg_{ak}"] = av
        elif k == "model_files":
            # keep as JSON string so it fits in CSV
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize and optionally plot training experiments.")
    ap.add_argument(
        "--in",
        dest="inp",
        type=str,
        default=str(ARTIFACTS / "experiments.jsonl"),
        help="Path to artifacts/experiments.jsonl",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default=str(ARTIFACTS / "experiments.csv"),
        help="Where to write a flat CSV summary",
    )
    ap.add_argument(
        "--out-png",
        type=str,
        default=str(ARTIFACTS / "experiments.png"),
        help="Where to write plots (requires matplotlib)",
    )
    ap.add_argument("--top", type=int, default=15, help="How many top runs to print")
    args = ap.parse_args()

    in_path = pathlib.Path(args.inp)
    if not in_path.exists():
        raise FileNotFoundError(f"No experiment log found: {in_path}")

    records: list[dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        print("No records found.")
        return

    rows = [_flatten(r) for r in records]
    df = pd.DataFrame(rows)

    # Ensure consistent ordering
    if "started_at_utc" in df.columns:
        df = df.sort_values("started_at_utc")

    out_csv = pathlib.Path(args.out_csv)
    out_csv.parent.mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote CSV summary: {out_csv} ({len(df)} runs)")

    # Print leaderboard by a simple composite proxy.
    # NOTE: This is NOT the official leaderboard metric; it's just to rank experiments locally.
    # Higher AUC is better; lower MAE is better.
    score_cols = [
        "purchase_1w_mean_auc",
        "purchase_2w_mean_auc",
        "qty_1w_mean_mae",
        "qty_2w_mean_mae",
    ]
    missing = [c for c in score_cols if c not in df.columns]
    if missing:
        print(f"Missing columns for ranking: {missing}")
    else:
        rank_df = df.copy()
        # crude proxy: avg_auc - 0.01*avg_mae (scales are different; tweak as you like)
        rank_df["avg_auc"] = rank_df[["purchase_1w_mean_auc", "purchase_2w_mean_auc"]].mean(axis=1)
        rank_df["avg_mae"] = rank_df[["qty_1w_mean_mae", "qty_2w_mean_mae"]].mean(axis=1)
        rank_df["proxy_score"] = rank_df["avg_auc"] - 0.01 * rank_df["avg_mae"]

        cols_to_show = [
            "started_at_utc",
            "run_id",
            "avg_auc",
            "avg_mae",
            "proxy_score",
            "arg_cv_folds",
            "arg_val_weeks",
            "arg_qty_objective",
        ]
        cols_to_show = [c for c in cols_to_show if c in rank_df.columns]
        top = rank_df.sort_values("proxy_score", ascending=False).tail(args.top)
        print("\nTop runs by proxy_score (higher is better):")
        print(top[cols_to_show].to_string(index=False))

    # Optional plotting
    try:
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        if "purchase_1w_mean_auc" in df.columns:
            ax[0].plot(df["started_at_utc"], df["purchase_1w_mean_auc"], label="AUC 1w")
        if "purchase_2w_mean_auc" in df.columns:
            ax[0].plot(df["started_at_utc"], df["purchase_2w_mean_auc"], label="AUC 2w")
        ax[0].set_ylabel("AUC")
        ax[0].legend(loc="best")
        ax[0].grid(True, alpha=0.2)

        if "qty_1w_mean_mae" in df.columns:
            ax[1].plot(df["started_at_utc"], df["qty_1w_mean_mae"], label="MAE 1w")
        if "qty_2w_mean_mae" in df.columns:
            ax[1].plot(df["started_at_utc"], df["qty_2w_mean_mae"], label="MAE 2w")
        ax[1].set_ylabel("MAE")
        ax[1].legend(loc="best")
        ax[1].grid(True, alpha=0.2)

        for a in ax:
            a.tick_params(axis="x", labelrotation=30)

        out_png = pathlib.Path(args.out_png)
        out_png.parent.mkdir(exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        print(f"Wrote plot: {out_png}")

    except ImportError:
        print("\nmatplotlib not installed; skipping plots. (You can: pip install matplotlib)")


if __name__ == "__main__":
    main()
