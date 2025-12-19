from __future__ import annotations

import pathlib

import polars as pl


ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _scan_csv(path: pathlib.Path) -> pl.LazyFrame:
    # Try to keep memory small: infer schema on a small sample, then stream aggregations.
    return pl.scan_csv(
        path,
        infer_schema_length=5_000,
        ignore_errors=True,
        try_parse_dates=True,
        low_memory=True,
    )


def main() -> None:
    train_path = DATA / "Train.csv"
    test_path = DATA / "Test.csv"

    train_lf = _scan_csv(train_path)
    test_lf = _scan_csv(test_path)

    print("## Files")
    print(f"Train: {train_path}")
    print(f"Test : {test_path}")

    print("\n## Row counts")
    counts = pl.DataFrame(
        {
            "split": ["train", "test"],
            "rows": [
                train_lf.select(pl.len()).collect(streaming=True).item(),
                test_lf.select(pl.len()).collect(streaming=True).item(),
            ],
        }
    )
    print(counts)

    print("\n## week_start ranges")
    for name, lf in [("train", train_lf), ("test", test_lf)]:
        if "week_start" not in lf.collect_schema().names():
            print(f"{name}: week_start missing")
            continue
        rng = (
            lf.select(
                pl.col("week_start").min().alias("min_week_start"),
                pl.col("week_start").max().alias("max_week_start"),
                pl.col("week_start").n_unique().alias("n_unique_week_start"),
            )
            .collect(streaming=True)
        )
        print(f"{name}:\n{rng}\n")

    print("\n## Key cardinalities (train)")
    key_cols = [
        "customer_id",
        "product_unit_variant_id",
        "product_id",
        "product_grade_variant_id",
        "customer_category",
        "customer_status",
        "grade_name",
        "unit_name",
    ]
    existing = [c for c in key_cols if c in train_lf.collect_schema().names()]
    if existing:
        card = train_lf.select([pl.col(c).n_unique().alias(f"n_unique_{c}") for c in existing]).collect(
            streaming=True
        )
        print(card)

    print("\n## Null rates (train, selected cols)")
    sel = [
        "qty_this_week",
        "num_orders_week",
        "spend_this_week",
        "purchased_this_week",
        "selling_price",
        "customer_created_at",
        "Target_purchase_next_1w",
        "Target_qty_next_1w",
        "Target_purchase_next_2w",
        "Target_qty_next_2w",
    ]
    sel = [c for c in sel if c in train_lf.collect_schema().names()]
    if sel:
        nulls = (
            train_lf.select(
                [
                    (pl.col(c).null_count() / pl.len()).alias(f"null_rate_{c}")
                    for c in sel
                ]
            )
            .collect(streaming=True)
        )
        print(nulls)


if __name__ == "__main__":
    main()
