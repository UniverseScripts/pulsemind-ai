"""Stage 7 -- gap filling, vectorised.

Reproduces the design of cleaning_and_filling_mising_values_claude.ipynb exactly
-- the same five companion columns per parameter -- but replaces its row-by-row
`df.at[i, col]` loops (~880k scalar lookups per function at 40k rows) with whole
column operations. That is what makes 4.2M rows tractable.

Two properties that are load-bearing rather than incidental:
  1. Reference fill values come from the TRAINING patients only and are written
     to disk, so the test split never contributes to them.
  2. ⚠️ `_delta_t_min` is the age of the value actually IN USE -- zero on a row
     that was measured -- not the gap since the previous observation. That is the
     quantity a model can act on, and it is why this column does not reproduce
     the original notebook's while the other 44 do.
"""
from __future__ import annotations

import json

import polars as pl

from .. import config as C
from ..common import account_parquet, cached_stage, log


def build_expressions(params: list[str]) -> list[pl.Expr]:
    """The five companion columns per parameter, as pure column algebra."""
    exprs: list[pl.Expr] = []
    for p in params:
        col = pl.col(p)
        observed = col.is_not_null()

        # Timestamp of the most recent real observation, carried within the stay.
        last_t = (
            pl.when(observed).then(pl.col("charttime")).otherwise(None)
            .forward_fill().over("stay_id")
        )
        age_min = (pl.col("charttime") - last_t).dt.total_seconds() / 60.0

        exprs += [
            observed.cast(pl.Int8).alias(f"{p}_observed"),
            age_min.cast(pl.Float32).alias(f"{p}_delta_t_min"),
            # Carry forward, but only while the value is still fresh enough.
            pl.when(age_min <= C.LOCF_CUTOFF_MIN)
              .then(col.forward_fill().over("stay_id"))
              .otherwise(None).cast(pl.Float32).alias(f"{p}_locf"),
            # Never measured anywhere in this stay -> device not used, not a gap.
            (observed.sum().over("stay_id") == 0).cast(pl.Int8)
              .alias(f"{p}_structurally_missing_in_stay"),
        ]
    return exprs


def main(force: bool = False) -> None:
    sources = [C.T2_WIDE_PQ, C.FOLDS_PQ]
    with cached_stage("s07_impute", sources=sources, output=C.T2_IMPUTED_PQ,
                      force=force) as ran:
        if not ran:
            return

        params = C.FROZEN_ORDER
        df = pl.read_parquet(C.T2_WIDE_PQ).sort("stay_id", "charttime")
        log(f"loaded {df.height:,} rows x {df.width} cols")

        folds = pl.read_parquet(C.FOLDS_PQ).select("subject_id", "split")
        df = df.join(folds, on="subject_id", how="left")

        # --- reference fill values: TRAIN PATIENTS ONLY ----------------------
        train_mask = pl.col("split") == "train"
        ref = {
            p: df.filter(train_mask).select(pl.col(p).median()).item()
            for p in params
        }
        C.REFERENCE_STATS_JSON.write_text(json.dumps(ref, indent=2))
        log("reference fill values (train-split medians, reused unchanged on test):")
        for p, v in ref.items():
            log(f"  {p:<24} {v if v is None else round(float(v), 2)}")

        # --- the five companion columns --------------------------------------
        df = df.with_columns(build_expressions(params))
        df = df.with_columns([
            pl.col(f"{p}_locf").fill_null(ref[p]).cast(pl.Float32).alias(f"{p}_final")
            for p in params
        ])

        ordered = ["stay_id", "subject_id", "charttime"]
        for p in params:
            ordered += [p] + [f"{p}{s}" for s in C.SUFFIXES]
        ordered += [C.LEGACY_TARGET]   # the original dropped this here; we keep it
        df = df.select(ordered)

        df.write_parquet(C.T2_IMPUTED_PQ, compression="zstd")
        log(f"wrote {df.height:,} rows x {df.width} cols")

        # --- what the filling actually did ------------------------------------
        log("observed -> carried forward -> reference-filled, per parameter:")
        for p in params:
            obs = df[f"{p}_observed"].sum()
            locf = df[f"{p}_locf"].is_not_null().sum()
            n = df.height
            log(f"  {p:<24} measured {100*obs/n:5.1f}%  "
                f"usable after carry-forward {100*locf/n:5.1f}%  "
                f"reference-filled {100*(n-locf)/n:5.1f}%")
        smis = {p: int(df[f"{p}_structurally_missing_in_stay"].sum()) for p in params}
        log("rows in admissions where the parameter was never measured at all:")
        for p, v in sorted(smis.items(), key=lambda kv: -kv[1]):
            log(f"  {p:<24} {v:>10,} ({100*v/df.height:5.1f}%)")

    account_parquet("Table 2 (imputed)", C.T2_IMPUTED_PQ)


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)
