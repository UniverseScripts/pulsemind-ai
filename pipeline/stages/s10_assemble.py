"""Stage 10 -- assemble the model matrix.

Joins Table 1 (broadcast per admission), Table 2 (native grain) and Table 3
(already resolved per timestamp) onto one row per (stay_id, charttime).

Also writes features.json: which columns exist, which are categorical, which
source table each came from, and the named feature sets the bake-off compares.
Column *names* are left alone so the like-for-like diff against raw-query.csv
still works; provenance is recorded in the manifest instead of by renaming.

Guards enforced here, not hoped for:
  * every leaky_/outcome column is excluded from the feature list
  * no identifier reaches the model
  * Table 3 really did use strictly-before intervals
"""
from __future__ import annotations

import json

import polars as pl

from .. import config as C
from ..common import (account_parquet, cached_stage, forward_label_columns, log)

FEATURES_JSON = C.BUILD / "features.json"

# Never model inputs: identifiers, timestamps, split bookkeeping, and BOTH
# labels.
#
# LEGACY_TARGET is listed explicitly and not just as "whatever C.TARGET is".
# When the target moved to a forward label, `warning` stopped matching C.TARGET
# and would have walked straight into the feature set -- it is still a column
# of the matrix, because verify.py diffs it against the BigQuery export. A
# guard keyed only on the CURRENT target silently stops protecting the previous
# one at the exact moment you switch.
NON_FEATURES = {"stay_id", "subject_id", "hadm_id", "charttime",
                C.TARGET, C.LEGACY_TARGET, "split", "fold"}

CATEGORICAL = ["gender", "admission_type", "admission_location", "insurance",
               "language", "marital_status", "race", "first_careunit",
               "ventilator_mode"]


def main(force: bool = False) -> None:
    sources = [C.T2_IMPUTED_PQ, C.T1_STATIC_PQ, C.T3_INTERV_PQ, C.FOLDS_PQ]
    with cached_stage("s10_assemble", sources=sources, output=C.MODEL_MATRIX_PQ,
                      force=force, extra=C.FP_MATRIX) as ran:
        if not ran:
            return

        t2 = pl.read_parquet(C.T2_IMPUTED_PQ)
        n_base, stays_base = t2.height, t2["stay_id"].n_unique()
        log(f"Table 2 base grain: {n_base:,} rows / {stays_base:,} admissions")

        t3 = pl.read_parquet(C.T3_INTERV_PQ)
        t1 = pl.read_parquet(C.T1_STATIC_PQ).drop(["subject_id", "hadm_id"])
        folds = pl.read_parquet(C.FOLDS_PQ).select("subject_id", "split", "fold")

        # Table 1 columns that describe an outcome or a post-discharge fact.
        leaky = [c for c in t1.columns if c.startswith("leaky_")]
        t1_feat = t1.drop(leaky + ["vent_start", "vent_end"])
        log(f"Table 1: {t1_feat.width - 1} feature columns "
            f"({len(leaky)} leaky_ columns dropped: {', '.join(leaky[:4])}...)")

        df = (t2
              .join(t3, on=["stay_id", "charttime"], how="left")
              .join(t1_feat, on="stay_id", how="left")
              .join(folds, on="subject_id", how="left"))

        assert df.height == n_base, (
            f"join changed the row count: {n_base:,} -> {df.height:,}")
        assert df["stay_id"].n_unique() == stays_base, "join changed the admission count"
        log(f"[green]join integrity: {df.height:,} rows, "
            f"{df['stay_id'].n_unique():,} admissions -- unchanged[/green]")

        # A LEFT join that matches nothing is still a successful LEFT join.
        # Row counts were checked here; whether the join MATCHED was not, and
        # 22.53% of rows once carried NULL across all 33 static features with
        # every assertion passing. See `s02_table1_static.py`'s docstring.
        unmatched = df.filter(pl.col("gender").is_null()).height
        pct = 100 * unmatched / df.height
        if unmatched:
            missing_stays = (df.filter(pl.col("gender").is_null())["stay_id"]
                             .n_unique())
            log(f"[yellow]Table 1 join: {unmatched:,} rows ({pct:.2f}%) across "
                f"{missing_stays:,} admissions have no static record[/yellow]")
        if C.S02_FINAL_COHORT:
            assert pct < 1.0, (
                f"{unmatched:,} rows ({pct:.2f}%) have no Table 1 match. Table 1 "
                f"and Table 2 are built on different cohorts -- check that s02 ran "
                f"after s04 and read {C.COHORT_PQ.name}.")
            log(f"[green]Table 1 coverage: {100 - pct:.2f}% of rows[/green]")
        else:
            # Deliberately reproducing the legacy strict-cohort build, so the
            # hole is expected. Warn loudly rather than assert -- the gate
            # harness needs to be able to rebuild this state on purpose.
            log("[yellow]S02_FINAL_COHORT is off: reproducing the legacy build "
                "with an incomplete Table 1 on purpose[/yellow]")

        # --- reverse-causation guard on Table 3 -------------------------------
        for g in ("vasopressor", "sedative", "opioid", "paralytic"):
            col = f"{g}_minutes_since_start"
            if col in df.columns:
                bad = df.filter((pl.col(col) != -1) & (pl.col(col) <= 0)).height
                assert bad == 0, (
                    f"{col}: {bad:,} rows have an infusion starting at or after the "
                    f"row timestamp -- the strictly-before guard failed")
        log("[green]reverse-causation guard: every intervention began strictly "
            "before its row[/green]")

        # ------------------------------------------------------------------
        # Tidal volume per kg of predicted body weight.
        #
        # The one genuinely new physiological signal in this pass. Absolute
        # tidal volume in millilitres is not comparable between patients: 450
        # mL is lung-protective for a 180 cm man and frankly injurious for a
        # 150 cm woman. ARDSNet targets 6-8 mL/kg PBW, and PBW depends on
        # height and sex only -- never on actual weight, which moves with
        # fluid balance and would make the denominator a treatment effect.
        #
        # This is a ratio of two columns already present, so it adds no new
        # information in the information-theoretic sense. It adds a SHAPE the
        # trees would otherwise have to discover by splitting on height and
        # tidal volume jointly, many times over, in every region of the space.
        # ------------------------------------------------------------------
        if C.ENABLE_PBW and "pbw_kg" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("pbw_kg") > 0)
                  .then(pl.col("tidal_volume_observed_final") / pl.col("pbw_kg"))
                  .otherwise(None)
                  .alias("tidal_volume_ml_per_kg_pbw"))
            got = df["tidal_volume_ml_per_kg_pbw"].is_not_null().sum()
            med = df["tidal_volume_ml_per_kg_pbw"].median()
            log(f"tidal volume per kg PBW: {got:,} rows ({100*got/df.height:.1f}%), "
                f"median {med:.2f} mL/kg [dim](ARDSNet target 6-8)[/dim]")

        df.write_parquet(C.MODEL_MATRIX_PQ, compression="zstd")

        # ------------------------------------------------------------------
        # Feature manifest
        # ------------------------------------------------------------------
        all_cols = [c for c in df.columns if c not in NON_FEATURES]
        t2_cols, t1_cols, t3_cols = [], [], []
        t1_names = set(t1_feat.columns) - {"stay_id"}
        t3_names = set(t3.columns) - {"stay_id", "charttime"}
        for c in all_cols:
            if c in t1_names:
                t1_cols.append(c)
            elif c in t3_names:
                t3_cols.append(c)
            else:
                t2_cols.append(c)

        final_only = [f"{p}_final" for p in C.FROZEN_ORDER]
        doc_only = ([f"{p}_observed" for p in C.FROZEN_ORDER]
                    + [f"{p}_delta_t_min" for p in C.FROZEN_ORDER]
                    + [f"{p}_structurally_missing_in_stay" for p in C.FROZEN_ORDER])
        # Raw (pre-imputation) columns duplicate _locf/_final information and are
        # mostly null by construction -- keep them out of the modelling sets.
        t2_model = [c for c in t2_cols if c not in C.FROZEN_ORDER]

        manifest = {
            "target": C.TARGET,
            "target_source": C.TARGET_SOURCE,
            "legacy_target": C.LEGACY_TARGET,
            "group_key": C.GROUP_KEY,
            "categorical": [c for c in CATEGORICAL if c in df.columns],
            "groups": {"t1_static": t1_cols, "t2_timeseries": t2_cols,
                       "t3_interventions": t3_cols},
            "sets": {
                "final_only": final_only,
                "t2_all": t2_model,
                "full": t2_model + t1_cols + t3_cols,
                "documentation_only": doc_only,
            },
        }
        FEATURES_JSON.write_text(json.dumps(manifest, indent=2))

        log(f"columns by source -- Table 1: {len(t1_cols)}  Table 2: {len(t2_cols)}  "
            f"Table 3: {len(t3_cols)}")
        for name, cols in manifest["sets"].items():
            log(f"  feature set '{name}': {len(cols)} columns")

        leaked = [c for c in manifest["sets"]["full"]
                  if c.startswith("leaky_") or c in NON_FEATURES]
        assert not leaked, f"leaky or identifier columns reached the feature set: {leaked}"

        # Nothing describing the future may be a feature. Checked against the
        # forward-target file's ACTUAL schema rather than a hardcoded prefix
        # list, so adding a label column there cannot quietly create a feature
        # here. Belt and braces: forward labels are not written into the model
        # matrix at all, so this should be unreachable -- which is exactly when
        # an assertion is worth having.
        fwd_cols = set(forward_label_columns())
        from_future = [c for c in manifest["sets"]["full"] if c in fwd_cols]
        assert not from_future, (
            f"forward-label columns reached the feature set: {from_future}")

        log("[green]feature guard: no identifier, outcome, leaky_ or "
            "forward-label column in any feature set[/green]")

    account_parquet("model matrix", C.MODEL_MATRIX_PQ)


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)
