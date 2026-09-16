"""Verify the environment: library versions, GPU availability, MIMIC file resolution."""
from __future__ import annotations

import numpy as np

from .. import config as C
from ..common import console, log


def main() -> None:
    console.rule("[bold cyan]Environment check")

    import catboost, duckdb, lightgbm as lgb, polars as pl, sklearn, xgboost as xgb
    log(f"duckdb {duckdb.__version__} | polars {pl.__version__} | numpy {np.__version__} "
        f"| sklearn {sklearn.__version__}")
    log(f"lightgbm {lgb.__version__} | xgboost {xgb.__version__} | catboost {catboost.__version__}")

    X = np.random.rand(50_000, 12).astype(np.float32)
    y = (X[:, 0] + X[:, 3] > 1.0).astype(np.int32)

    # --- XGBoost on CUDA ---
    try:
        m = xgb.XGBClassifier(device="cuda", tree_method="hist",
                              n_estimators=25, max_bin=C.GPU_MAX_BIN, verbosity=0)
        m.fit(X, y)
        m.predict_proba(X[:100])
        log("[green]XGBoost  CUDA : OK[/green]")
    except Exception as e:  # noqa: BLE001
        log(f"[red]XGBoost  CUDA : FAIL[/red] {type(e).__name__}: {str(e)[:220]}")

    # --- CatBoost on GPU ---
    try:
        c = catboost.CatBoostClassifier(task_type="GPU", devices="0",
                                        iterations=25, verbose=0, allow_writing_files=False)
        c.fit(X, y)
        log("[green]CatBoost GPU  : OK[/green]")
    except Exception as e:  # noqa: BLE001
        log(f"[red]CatBoost GPU  : FAIL[/red] {type(e).__name__}: {str(e)[:220]}")

    # --- LightGBM: pip wheels on Windows are CPU-only by design ---
    try:
        lgb.train({"objective": "binary", "device_type": "gpu", "verbose": -1},
                  lgb.Dataset(X, label=y), num_boost_round=5)
        log("[green]LightGBM GPU  : OK[/green]")
    except Exception as e:  # noqa: BLE001
        log(f"[yellow]LightGBM GPU  : not available[/yellow] ({str(e)[:90]}) "
            f"-- will run on {C.N_CPU_THREADS} CPU threads, which is fast at this size")

    # --- PyTorch / Blackwell sm_120 ---
    try:
        import torch
        ok = torch.cuda.is_available()
        log(f"torch {torch.__version__} | cuda build {torch.version.cuda} | available={ok}")
        if ok:
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            # Driver figures, not CUDA's -- torch over-reports free VRAM on this
            # WDDM setup by gigabytes. See pipeline/core/generate.vram_status.
            from pipeline.core.generate import vram_status
            v = vram_status()
            log(f"[green]PyTorch  CUDA : OK[/green] {name} sm_{cap[0]}{cap[1]} | "
                f"{v['free_mib']} MiB free of {v['total_mib']} ({v['source']})")
            t = torch.randn(4096, 4096, device="cuda")
            torch.cuda.synchronize()
            log(f"  matmul check: {float((t @ t).sum()):.1f}")
    except ImportError:
        log("[yellow]PyTorch not installed yet[/yellow] (needed only for the neural candidates)")
    except Exception as e:  # noqa: BLE001
        log(f"[red]PyTorch  CUDA : FAIL[/red] {type(e).__name__}: {str(e)[:220]}")

    # --- MIMIC files ---
    console.rule("[bold cyan]MIMIC-IV files")
    for label, p in [("chartevents", C.CHARTEVENTS), ("icustays", C.ICUSTAYS),
                     ("procedureevents", C.PROCEDUREEVENTS), ("inputevents", C.INPUTEVENTS),
                     ("d_items", C.D_ITEMS), ("patients", C.PATIENTS),
                     ("admissions", C.ADMISSIONS), ("diagnoses_icd", C.DIAGNOSES_ICD),
                     ("transfers", C.TRANSFERS), ("raw-query.csv", C.BQ_EXPORT)]:
        log(f"  {label:<16} {p.stat().st_size/1e9:>7.2f} GB  {p}")
    log(f"cache superset: {len(C.CACHE_ITEMIDS)} itemids "
        f"({len(C.FROZEN_PARAMS)} frozen + {len(C.EXTRA_CACHED_ITEMS)} stored-not-trained)")


if __name__ == "__main__":
    main()
