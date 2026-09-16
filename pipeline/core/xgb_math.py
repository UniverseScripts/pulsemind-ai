"""The two XGBoost/logit primitives shared by fitting and by serving.

⚠️ These live here, not in `scoring.py`, because `s13_calibrate` needs them at
fit time and one definition is the point -- and NOT in `s13_calibrate`, because
serving would then import a training stage and drag in Optuna and the bake-off.
`s13_calibrate` re-exports them, so `from ..stages.s13_calibrate import
xgb_predict` keeps working.

Nothing here reads config or touches disk.
"""
from __future__ import annotations

import numpy as np


def _logit(p, eps: float = 1e-12):
    """Log-odds, clipped away from the asymptotes.

    eps is 1e-12 rather than the usual 1e-6 so clipping cannot tie
    apart-but-tiny scores together and perturb average precision. Platt is
    fitted on this transform, so the constant is part of the calibrator's
    identity -- it is persisted alongside the model as `logit_eps`.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def xgb_predict(bst, X, chunk: int = 500_000):
    """Predict in chunks so a multi-million-row pass cannot blow up 8 GB of VRAM.

    `iteration_range` is pinned to the booster's best iteration: anything that
    decomposes or re-scores this model must use the same tree count, or it
    describes a different model from the one that produced the score.
    """
    import xgboost as xgb
    out = []
    for i in range(0, len(X), chunk):
        d = xgb.DMatrix(X.iloc[i:i + chunk], enable_categorical=True)
        out.append(bst.predict(d, iteration_range=(0, bst.best_iteration + 1)))
    return np.concatenate(out)
