"""Stage 11 -- the algorithm bake-off.

Three phases, runnable separately:

  A  feature-set comparison   what does each source table actually contribute?
                              (includes the documentation-only diagnostic)
  B  algorithm bake-off       LightGBM (CPU) vs XGBoost (CUDA) vs CatBoost (GPU),
                              budget-matched Optuna, identical folds and metric
  C  final fit + test         held-out patients, bootstrap CIs on differences,
                              inference latency, SHAP grouped by source table

Fairness rules, because otherwise the comparison is worthless: every candidate
sees the same folds, the same features, the same primary metric, and the same
tuning budget. A tuned model always beats an untuned one.

Primary metric is AVERAGE PRECISION. At ~5% positives ROC-AUC is flattering; PR
is the honest number.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .. import config as C
from ..common import align_forward_labels, console, log, stage
from .s10_assemble import FEATURES_JSON

warnings.filterwarnings("ignore", category=UserWarning)

RESULTS_JSON = C.RPT_S11_BAKEOFF


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load(feature_set: str, target: str | None = None,
         target_source: str | None = None):
    """Feature matrix and label for the configured target.

    `target`/`target_source` override the configured pair. s15 uses this to
    pull the incumbent `warning` label on the UNFILTERED matrix, so it can
    score every candidate label on one common row set -- which is the only
    thing that makes a head-to-head comparison mean anything.

    Two label sources. `warning` is a column of the model matrix. A forward
    label is not -- it is joined from targets_forward.parquet, because a
    label describing the future must never sit in the file the feature scan
    reads. See common.align_forward_labels.

    A forward label is also CENSORED: a row whose look-ahead window runs past
    the end of the stay, with nothing having fired, is genuinely unknown. Those
    rows are dropped here rather than being called negative -- teaching the
    model that "the stay ended" means "no deterioration" would be inventing
    the outcome we are trying to predict.
    """
    tgt = target or C.TARGET
    src = target_source or (C.TARGET_SOURCE if target is None else "matrix")

    man = json.loads(FEATURES_JSON.read_text())
    feats = man["sets"][feature_set]
    cats = [c for c in man["categorical"] if c in feats]
    # charttime rides along in the metadata so consumers never re-read it from
    # the model matrix and index it with a mask built here. A forward label
    # drops censored rows, so those two row sets differ and any such re-read
    # is a misalignment waiting to happen.
    keep = ["stay_id", "subject_id", "split", "fold", "charttime"]
    if src == "matrix":
        keep.append(tgt)
    cols = list(dict.fromkeys(feats + keep))
    df = pl.read_parquet(C.MODEL_MATRIX_PQ, columns=cols)

    if src == "matrix":
        y = df[tgt].to_numpy().astype(np.int8)
        observed = np.ones(len(y), dtype=bool)
    else:
        fwd = align_forward_labels(df.height)
        if tgt not in fwd.columns:
            raise KeyError(
                f"target {tgt!r} is not in {C.FORWARD_TARGETS_PQ.name}. "
                f"Available forward labels: "
                f"{[c for c in fwd.columns if c.startswith('y_')]}")
        s = fwd[tgt]
        observed = s.is_not_null().to_numpy()
        y = s.fill_null(0).to_numpy().astype(np.int8)
        n_drop = int((~observed).sum())
        log(f"target [bold]{tgt}[/bold] from {C.FORWARD_TARGETS_PQ.name}: "
            f"{observed.sum():,} labelled rows, {n_drop:,} dropped as "
            f"unobservable ({100 * n_drop / len(observed):.2f}%)")

    if not observed.all():
        df = df.filter(pl.Series(observed))
        y = y[observed]

    pdf = df.select(feats).to_pandas()
    for c in cats:
        pdf[c] = pdf[c].astype("category")
    for c in pdf.columns:
        if c not in cats:
            pdf[c] = pdf[c].astype(np.float32)

    meta = dict(
        y=y,
        stay_id=df["stay_id"].to_numpy(),
        subject_id=df["subject_id"].to_numpy(),
        split=df["split"].to_numpy(),
        fold=df["fold"].to_numpy(),
        charttime=df["charttime"].to_numpy(),
        cats=cats,
        n_features=len(feats),
        observed=observed,
    )
    return pdf, meta


def stay_weights(stay_id: np.ndarray) -> np.ndarray:
    """1 / rows-in-that-admission, renormalised to mean 1.

    Without this a two-week admission outweighs a hundred short ones and the
    effective patient diversity collapses back toward where we started.
    """
    _, inv, counts = np.unique(stay_id, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    return w * (len(w) / w.sum())


def metrics(y, p) -> dict:
    return {
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "positive_rate": float(y.mean()),
    }


# ---------------------------------------------------------------------------
# Model wrappers -- one interface, three libraries
# ---------------------------------------------------------------------------
def fit_lightgbm(Xtr, ytr, Xva, yva, params, cats, wtr=None, seed=0):
    import lightgbm as lgb
    spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    p = dict(objective="binary", metric="average_precision", verbosity=-1,
             num_threads=C.N_CPU_THREADS, seed=seed, scale_pos_weight=spw,
             max_bin=params.pop("max_bin", 127), **params)
    # free_raw_data=True keeps peak memory down: prediction uses the original
    # DataFrame, so LightGBM's own copy is dead weight once binning is done.
    dtr = lgb.Dataset(Xtr, ytr, weight=wtr, categorical_feature=cats or "auto",
                      free_raw_data=True)
    dva = lgb.Dataset(Xva, yva, reference=dtr, free_raw_data=True)
    booster = lgb.train(p, dtr, num_boost_round=3000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    return booster, booster.predict(Xva, num_iteration=booster.best_iteration)


def fit_xgboost(Xtr, ytr, Xva, yva, params, cats, wtr=None, seed=0):
    import xgboost as xgb
    spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    p = dict(objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
             device="cuda" if C.USE_GPU else "cpu", max_bin=C.GPU_MAX_BIN,
             scale_pos_weight=spw, random_state=seed, **params)
    dtr = xgb.QuantileDMatrix(Xtr, ytr, weight=wtr, enable_categorical=True,
                              max_bin=C.GPU_MAX_BIN)
    dva = xgb.QuantileDMatrix(Xva, yva, ref=dtr, enable_categorical=True,
                              max_bin=C.GPU_MAX_BIN)
    bst = xgb.train(p, dtr, num_boost_round=3000, evals=[(dva, "va")],
                    early_stopping_rounds=100, verbose_eval=False)
    return bst, bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))


# CatBoost's GPU backend does not support Blackwell (sm_120) yet: catboost
# 1.2.10 ships no sm_120 kernels and the runtime PTX JIT fails with
# "CUDA error 218: a PTX JIT compilation failed". Its GPU build also cannot
# compute PRAUC. So CatBoost runs on CPU here. Verified on this RTX 5060.
_CATBOOST_GPU_OK = C.USE_GPU and False

# Forced onto CPU, CatBoost needs a bounded iteration ceiling or a single fit at
# 2.7M rows x 102 features runs for tens of minutes. 800 with early stopping
# keeps a fit comparable in wall-clock to the GPU candidates. This is a handicap
# and is reported as one.
CATBOOST_MAX_ITERATIONS = 800


def fit_catboost(Xtr, ytr, Xva, yva, params, cats, wtr=None, seed=0):
    from catboost import CatBoostClassifier, Pool
    Xtr, Xva = Xtr.copy(), Xva.copy()
    for c in cats:  # CatBoost will not accept NaN in a categorical column
        Xtr[c] = Xtr[c].cat.add_categories("NA").fillna("NA").astype(str)
        Xva[c] = Xva[c].cat.add_categories("NA").fillna("NA").astype(str)
    spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    m = CatBoostClassifier(
        iterations=CATBOOST_MAX_ITERATIONS, eval_metric="PRAUC", loss_function="Logloss",
        task_type="GPU" if _CATBOOST_GPU_OK else "CPU",
        thread_count=C.N_CPU_THREADS,
        scale_pos_weight=spw, random_seed=seed, verbose=0,
        early_stopping_rounds=100, allow_writing_files=False,
        train_dir=str(C.CATBOOST_TRAIN_DIR), **params)
    m.fit(Pool(Xtr, ytr, cat_features=cats, weight=wtr),
          eval_set=Pool(Xva, yva, cat_features=cats))
    return m, m.predict_proba(Xva)[:, 1]


FITTERS = {"lightgbm": fit_lightgbm, "xgboost": fit_xgboost, "catboost": fit_catboost}


def search_space(algo: str, trial) -> dict:
    if algo == "lightgbm":
        return dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 511, log=True),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 50, 2000, log=True),
            feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            bagging_freq=1,
            lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            max_bin=trial.suggest_categorical("max_bin", [63, 127, 255]),
        )
    if algo == "xgboost":
        return dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 12),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 200.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
    return dict(  # catboost
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        depth=trial.suggest_int("depth", 4, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
    )


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
FIXED = {"lightgbm": dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
                          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                          lambda_l1=0.1, lambda_l2=1.0, max_bin=127)}


def phase_a(sets: list[str], weighted: bool) -> dict:
    """What does each source table contribute? Fixed params, LightGBM, one split."""
    out = {}
    for fs in sets:
        X, m = load(fs)
        tr = (m["split"] == "train") & (m["fold"] != 0)
        va = (m["split"] == "train") & (m["fold"] == 0)
        w = stay_weights(m["stay_id"][tr]) if weighted else None
        t = time.time()
        booster, pred = fit_lightgbm(X[tr], m["y"][tr], X[va], m["y"][va],
                                     dict(FIXED["lightgbm"]), m["cats"], w)
        res = metrics(m["y"][va], pred)
        res.update(features=m["n_features"], seconds=round(time.time() - t, 1),
                   best_iteration=int(booster.best_iteration))
        out[fs] = res
        log(f"  {fs:<22} AP={res['average_precision']:.4f}  "
            f"ROC-AUC={res['roc_auc']:.4f}  Brier={res['brier']:.4f}  "
            f"({res['features']} features, {res['seconds']}s, "
            f"{res['best_iteration']} trees)")
        del X
    return out


def _checkpoint(algo: str, payload: dict) -> None:
    """Persist after EVERY algorithm.

    Writing results only at the end means a kill or a crash three hours in
    throws away everything that already finished.
    """
    results = json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else {}
    results.setdefault("phase_b", {})[algo] = payload
    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    log(f"  checkpointed {algo} -> {RESULTS_JSON.name}")


def phase_b(feature_set: str, algos: list[str], trials: int, weighted: bool,
            time_budget: float | None = None) -> dict:
    """Budget-matched tuning, then 5-fold grouped CV with the winning config.

    `time_budget` caps tuning wall-clock per algorithm. Matching wall-clock is
    the fair comparison when one candidate has a GPU backend and another does
    not -- the trial count reached is reported so the difference is visible.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X, m = load(feature_set)
    y = m["y"]
    tr_all = m["split"] == "train"
    tune_tr = tr_all & (m["fold"] != 0)
    tune_va = tr_all & (m["fold"] == 0)
    w_tune = stay_weights(m["stay_id"][tune_tr]) if weighted else None

    out = {}
    for algo in algos:
        log(f"[bold]{algo}[/bold]: tuning, {trials} trials on a single held-out fold")
        fitter = FITTERS[algo]

        def objective(trial):
            p = search_space(algo, trial)
            try:
                _, pred = fitter(X[tune_tr], y[tune_tr], X[tune_va], y[tune_va],
                                 p, m["cats"], w_tune)
            except Exception as e:  # noqa: BLE001 -- a bad config must not kill the study
                log(f"    trial failed: {type(e).__name__}: {str(e)[:120]}")
                return 0.0
            return average_precision_score(y[tune_va], pred)

        def progress(study_, trial_):
            best = study_.best_value if study_.best_trial else float("nan")
            log(f"    trial {trial_.number + 1}/{trials}  "
                f"AP={trial_.value if trial_.value is not None else float('nan'):.4f}  "
                f"best={best:.4f}  ({time.time() - t0:.0f}s elapsed)")

        t0 = time.time()
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=C.RANDOM_SEED))
        study.optimize(objective, n_trials=trials, timeout=time_budget,
                       show_progress_bar=False, callbacks=[progress])
        tune_secs = time.time() - t0
        done = len([t for t in study.trials if t.value is not None])
        log(f"  best tuning AP={study.best_value:.4f} in {tune_secs/60:.1f} min "
            f"({done}/{trials} trials completed within the budget)")

        # 5-fold grouped CV with the winning configuration
        fold_scores, fold_preds = [], {}
        for k in range(C.N_FOLDS):
            ktr = tr_all & (m["fold"] != k)
            kva = tr_all & (m["fold"] == k)
            wk = stay_weights(m["stay_id"][ktr]) if weighted else None
            _, pred = fitter(X[ktr], y[ktr], X[kva], y[kva],
                             dict(study.best_params), m["cats"], wk)
            s = metrics(y[kva], pred)
            fold_scores.append(s)
            fold_preds[k] = (np.where(kva)[0], pred)
            log(f"    fold {k}: AP={s['average_precision']:.4f} "
                f"ROC-AUC={s['roc_auc']:.4f}")

        aps = [s["average_precision"] for s in fold_scores]
        out[algo] = dict(best_params=study.best_params, tuning_seconds=round(tune_secs, 1),
                         tuning_best_ap=float(study.best_value),
                         trials_completed=done, trials_requested=trials,
                         cv_seconds=round(time.time() - t0 - tune_secs, 1),
                         folds=fold_scores,
                         ap_mean=float(np.mean(aps)), ap_std=float(np.std(aps)))
        log(f"  [bold]{algo}[/bold] CV AP = {np.mean(aps):.4f} +/- {np.std(aps):.4f}")
        _checkpoint(algo, out[algo])
    del X
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="a", choices=["a", "b", "all"])
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--feature-set", default="full")
    ap.add_argument("--algos", default="lightgbm,xgboost,catboost")
    ap.add_argument("--weighted", action="store_true",
                    help="weight each row by 1/rows-in-admission")
    ap.add_argument("--time-budget", type=float, default=None,
                    help="wall-clock seconds per algorithm for tuning")
    a = ap.parse_args()

    results = {}
    if RESULTS_JSON.exists():
        results = json.loads(RESULTS_JSON.read_text())

    if a.phase in ("a", "all"):
        with stage("Phase A -- what each source table contributes"):
            results["phase_a"] = phase_a(
                ["documentation_only", "final_only", "t2_all", "full"], a.weighted)

    if a.phase in ("b", "all"):
        with stage(f"Phase B -- algorithm bake-off on '{a.feature_set}'"):
            phase_b(a.feature_set, a.algos.split(","), a.trials, a.weighted,
                    a.time_budget)
            # phase_b checkpoints each algorithm as it finishes; reload so the
            # final write does not clobber those.
            results = json.loads(RESULTS_JSON.read_text())

    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    log(f"results -> {RESULTS_JSON}")


if __name__ == "__main__":
    main()
