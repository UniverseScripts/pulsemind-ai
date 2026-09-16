"""Central configuration for the Pulsemind MIMIC-IV pipeline.

Everything path-, code- and threshold-related lives here so no stage hardcodes
a magic number.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
#
# This repo holds code and results. Everything large or credentialed lives in the
# workspace beside it and is never committed:
#
#   <workspace>/data     MIMIC-IV 3.1 -- 43 GB, PhysioNet DUA, not redistributable.
#                        ⚠️ A PARTIAL extract: 11 of 31 tables since 2026-08-09.
#   <workspace>/build    pipeline artifacts -- ~440 MB, regenerable
#   <workspace>/models   fitted models ~210 MB, plus the 7B weights and the
#                        MedCPT encoders -- 17 GB in total, all re-derivable
#   <repo>/reports       measured results -- small, tracked, the deliverable
#
# Every root is environment-overridable, so a checkout can keep its data
# somewhere else entirely without editing this file.
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = Path(os.getenv("PM_WORKSPACE", REPO_ROOT.parent))

DATA_ROOT = Path(os.getenv("PM_DATA_ROOT", WORKSPACE / "data"))
MIMIC = DATA_ROOT / "mimic-iv-3.1"


# MIMIC-IV ships each table inside a directory of the same name. chartevents is
# nested one level deeper than the rest -- easy to get wrong, so resolve it once.
def mimic_csv(module: str, table: str) -> Path:
    """Resolve a MIMIC table to its CSV, tolerating the repeated-directory nesting.

    Returns the expected path even when the file is absent, so importing this
    module never fails on a checkout without the source tree -- the reports and
    the code should be readable without it. Stages that actually read MIMIC call
    `require_mimic()` first, which fails with a useful message.
    """
    base = MIMIC / module / f"{table}.csv"
    candidates = [
        base / f"{table}.csv" / f"{table}.csv",  # chartevents is triple-nested
        base / f"{table}.csv",
        base,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[1]  # the usual layout; require_mimic() reports it properly


CHARTEVENTS = mimic_csv("icu", "chartevents")
ICUSTAYS = mimic_csv("icu", "icustays")
PROCEDUREEVENTS = mimic_csv("icu", "procedureevents")
INPUTEVENTS = mimic_csv("icu", "inputevents")
D_ITEMS = mimic_csv("icu", "d_items")
PATIENTS = mimic_csv("hosp", "patients")
ADMISSIONS = mimic_csv("hosp", "admissions")
DIAGNOSES_ICD = mimic_csv("hosp", "diagnoses_icd")
TRANSFERS = mimic_csv("hosp", "transfers")
# Height and weight. 306 MB, so reading it costs nothing next to chartevents'
# 40 GB, and it keeps Table 1 on a single provenance path -- raw MIMIC through
# s02 -- rather than mixing in a BigQuery export with a different cohort.
OMR = mimic_csv("hosp", "omr")

MIMIC_TABLES = {
    "icu/chartevents": CHARTEVENTS, "icu/icustays": ICUSTAYS,
    "icu/procedureevents": PROCEDUREEVENTS, "icu/inputevents": INPUTEVENTS,
    "icu/d_items": D_ITEMS, "hosp/patients": PATIENTS,
    "hosp/admissions": ADMISSIONS, "hosp/diagnoses_icd": DIAGNOSES_ICD,
    "hosp/transfers": TRANSFERS, "hosp/omr": OMR,
}


def require_mimic() -> None:
    """Fail early and legibly when the MIMIC source tree is not where we expect.

    Called once by run_all before any stage runs, so a misconfigured data root
    surfaces immediately rather than as a DuckDB IO error six minutes in.
    """
    missing = sorted(n for n, p in MIMIC_TABLES.items() if not p.is_file())
    if missing:
        raise FileNotFoundError(
            f"MIMIC-IV source tables not found under {MIMIC}\n"
            f"  missing: {', '.join(missing)}\n"
            "  Set PM_DATA_ROOT to the directory holding mimic-iv-3.1/, or "
            "PM_WORKSPACE to its parent."
        )


# Reference exports used to verify the local extract reproduces BigQuery
# semantics. Neither is a pipeline INPUT -- the pipeline reads raw MIMIC and
# these say whether it read it correctly.
#   raw-query.csv        -- the frozen eleven, per (stay_id, charttime)
#   static-demographic-  -- Table 1: gender, race, the 17 Charlson categories
#     query.csv             and the index. Its comorbidity flags agree with
#                           ours on 100.00% of joined admissions; its Charlson
#                           INDEX does not, which is how the scoring defects in
#                           charlson.py were found.
BQ_EXPORT = DATA_ROOT / "raw-query.csv"
BQ_STATIC_EXPORT = DATA_ROOT / "static-demographic-query.csv"

# Pipeline outputs. Derived and large -- deliberately outside the repo.
BUILD = Path(os.getenv("PM_BUILD_ROOT", WORKSPACE / "build"))
COHORT_PQ = BUILD / "cohort.parquet"
TS_LONG_PQ = BUILD / "ts_long.parquet"          # the cache -- the only 42 GB scan feeds this
T1_STATIC_PQ = BUILD / "t1_static.parquet"
T2_WIDE_PQ = BUILD / "t2_timeseries.parquet"
T2_IMPUTED_PQ = BUILD / "t2_imputed.parquet"
T3_INTERV_PQ = BUILD / "t3_interventions.parquet"
T4_OUTCOMES_PQ = BUILD / "t4_outcomes.parquet"
MODEL_MATRIX_PQ = BUILD / "model_matrix.parquet"
FOLDS_PQ = BUILD / "folds.parquet"
REFERENCE_STATS_JSON = BUILD / "reference_stats.json"
MANIFEST_DIR = BUILD / "manifests"
REPORTS = REPO_ROOT / "reports"                                  # tracked in git
MODELS = Path(os.getenv("PM_MODELS_ROOT", WORKSPACE / "models"))  # derived, not tracked

SCRATCH = BUILD / "scratch"

for _d in (BUILD, MANIFEST_DIR, REPORTS, MODELS, SCRATCH):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Report paths -- ONE definition each, named for the module that writes them.
#
# ⚠️ A rename that updates the writer and misses a reader is SILENT here: s13's
# tuned_params() falls back to XGB_FALLBACK when the file is missing, so the next
# model trains on default hyperparameters and logs nothing.
#
# The prefix is the producer, so `ls reports/` reads in pipeline order and an
# unfamiliar file's origin is never a guess.
# --------------------------------------------------------------------------
RPT_S11_BAKEOFF = REPORTS / "s11_bakeoff.json"
RPT_S12_EVALUATION = REPORTS / "s12_final_evaluation.json"
RPT_S13_CALIBRATION = REPORTS / "s13_calibration.json"
RPT_S14_FORWARD = REPORTS / "s14_forward_targets.json"
RPT_S15_COMPARISON = REPORTS / "s15_target_comparison.json"
RPT_S16_BANDS = REPORTS / "s16_bands.json"
RPT_S17_RECORDS = REPORTS / "s17_records.json"
RPT_S18_EXPLANATIONS = REPORTS / "s18_explanations.json"
RPT_S19_LLM = REPORTS / "s19_llm_explanations.json"
RPT_S20_CORPUS = REPORTS / "s20_corpus.json"
RPT_S21_EVIDENCE_MD = REPORTS / "s21_evidence_map.md"

# tools/ -- these answer a question and change nothing, so they are prefixed
# `tool_` rather than by a stage number they do not have.
RPT_VERIFY = REPORTS / "verify.json"
RPT_TOOL_ABLATION = REPORTS / "tool_method_ablation.json"
RPT_TOOL_GATE_PIVOT = REPORTS / "tool_gate_pivot.json"
RPT_TOOL_TARGETS = REPORTS / "tool_target_candidates.json"
RPT_TOOL_CAUSAL_PARITY = REPORTS / "tool_causal_parity.json"
RPT_TOOL_VRAM_PROBE = REPORTS / "tool_vram_probe.json"

# --------------------------------------------------------------------------
# Keep every temporary file on D:. C: and D: are partitions of one SSD and a
# spilling DuckDB query or a CatBoost training directory can be many GB.
# ⚠️ Must run before duckdb/matplotlib/numba/catboost are first used.
# --------------------------------------------------------------------------
for _var in ("TMPDIR", "TEMP", "TMP"):
    os.environ[_var] = str(SCRATCH)
os.environ.setdefault("MPLCONFIGDIR", str(SCRATCH / "mpl"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(SCRATCH / "numba"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRATCH / "xdg"))

# LLM weights are ~5 GB and must NOT land in either of the two obvious places:
# the C:\Users\...\.cache default (C: is the smaller partition) or SCRATCH,
# which is deleted between runs and would re-download them every time. MODELS
# is outside the repo, gitignored, and already holds fitted artifacts.
LLM_CACHE = MODELS / "llm"
LLM_CACHE.mkdir(parents=True, exist_ok=True)
# BOTH variables, and set rather than setdefault for the hub cache. HF_HOME
# alone did not hold on 2026-08-08: 14.19 GB of weights appeared under
# C:\Users\...\.cache\huggingface\hub as a second copy and took the system drive
# to 1.23 GB free. HF_HUB_CACHE is the one that directly names the directory, so
# it is pinned explicitly. generate.capability_check() asserts the resolved
# location and refuses to load if it is anywhere else.
os.environ.setdefault("HF_HOME", str(LLM_CACHE))
os.environ["HF_HUB_CACHE"] = str(LLM_CACHE / "hub")

# A 7B in NF4 needs a ~4.8 GB block on a card with ~6.4 GB free once the display
# has taken its share. The default caching allocator asks for that contiguously
# and fails on fragmentation rather than on capacity; expandable segments let it
# grow instead. Must be set before torch allocates anything.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import tempfile as _tempfile  # noqa: E402

_tempfile.tempdir = str(SCRATCH)

CATBOOST_TRAIN_DIR = SCRATCH / "catboost"  # never let CatBoost write to CWD/C:

# --------------------------------------------------------------------------
# DuckDB
# --------------------------------------------------------------------------
DUCKDB_MEMORY_LIMIT = os.getenv("PM_DUCKDB_MEM", "20GB")
DUCKDB_THREADS = int(os.getenv("PM_DUCKDB_THREADS", "24"))
DUCKDB_TEMP_DIR = BUILD / "duckdb_tmp"
DUCKDB_TEMP_DIR.mkdir(parents=True, exist_ok=True)
DUCKDB_MAX_TEMP = os.getenv("PM_DUCKDB_MAX_TEMP", "12GB")

# --------------------------------------------------------------------------
# Cohort
# --------------------------------------------------------------------------
ITEM_INVASIVE_VENT = 225792
ITEM_NONINVASIVE_VENT = 225794

# --------------------------------------------------------------------------
# THE FROZEN ELEVEN -- the model's only time-series inputs. Immutable.
# --------------------------------------------------------------------------
FROZEN_PARAMS: dict[str, int] = {
    "spo2": 220277,
    "fio2": 223835,
    "flow_rate": 224691,
    "peep": 220339,
    "pip": 224695,
    "respiratory_rate_total": 224690,
    "minute_volume": 224687,
    "tidal_volume_observed": 224685,
    "etco2": 228640,
    "inspiratory_ratio": 226873,
    "expiratory_ratio": 226871,
}
# Column order must match data/raw-query.csv exactly for the verification diff.
FROZEN_ORDER = list(FROZEN_PARAMS.keys())

# Some frozen columns are sourced from more than one itemid, taken in preference
# order (first non-null wins).
#
# `peep`: verified against data/raw-query.csv -- on all 14,288 rows where our
# PEEP-set-only value disagreed with the export, the export matched Total PEEP
# Level (224700) exactly, and none matched PEEP set (220339). So the frozen
# BigQuery definition prefers Total PEEP, which is also the better measure
# clinically: it is set PEEP plus any auto-PEEP, i.e. the pressure actually in
# the lung. 224700 is already in the cache, so this costs no rescan.
FROZEN_PARAM_SOURCES: dict[str, list[int]] = {n: [i] for n, i in FROZEN_PARAMS.items()}
FROZEN_PARAM_SOURCES["peep"] = [224700, 220339]

FROZEN_SOURCE_ITEMIDS: list[int] = sorted(
    {i for ids in FROZEN_PARAM_SOURCES.values() for i in ids})

# --------------------------------------------------------------------------
# Cache superset -- STORED, NOT TRAINED ON.
# Extra codes are ~free during a scan that already reads every byte, but adding
# one later would cost another full pass. This does not change the model.
# --------------------------------------------------------------------------
EXTRA_CACHED_ITEMS: dict[str, int] = {
    "o2_flow": 223834,
    "total_peep": 224700,
    "respiratory_rate": 220210,
    "plateau_pressure": 224696,
    "mean_airway_pressure": 224697,
    "compliance": 229661,
    "tidal_volume_set": 224684,
    "spont_tidal_volume": 224421,
    "spont_respiratory_rate": 224422,
    "pf_ratio_value": 229393,
    "heart_rate": 220045,
    "temperature_f": 223761,
    "temperature_c": 223762,
    "arterial_bp_mean": 220052,
    "wbc": 220546,
    "resistance": 220283,
    "ventilator_mode": 223849,
    "ventilator_type": 223848,
    "o2_delivery_device": 226732,
    # Body metrics -- static per admission, but only chartevents records them
    # for ICU patients. hosp/omr was tried first and reaches just 51.5% of the
    # cohort, because omr is an OUTPATIENT record: a stay has a row there only
    # if the patient also had clinic care at this hospital. That missingness
    # is a care-continuity signal, which is precisely the documentation
    # confound this whole target switch exists to escape. Charted admission
    # weight has no such structure.
    "admission_weight_kg": 226512,
    "admission_weight_lb": 226531,
    "daily_weight_kg": 224639,
    "height_cm_item": 226730,
    "height_in_item": 226707,
}
CACHE_ITEMS: dict[str, int] = {**FROZEN_PARAMS, **EXTRA_CACHED_ITEMS}
CACHE_ITEMIDS: list[int] = sorted(set(CACHE_ITEMS.values()))
ITEMID_TO_NAME: dict[int, str] = {v: k for k, v in CACHE_ITEMS.items()}

# Read by s02 out of the cache, in preference order within each metric.
BODY_WEIGHT_ITEMIDS = {226512: 1.0, 224639: 1.0, 226531: 0.45359237}  # -> kg
BODY_HEIGHT_ITEMIDS = {226730: 1.0, 226707: 2.54}                     # -> cm

# --------------------------------------------------------------------------
# Physiologic plausibility ranges. Out-of-range -> NULL, so bad values flow into
# the missingness machinery instead of corrupting medians. Absent from the
# original pipeline entirely.
# --------------------------------------------------------------------------
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "spo2": (0.0, 100.0),
    "fio2": (21.0, 100.0),
    "flow_rate": (0.0, 150.0),
    "peep": (0.0, 35.0),
    "pip": (0.0, 80.0),
    "respiratory_rate_total": (0.0, 70.0),
    "minute_volume": (0.0, 40.0),
    "tidal_volume_observed": (0.0, 2000.0),
    "etco2": (0.0, 100.0),
    "inspiratory_ratio": (0.0, 10.0),
    "expiratory_ratio": (0.0, 20.0),
}

# FiO2 is charted both as a fraction (0.21-1.0) and a percentage (21-100).
# Values at or below this bound are treated as fractions and scaled up.
FIO2_FRACTION_MAX = 1.0

# --------------------------------------------------------------------------
# Imputation
# --------------------------------------------------------------------------
LOCF_CUTOFF_MIN = 240.0  # a setting from >4h ago is not "current"
SUFFIXES = ("_observed", "_delta_t_min", "_locf", "_structurally_missing_in_stay", "_final")


def _flag(name: str, default: bool) -> bool:
    """Env-overridable boolean, so the gate harness can rebuild any prior state."""
    return os.getenv(name, "1" if default else "0") == "1"


# --------------------------------------------------------------------------
# Table 1 (static) -- see s02_table1_static.py
#
# S02_FINAL_COHORT: build Table 1 on the FINAL cohort (39,394 admissions), not
# the strict one (31,969). ⚠️ Turning this off reinstates a measured leak --
# 946,390 rows with all 33 static features NULL, a null pattern that identifies
# the cohort arm exactly. Full account in `s02_table1_static.py`'s docstring.
#
# CHARLSON_HIERARCHY: standard Charlson counts only the higher member of each
# graded pair (mild/severe liver, uncomplicated/complicated diabetes,
# malignancy/metastatic). charlson.py summed both, inflating the index.
#
# STATIC_BODY_METRICS: height and weight out of the chartevents cache, plus
# the predicted body weight derived from them -- which is what makes tidal
# volume comparable between patients. hosp/omr was the obvious source and is
# the wrong one: it reaches 51.5% of the cohort against 99.3% for charted
# admission weight, and its missingness tracks whether the patient also had
# outpatient care here.
# --------------------------------------------------------------------------
S02_FINAL_COHORT = _flag("PM_S02_FINAL_COHORT", True)
CHARLSON_HIERARCHY = _flag("PM_CHARLSON_HIERARCHY", True)
STATIC_BODY_METRICS = _flag("PM_STATIC_BODY_METRICS", True)

# Dobutamine is an inotrope, not a vasopressor, but it sat in the vasopressor
# regex. Splitting it into its own group corrects both the `d_vaso` label arm
# and the `vasopressor_*` features without discarding the drug.
SPLIT_INOTROPE = _flag("PM_SPLIT_INOTROPE", True)

# Tidal volume as mL/kg predicted body weight -- the ARDSNet lung-protective
# quantity. Absolute mL means different things in a 150 cm and a 190 cm
# patient, so the raw column is not comparable across patients and this is.
ENABLE_PBW = _flag("PM_ENABLE_PBW", True)

# --------------------------------------------------------------------------
# Modelling
#
# The target is `y_resp_6h` -- the respiratory arm of composite deterioration at
# a 6 h look-ahead. See the forward-targets block below for the label itself.
#
# `warning` is still built and still lands in the model matrix, because
# verify.py diffs it against the BigQuery export to prove the extract is
# faithful. It is excluded from every feature set by name -- see LEGACY_TARGET
# and the guards in s10_assemble / verify.
#
# TARGET_SOURCE says WHERE the label lives:
#   "matrix"  -- a column of model_matrix.parquet (how `warning` worked)
#   "forward" -- a column of targets_forward.parquet, joined at training time
# A forward label must never be reachable as a feature, so it is deliberately
# kept out of the model matrix and joined on (stay_id, charttime) instead.
# --------------------------------------------------------------------------


GROUP_KEY = "subject_id"   # patient, NOT stay -- one patient can have several stays
N_FOLDS = 5
RANDOM_SEED = 42

LEGACY_TARGET = "warning"  # kept as a column for verification; never a feature

PRIMARY_HORIZON_H = 6      # of HORIZONS_H below; 2 and 12 justify the choice

# The trained label. Respiratory arm only: the circulatory arm is computed and
# reported beside it, but 29.1% of the four-arm label's positives were
# circulatory-only, and this is a ventilation monitor.
TARGET = os.getenv("PM_TARGET", f"y_resp_{PRIMARY_HORIZON_H}h")
TARGET_SOURCE = "matrix" if TARGET == LEGACY_TARGET else "forward"

# --------------------------------------------------------------------------
# Forward-looking targets (stage 14) -- candidate D, composite deterioration
#
# `warning` is a caregiver documentation flag: 84.6% of its achievable AP skill
# (95.2% of ROC-AUC skill) is reachable from charting pattern alone. D asks a
# different question -- "will this patient deteriorate in the next H hours?" --
# built from measured values and orders rather than the act of charting.
#
# Two of the four components are clinician RESPONSES rather than patient states,
# so D carries some reverse-causation exposure. Components are therefore scored
# individually, not just as an OR, so the exposure is measured and not asserted.
# --------------------------------------------------------------------------
FORWARD_TARGETS_PQ = BUILD / "targets_forward.parquet"

HORIZONS_H = (2, 6, 12)      # 6 is trained; 2 and 12 justify the choice

D_FIO2_RISE = 20.0           # percentage points above the setting in force
D_PEEP_RISE = 3.0            # cmH2O above the setting in force
D_SPO2_BELOW = 88.0          # the one component that is a patient state
STRICT_BASELINE_AGE_MIN = 60.0  # `y_strict`: escalation judged only against a fresh setting

# Baselines come from the LOCF value (the setting currently in force, capped at
# LOCF_CUTOFF_MIN) because that is what a clinician actually sees. Forward values
# must be MEASURED, never imputed -- an event has to be observed to count.

# ---- Tightenings, from the component audit -------------------------------
#
# The audit found two of the four arms substantially artifact:
#
#   FiO2 -- 53.9% of escalations land at exactly 100% and 48.4% are already
#   back near baseline by the next reading. Both signatures are the same
#   event: pre-oxygenation before endotracheal suctioning. A transient spike
#   is not deterioration, so the rise must still be there at the next MEASURED
#   reading. One mechanism kills both signatures.
#
#   Vasopressor -- only 21.4% of flips are the stay's first pressor and 58.4%
#   are restarts within 4 h. MIMIC chunks one continuous infusion into ~14.9
#   distinct start times, so "not running now, running later" fires on pauses.
#   A flip counts only if it is the stay's first or follows a real off-period.
#
# Both default ON. Turn off via env to reproduce the pre-tightening label.
D_FIO2_PERSIST = _flag("PM_D_FIO2_PERSIST", True)
D_VASO_STRICT = _flag("PM_D_VASO_STRICT", True)
D_VASO_OFF_MIN = 360.0       # 6 h off before a restart counts as a new event

# Prevalence sanity bound asserted in s14. The respiratory arm alone is
# materially rarer than the four-arm composite, so the floor is lower than the
# 0.02 that suited the composite.
D_PREVALENCE_BOUNDS = (0.005, 0.35)

# --------------------------------------------------------------------------
# Calibration (stage 13)
#
# AP and ROC-AUC measure RANK and are invariant under any monotone rescaling of
# the score. The product does not consume a rank -- a screen applies a threshold
# to `risk_prob`, which is a test on the LEVEL. Stage 13 measures the level and
# fits the map that makes it mean something.
#
# Two folds are carved out of the training patients. Fold 3 drives early
# stopping; fold 4 fits the calibrator. Neither is ever the test set: fitting a
# calibrator on test is precisely the defect in training_calinerating.ipynb.
# --------------------------------------------------------------------------
EARLY_STOP_FOLD = 3        # chooses the tree count
CALIB_FOLD = 4             # fits the calibrator -- unseen by the model
TRAIN_FOLDS = (0, 1, 2)

SERVING_THRESHOLD = 0.70   # the constant in bki/backend/main.py:85

# Alerts per ventilated patient-day. A bedside monitor competes for attention,
# so the threshold is chosen against an alert budget rather than picked round.
ALERT_BUDGETS = (1.0, 2.0, 4.0)
ALERT_DEDUP_MINUTES = 60   # rows are irregularly spaced; count alert-hours, not rows

# Named for the label they were fitted on. Every band and threshold is a
# property of (model, label, cohort), so a model file that does not say which
# label it predicts is a loaded gun. The `warning` artifacts stay on disk
# alongside these, which is what makes the switch reversible without a retrain.
MODEL_XGB = MODELS / f"xgboost_{TARGET}.ubj"
MODEL_LGB = MODELS / f"lightgbm_{TARGET}.txt"
CALIBRATOR_PKL = MODELS / f"calibrator_{TARGET}.joblib"
OPERATING_POINT_JSON = MODELS / f"operating_point_{TARGET}.json"

# --------------------------------------------------------------------------
# Risk bands -- the pipeline's OUTPUT CONTRACT
#
# The four labels are consumed downstream by the LLM + RAG explanatory layer, so
# they are an interface, not a badge colour. An LLM handed the bare string "HIGH"
# invents what HIGH means and sounds confident doing it, which is why the band
# ships with its measured event rate and the envelope that rate is allowed to
# occupy. See pipeline/bands.py for the machine and s16_bands.py for the fit.
#
# Renaming a band is expensive once prompts and RAG retrieval keys are built on
# it; adding a field is cheap. These names are final.
# --------------------------------------------------------------------------
BAND_NAMES = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
# `imputed_share` and `documentation_share` are taken over one denominator on
# disjoint feature sets, and `attribution_total` IS that denominator -- so a
# consumer holding only the top-8 contributors can still express one as a share
# of the whole decision. All three need the full 110-column contribution matrix,
# which exists only inside s17.
RISK_SCHEMA_VERSION = "1.2.0"
BAND_TABLE_JSON = MODELS / f"risk_bands_{TARGET}.json"

# Emitted records carry per-patient MIMIC values keyed to stay_id, so they are
# DUA-covered and belong in build/ (gitignored, outside the repo) -- NEVER in
# reports/, which is tracked. reports/records.json gets aggregates only.
# JSONL rather than Parquet: the record is a nested contract, not a table, and
# its consumer builds prompts one record at a time. Gzipped, so a golden set
# stays around 15 MB on a machine with little headroom.
RECORDS_JSONL = BUILD / "risk_records.jsonl.gz"
RECORD_SAMPLE_STAYS = 500   # ~70k records; a golden set for prompt work, not a dump
CONTRIB_TOP_K = 8           # contributors kept per reading, by |attribution|

# sigmoid(sum(contribs) + bias) against the score the record reports. Measured
# max discrepancy is 1.2e-06, from two float32 sources that XGBoost gives no way
# around: summing 110 float32 SHAP terms (9.7e-07) and the float32 probability
# round-trip through logit that Scorer.score performs to stay bit-identical with
# s13 (5.3e-07). Neither is a modelling error.
#
# The bound is set by what it has to guarantee, not by what makes the assertion
# pass: at 1e-05 it is ~3,000x smaller than the tightest feature of the band
# table (the 0.0313 gap between the MEDIUM cut and its deadband floor), so a
# reconstruction discrepancy can never move a band.
CONTRIB_RECON_TOL = 1e-5

# --------------------------------------------------------------------------
# THE EXPLANATION LAYER -- s18
#
# Display names and units for the frozen eleven. FROZEN_PARAMS is name -> itemid
# and nothing else in this pipeline knows that peep is cmH2O, which is fine for
# a model and useless for a sentence a clinician reads. ASCII on purpose: these
# strings reach a Windows console.
# --------------------------------------------------------------------------
PARAM_DISPLAY: dict[str, tuple[str, str, int]] = {
    "spo2":                   ("SpO2", "%", 0),
    "fio2":                   ("FiO2", "%", 0),
    "flow_rate":              ("flow rate", "L/min", 1),
    "peep":                   ("PEEP", "cmH2O", 0),
    "pip":                    ("PIP", "cmH2O", 0),
    "respiratory_rate_total": ("respiratory rate", "/min", 0),
    "minute_volume":          ("minute volume", "L/min", 1),
    "tidal_volume_observed":  ("tidal volume", "mL", 0),
    "etco2":                  ("EtCO2", "mmHg", 0),
    "inspiratory_ratio":      ("I:E inspiratory part", "", 1),
    "expiratory_ratio":       ("I:E expiratory part", "", 1),
}

# LEGAL: when a reading does not clear the data floor, Pulsemind says so rather
# than explaining it. Degrade explicitly -- never emit a synthesised rationale in
# a slot that is meant to signal absence.
INSUFFICIENT_DATA_TEXT = "Insufficient data"
# A DIFFERENT condition, and not a data problem: the reading is fit to explain
# and the generator is down. Band, risk, telemetry and contributors all still
# emit; only the narrative is withheld.
EXPLANATION_UNAVAILABLE_TEXT = (
    "LLM explanation and recommendations are not available")

# --------------------------------------------------------------------------
# The data-sufficiency floor -- MEASURED, not chosen.
#
# Both signals are shares of the same |attribution| denominator over disjoint
# feature sets (see s17_records): imputed_share is the part of the score that
# came from cohort defaults this patient never had, documentation_share the part
# that came from charting behaviour rather than physiology. Above either
# threshold the score is not substantially an assessment of this patient.
#
# On the golden set this suppresses 6.7% of readings -- LOW 8.1%, MEDIUM 3.4%,
# HIGH 2.8%, CRITICAL 0.5% -- and the suppressed readings carry 0.41x the
# forward-label prevalence of the kept ones. It removes the uninformative, not
# the sick. No admission is suppressed in full; the median stay loses 1.2%.
#
# !! THERE IS DELIBERATELY NO STALENESS GATE. `attribution_age_min` is in the
# record and reads like an obvious third condition -- a score driven by values
# two days old is not a current assessment. Measured, it is the opposite:
# readings with attribution_age > 2880 min carry 1.38x the label prevalence of
# the rest, so gating on staleness would suppress readings at ABOVE the base
# deterioration rate. Staleness is DISCLOSED instead -- every parameter carries
# its own age_min into the payload -- and never gated.
# --------------------------------------------------------------------------
SUFFICIENCY_MAX_IMPUTED_SHARE = 0.30
SUFFICIENCY_MAX_DOC_SHARE = 0.30

EXPLANATIONS_JSONL = BUILD / "explanations.jsonl.gz"   # DUA -- build/, not reports/
EXPLAIN_SCHEMA_VERSION = "1.0.0"
EXPLAIN_CONTRIBUTOR_K = 5      # contributors rendered into the payload, of CONTRIB_TOP_K
EXPLAIN_MUTATION_SAMPLE = 200  # records the adversarial self-check runs over

# --------------------------------------------------------------------------
# THE GENERATOR -- s19
#
# LOCAL INFERENCE ONLY. The golden set is DUA-covered, so handing it to a
# hosted API would redistribute credentialed MIMIC data. This is a constraint,
# not a preference.
#
# Qwen2.5-7B-Instruct is Apache-2.0 and ungated: no HF token, no licence click.
# NF4 puts a 7B in ~4.5 GB of weights against the ~6.5 GB this card can offer;
# `none` is the fp16 fallback for a 3B if quantisation ever stops working here.
# --------------------------------------------------------------------------
LLM_MODEL_ID = os.getenv("PM_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
LLM_REVISION = os.getenv("PM_LLM_REVISION", "main")
LLM_QUANT = os.getenv("PM_LLM_QUANT", "nf4")          # nf4 | none
LLM_PREFLIGHT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"    # ~350 MB toolchain probe
# WHERE THE EMBEDDING TABLE LIVES ONCE LOADED. `cpu` buys ~600 MiB of headroom
# during generation and makes it 37% faster (14.4/13.2/13.3 s against
# 25.6/17.8/21.6 s), at byte-identical output.
#
# ⚠️ It does NOT move the load PEAK, so it cannot lower LLM_MIN_FREE_VRAM_MIB:
# the table is on the card while `from_pretrained` runs, whatever happens to it
# afterwards. And ⚠️ the driver returns only 130 MiB of the 1039 MiB freed, so
# reading right after the load says "not worth it" -- that is the wrong place to
# read. Mechanism, the arena that explains the gap, and the `model.device` trap
# it sets: `core/generate.py:_host_embedding`.
LLM_EMBED_DEVICE = os.getenv("PM_LLM_EMBED_DEVICE", "cpu")   # cpu | cuda
#: Free VRAM the 7B needs, from the DRIVER, before a load may start. ONE
#: definition, two readers: `s19_generate`'s capability check and the demo
#: service's `explanation.generator()`. ⚠️ Never let a second copy exist -- two
#: independently-declared gates drift, and each passes its own check while doing so.
#:
#: MEASURED, not bracketed. `tools/vram_probe.py` samples the driver at 5 Hz
#: across the load; three consecutive runs peaked at 6059, 6171 and 6239 MiB,
#: spread 180. 6420 = max peak + spread. The full table, the accepted risk and
#: the reason the peak does not move with LLM_EMBED_DEVICE are on
#: `MIN_FREE_VRAM_MIB` in `pulsemind_demo/back-end/pythonService/explanation.py`.
LLM_MIN_FREE_VRAM_MIB = 6420
LLM_MAX_NEW_TOKENS = 220
# Greedy, fixed seed. Non-determinism would make a prompt change unattributable,
# and attributing changes is the entire purpose of the grounding checker.
LLM_SEED = 20260808
# Sampled, and the stage says so: the full set at even a few seconds a record is
# tens of hours. This is a golden set for prompt work, not a production run.
LLM_SAMPLE = 200
LLM_EXPLANATIONS_JSONL = BUILD / "llm_explanations.jsonl.gz"   # DUA -- build/
LLM_SCHEMA_VERSION = "1.0.0"

# --------------------------------------------------------------------------
# THE GUIDELINE CORPUS -- s20
#
# Retrieval grounding. The corpus holds published guideline text and NO patient
# data, which is why its derived artifacts may be read and reviewed freely --
# the opposite of build/risk_records.jsonl.gz.
#
# TWO source directories on purpose. `rag document` is tracked in git and
# already public on two remotes; anything added now goes to an UNTRACKED
# directory instead, and the manifest records filename + SHA-256 + source URL
# so the corpus is reproducible without redistributing anyone's PDF.
# --------------------------------------------------------------------------
RAG_DOCS_DIR = REPO_ROOT / "rag document"        # tracked; the space is load-bearing
RAG_EXTRA_DIR = Path(os.getenv("PM_CORPUS_ROOT", WORKSPACE / "corpus"))
RAG_EXTRA_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_CHUNKS_JSONL = BUILD / "corpus_chunks.jsonl"
CORPUS_REPORT_JSON = RPT_S20_CORPUS
CORPUS_SCHEMA_VERSION = "1.0.0"

# THE ADMISSION RULE. A document is admitted only if it is registered here with
# a named issuing body and a year. s20 REFUSES any PDF it finds that is not in
# this table rather than ingesting it silently -- an unattributable passage is
# worse than a missing one, because it still gets quoted to a clinician.
#
# `role` decides how the document is read, and it is a curation judgement rather
# than something detectable from the text:
#   surveillance -- defines vocabulary and thresholds. Contains NO recommendations.
#   guideline    -- GRADE-rated "We recommend / We suggest" statements.
#   protocol     -- operational outline steps with concrete parameter targets.
CORPUS_DOCS: dict[str, dict[str, str]] = {
    "nhsn_vae_vap.pdf": {
        "key": "nhsn_vae",
        "title": "Ventilator-Associated Event (VAE)",
        "issuer": "CDC / National Healthcare Safety Network",
        "year": "2026",
        "role": "surveillance",
        "cite": "NHSN VAE protocol (Jan 2026)",
        "url": "https://www.cdc.gov/nhsn/pdfs/pscmanual/10-vae_final.pdf",
    },
    "aarc_clinical_guideline.pdf": {
        "key": "aarc_pva",
        "title": "AARC Clinical Practice Guideline: Patient-Ventilator Assessment",
        "issuer": "American Association for Respiratory Care",
        "year": "2024",
        "role": "guideline",
        "cite": "AARC CPG, Respir Care 2024;69(8):1042-1054",
        "url": "https://rc.rcjournal.com/content/69/8/1042",
    },
    "adult_ventilator_protocols.pdf": {
        "key": "aarc_protocol",
        "title": "AARC Adult Mechanical Ventilator Protocols",
        "issuer": "AARC Protocol Committee, Subcommittee Adult Critical Care",
        "year": "2003",
        "role": "protocol",
        "cite": "AARC Adult Mechanical Ventilator Protocols v1.0a (2003)",
        "url": "https://www.aarc.org/wp-content/uploads/2014/08/vent_protocols.pdf",
    },
}

# MEASURED, not assumed. pypdfium2 is the text extractor and pdfplumber supplies
# per-character font metadata for heading detection; both were probed against
# all three PDFs before being chosen (2026-08-09):
#
#   * pdfplumber tears subscripts onto their own line -- NHSN "8 cmH O" + "2",
#     AARC "assessment of P" + "plat". pypdfium2 returns "cmH2O" and "Pplat".
#     Those are clinical parameter names inside the text that gets quoted
#     verbatim, so this is a correctness difference, not a tidiness one.
#   * ⚠️ Any extractor with a layout model that REFLOWS text is disqualified
#     here, however good it is: the design rests on byte-identical quoting.
#
# Three extraction defects were measured and are corrected in s20:
#   * U+00BC is a mis-mapped '=' in the AARC subset font ("VT ¼ tidal volume",
#     "n ¼ 2,822"), 30 occurrences, 29 of them space-flanked, zero in the other
#     two documents. NFKC would turn it into "1/4", so it is substituted first.
#   * NHSN formula blocks come back as doubled MATHEMATICAL ITALIC letters
#     ("SIR = OOOOOOOO (OO)HHHH"). Unrecoverable, so those runs are excised and
#     counted rather than normalised into plausible-looking garbage.
#   * U+FFFE is "a hyphen at a line break" and is genuinely ambiguous --
#     "rec|ommendations" wants a join, "ventilator|associated" wants a hyphen.
#     Resolved against the corpus's own vocabulary: of 197 breaks, 149 join,
#     12 hyphenate, 0 are attested both ways, 36 are attested neither way.
#     Unresolved cases default to JOIN (28 of the 36 are word continuations)
#     and EVERY one is listed in reports/corpus.json for review.
CORPUS_HYPHEN_DEFAULT_JOIN = True
CORPUS_MATH_ITALIC_MAX_RUN = 3   # >N math-italic chars in a row = a dead formula

# Chunking. MedRAG chunks paragraph-wise and splices the heading path onto the
# snippet as its title -- contextual retrieval without paying an LLM to write
# the context. Its measured snippet sizes are short: 119 tokens (StatPearls,
# the closest analogue to a guideline), 182 textbooks, 296 PubMed abstracts.
CORPUS_CHUNK_MIN_WORDS = 25
CORPUS_CHUNK_TARGET_WORDS = 220
CORPUS_CHUNK_MAX_WORDS = 320

# Type priority, applied as a structural filter on chunk kind rather than as a
# rerank (ClinicBot, arXiv 2605.00846). Order is load-bearing: the most
# clinically actionable and citable content has to dominate the bundle.
CORPUS_KIND_PRIORITY = ("recommendation", "table", "prose")

# --------------------------------------------------------------------------
# THE EVIDENCE MAP -- s21
#
# The query space is CLOSED: 11 frozen parameters x 5 derived forms, 4 bands,
# kind in {physiology, documentation}. Nobody types a question. So the entire
# retrieval result set is materialised here at build time, reviewed once, and
# frozen -- and inference is a dict lookup with zero VRAM and zero latency.
# --------------------------------------------------------------------------
EVIDENCE_MAP_JSON = MODELS / "evidence_map.json"     # the contract artifact
EVIDENCE_MAP_MD = RPT_S21_EVIDENCE_MD                # the human review document
EVIDENCE_SCHEMA_VERSION = "1.0.0"

EVIDENCE_CONTEXT_K = 1     # definitional passages attached per reading
EVIDENCE_ACTIONS_K = 2     # graded action statements attached per reading

# Retrieval vocabulary. PARAM_DISPLAY says what a clinician should READ; this
# says what the guideline literature CALLS the same thing, which is not the same
# list -- the corpus writes "positive end-expiratory pressure", "FIO2", "V T",
# "plateau pressure", never "peep" or "fio2".
#
# ANCHOR TERMS ARE A HARD GATE, not a ranking hint. A passage is admissible for
# a parameter only if it actually mentions that parameter. On a closed query set
# that is strictly better than a similarity threshold: it is decidable, it is
# reviewable by eye, and it makes "no adequate passage" an honest outcome rather
# than whatever scored highest among things that were all wrong.
PARAM_QUERY_TERMS: dict[str, tuple[str, ...]] = {
    "spo2":                   ("SpO2", "oxygen saturation", "pulse oximetry",
                               "oximetry", "saturation", "hypoxemia", "normoxemia"),
    "fio2":                   ("FiO2", "FIO2", "oxygen concentration",
                               "fraction of inspired oxygen", "inspired oxygen"),
    # NOT bare "flow" -- it matches "flow-cycled", a mode descriptor rather than
    # a flow rate. NOT "L/min" either: it is the unit of MINUTE VENTILATION as
    # well, and it made the AARC minute-ventilation formula ("4.0 x BSA = VE
    # (L/min)") the top action for flow rate at all four bands. Both were caught
    # by reading reports/evidence_map.md, not by any score. The consequence is
    # that flow rate has no coverage in this corpus, which is the true answer.
    "flow_rate":              ("flow rate", "inspiratory flow", "peak flow"),
    "peep":                   ("PEEP", "positive end-expiratory pressure",
                               "end-expiratory pressure", "auto-PEEP", "CPAP"),
    # NOT "plateau pressure" / "Pplat". They are a DIFFERENT measurement from
    # peak inspiratory pressure, and listing them here made the AARC plateau-
    # pressure recommendation the top action for PIP at two bands -- a clinical
    # conflation, produced by the retrieval vocabulary rather than by the model.
    # Plateau pressure is not one of the frozen eleven; it is reachable through
    # tidal volume and lung-protective ventilation, which is where it belongs.
    "pip":                    ("PIP", "peak inspiratory pressure", "peak pressure",
                               "peak airway pressure"),
    "respiratory_rate_total": ("respiratory rate", "breathing frequency",
                               "breaths/minute", "breaths per minute", "tachypnea"),
    "minute_volume":          ("minute ventilation", "minute volume", "VE"),
    "tidal_volume_observed":  ("tidal volume", "VT", "mL/kg", "lung-protective",
                               "lung protective", "predicted body weight"),
    "etco2":                  ("EtCO2", "end-tidal", "capnography", "carbon dioxide"),
    # NOT bare "inspiratory" / "expiratory" -- they match "pressure targeted,
    # flow-cycled" prose and "end-expiratory pressure", which is PEEP.
    "inspiratory_ratio":      ("I:E", "I:E ratio", "inspiratory time",
                               "inspiratory-to-expiratory"),
    "expiratory_ratio":       ("I:E", "I:E ratio", "expiratory time",
                               "inspiratory-to-expiratory"),
}

# What is worth SUGGESTING differs by band, and this is the only place band
# enters retrieval. Weaning readiness is the right thought at LOW and the wrong
# one at CRITICAL; escalation is the reverse.
BAND_QUERY_INTENT: dict[str, tuple[str, ...]] = {
    "LOW":      ("weaning", "liberation", "spontaneous breathing trial",
                 "extubation", "readiness", "reduce support"),
    "MEDIUM":   ("assessment", "assess", "monitoring", "evaluate", "documenting"),
    "HIGH":     ("lung-protective", "adjust", "escalation", "increase support",
                 "ventilator adjustments", "assess"),
    "CRITICAL": ("escalation", "urgent", "deterioration", "physician",
                 "lung-protective", "assess"),
}

# The two feature KINDS carried in the payload. `documentation` is the one that
# needs saying out loud: 18.1% of this model's skill comes from charting
# behaviour, and a clinician reading a risk board deserves the surveillance
# definition of what is and is not being charted.
KIND_QUERY_TERMS: dict[str, tuple[str, ...]] = {
    "physiology":    ("patient-ventilator assessment", "assessment",
                      "physiologic", "clinical assessment"),
    "documentation": ("documentation", "documenting", "recorded", "charting",
                      "surveillance", "data collection", "reporting"),
}

# Lexical is the PRIMARY channel: sqlite FTS5 is compiled in, costs nothing,
# and MedRAG measures the best biomedical dense retriever beating BM25 by 0.01
# points on MedCorp (70.06 vs 70.05). Dense is a cheap second opinion, not the
# backbone -- and if it will not load, s21 records that and ships lexical-only.
RAG_DENSE_ENABLED = os.getenv("PM_RAG_DENSE", "1") == "1"
RAG_QUERY_ENCODER = "ncbi/MedCPT-Query-Encoder"
RAG_DOC_ENCODER = "ncbi/MedCPT-Article-Encoder"
RAG_CROSS_ENCODER = "ncbi/MedCPT-Cross-Encoder"
RAG_DENSE_DEVICE = "cpu"   # one build-time pass over ~1k chunks; never contends with the 7B
RAG_RERANK_TOP_N = 12      # candidates handed to the cross-encoder

# The quality flag is a COUNT, not a score, and that is deliberate. Fusing a
# normalised BM25 with a raw MedCPT cosine mixes incomparable scales: BERT [CLS]
# cosines sit at 0.7-0.9 almost regardless of relevance, so enabling the dense
# channel moved every fused score above any fixed threshold and took the
# "weak match" count from 11 to 0 with no passage having changed. Ranking is
# therefore done by reciprocal rank fusion, which has no units, and the flag is
# the one signal a reviewer can independently check: how many times the passage
# actually names the parameter. Once is usually in passing.
EVIDENCE_MIN_ANCHOR_HITS = 2

# Review gate. An entry flagged during review and not signed off is emitted with
# its passage SUPPRESSED, not dropped -- a missing key has to show up in the
# output rather than disappear from it.

# Each cut is solved to an alert budget counted in PROMOTION EVENTS per
# ventilated patient-day -- a patient held at HIGH for six hours is one alert,
# not six, and interruptions are what alarm fatigue is about. s13 counts
# occupancy instead, which is why its published cuts and these differ.
#
# These are NOT ALERT_BUDGETS. Changing the unit from occupancy-hours to
# promotion events changes the achievable range with it: measured on the
# calibration fold, the promotion rate peaks near 1.0 per patient-day at ANY
# boundary and falls away on both sides, so budgets of 1/2/4 are unreachable --
# no cut produces four escalations a day. Reusing them made every cut collapse
# to the grid floor and put 97% of readings in HIGH or CRITICAL.
#
# Descending, because the cuts ascend in severity: MEDIUM is the loosest floor
# and CRITICAL the tightest. Read as "escalate a patient into this band at most
# this many times per ventilated patient-day".
BAND_PROMOTION_BUDGETS = (0.70, 0.45, 0.20)   # MEDIUM / HIGH / CRITICAL
BAND_GRID_POINTS = 400            # cut-search resolution, matching s13's grid

# A monitor whose top band is not rare is not a monitor. Asserted on the
# displayed band, which is what a clinician sees.
BAND_MIN_LOW_SHARE = 0.50

# Hysteresis sweep. Measured, not chosen: s16 evaluates every combination and
# picks by BAND_MAX_LOST_LEAD_MIN below. promote_min_readings is derived rather
# than swept -- 1 when no dwell is asked for (the no-hysteresis reference), else
# 2, so a single reading cannot satisfy a dwell purely by sitting next to a long
# gap in the charting.
BAND_PROMOTE_DWELL_MIN = (0.0, 30.0, 60.0)
BAND_DEMOTE_MARGIN_FRAC = (0.0, 0.25, 0.5)   # of the gap down to the band below
BAND_DEMOTE_DWELL_MIN = (0.0, 60.0, 120.0)

# Selection rule, stated as constants so the choice is reproducible rather than
# eyeballed: at a fixed promotion budget, take the configuration that DETECTS
# most, subject to a cap on lost warning time and on added flicker.
#
# Selecting on flips directly looks obvious and is wrong. Each configuration
# re-solves its cuts to hold the budget, and hysteresis suppresses
# re-promotions, so it meets the same budget at a LOWER cut -- measured, 0.1693
# against 0.2415 for the MEDIUM floor. A lower cut is crossed more often, so
# flips rise even though the machine is doing its job. The flicker claim is
# tested separately, at FIXED cuts, where it is not confounded.
BAND_MAX_LOST_LEAD_MIN = 30.0
BAND_MAX_FLIP_INCREASE = 0.25     # vs the no-hysteresis reference

# A band's observed rate must stay inside [CI_lo - tol, CI_hi + tol] measured on
# the calibration fold. A retrain that moves HIGH's real rate outside its
# envelope fails the build instead of quietly changing what the explanatory
# layer is told HIGH means.
BAND_RATE_TOLERANCE = 0.02

# Ratchet guard. Holding the top band longer than the raw score sits above its
# cut is not a defect -- it is exactly what demote-slow does, and measured here
# it inflates CRITICAL occupancy 1.9x. The failure mode worth catching is
# UNBOUNDED holding: a band that absorbs patients, so a long stay ends up
# permanently red.
#
# So the test is not the inflation ratio itself but whether that ratio GROWS
# with stay length. Comparing displayed against instant occupancy within each
# stay-length quartile controls for the fact that longer stays are sicker and
# would legitimately spend more time in the top band anyway.
BAND_MAX_RATCHET = 1.5      # ratio in the longest quartile vs the shortest

# ILLUSTRATIVE deployment profile, recorded in the artifact so that a dwell
# fitted on MIMIC's grid is never mistaken for one fitted on the bedside grid.
# A live ventilator feed samples far faster than MIMIC charts -- these numbers
# stand in for that gap; they are not a commitment to any particular rate. The
# caveat they carry holds for any deployment grid finer than the charting one:
# the dwell has to be re-expressed, and the documentation features degenerate.
BAND_DEPLOY_HZ = 1.0
BAND_DEPLOY_PROMOTE_SEC = 10.0

# GPU: RTX 5060 (Blackwell sm_120), 8 GB VRAM with ~6.5 GB usable.
# Keep max_bin low so the quantised matrix stays comfortably inside VRAM.
GPU_MAX_BIN = 64
USE_GPU = os.getenv("PM_USE_GPU", "1") == "1"
N_CPU_THREADS = int(os.getenv("PM_THREADS", "30"))

# --------------------------------------------------------------------------
# Per-stage cache fingerprints
#
# common.config_hash() hashes the extraction contract, which every stage
# inherits. Anything ONE stage depends on belongs here instead, passed as
# `extra=` to cached_stage, because a constant added to the base payload
# invalidates every manifest -- including s03's, which would re-scan 40 GB of
# chartevents because a label threshold moved.
#
# The rule: if a stage reads a constant and that constant is not in its
# fingerprint, changing it makes the stage SKIP and the pipeline silently
# serves stale data. That defect shipped once; these tuples are the fix.
# --------------------------------------------------------------------------
FP_STATIC = (S02_FINAL_COHORT, CHARLSON_HIERARCHY, STATIC_BODY_METRICS)
FP_INTERVENTIONS = (SPLIT_INOTROPE,)
FP_FORWARD = (HORIZONS_H, D_FIO2_RISE, D_PEEP_RISE, D_SPO2_BELOW,
              STRICT_BASELINE_AGE_MIN, D_FIO2_PERSIST, D_VASO_STRICT,
              D_VASO_OFF_MIN)
FP_MATRIX = (TARGET, TARGET_SOURCE, LEGACY_TARGET, ENABLE_PBW)
FP_CALIBRATE = FP_MATRIX + (ALERT_BUDGETS, ALERT_DEDUP_MINUTES,
                            EARLY_STOP_FOLD, CALIB_FOLD, TRAIN_FOLDS)
FP_COMPARE = (TARGET, PRIMARY_HORIZON_H, STRICT_BASELINE_AGE_MIN)
FP_BANDS = FP_CALIBRATE + (BAND_NAMES, BAND_PROMOTION_BUDGETS,
                           RISK_SCHEMA_VERSION, BAND_GRID_POINTS,
                           BAND_PROMOTE_DWELL_MIN, BAND_DEMOTE_MARGIN_FRAC,
                           BAND_DEMOTE_DWELL_MIN, BAND_MAX_LOST_LEAD_MIN,
                           BAND_MAX_FLIP_INCREASE, BAND_RATE_TOLERANCE,
                           BAND_MAX_RATCHET, BAND_MIN_LOW_SHARE,
                           LOCF_CUTOFF_MIN)
FP_RECORDS = FP_BANDS + (RECORD_SAMPLE_STAYS, CONTRIB_TOP_K, CONTRIB_RECON_TOL)
FP_EXPLAIN = FP_RECORDS + (EXPLAIN_SCHEMA_VERSION, EXPLAIN_CONTRIBUTOR_K,
                           EXPLAIN_MUTATION_SAMPLE, INSUFFICIENT_DATA_TEXT,
                           EXPLANATION_UNAVAILABLE_TEXT,
                           SUFFICIENCY_MAX_IMPUTED_SHARE,
                           SUFFICIENCY_MAX_DOC_SHARE, tuple(PARAM_DISPLAY.items()))
# ⚠️ INCOMPLETE ON PURPOSE. s19 appends a hash of explain.SYSTEM_PROMPT at the
# call site, because config must not import the pure modules (they take their
# policy as an argument precisely so serving can use them standalone).
#
# The prompt is THE tunable constant of that stage. Leave it out of the
# fingerprint and editing it makes s19 skip -- you would then compare a new
# prompt against output generated by the old one and conclude nothing. This
# repo has shipped that defect twice; the prompt is the shape it takes here.
FP_GENERATE = FP_EXPLAIN + (LLM_MODEL_ID, LLM_REVISION, LLM_QUANT, LLM_SAMPLE,
                            LLM_MAX_NEW_TOKENS, LLM_SEED, LLM_SCHEMA_VERSION)

# The corpus fingerprint deliberately does NOT include the source files -- those
# are passed to cached_stage as `sources=`, which fingerprints them by size and
# mtime. What belongs here is the extraction and chunking contract, because a
# changed hyphen rule or chunk size re-chunks the same PDFs into different text.
FP_CORPUS = (CORPUS_SCHEMA_VERSION, CORPUS_HYPHEN_DEFAULT_JOIN,
             CORPUS_MATH_ITALIC_MAX_RUN, CORPUS_CHUNK_MIN_WORDS,
             CORPUS_CHUNK_TARGET_WORDS, CORPUS_CHUNK_MAX_WORDS,
             CORPUS_KIND_PRIORITY,
             tuple(sorted((k, v["role"], v["cite"]) for k, v in CORPUS_DOCS.items())))
# ⚠️ INCOMPLETE ON PURPOSE, the same way FP_GENERATE is. s21 appends the corpus
# manifest hash at the call site: the evidence map is a function of the corpus
# CONTENT, and content is not recoverable from a constant. Without it, editing a
# PDF would leave s21 logging SKIP and serving a map built from the old text.
FP_EVIDENCE = FP_CORPUS + (EVIDENCE_SCHEMA_VERSION, EVIDENCE_CONTEXT_K,
                           EVIDENCE_ACTIONS_K, EVIDENCE_MIN_ANCHOR_HITS,
                           RAG_DENSE_ENABLED, RAG_QUERY_ENCODER,
                           RAG_DOC_ENCODER, RAG_CROSS_ENCODER,
                           RAG_RERANK_TOP_N, tuple(PARAM_DISPLAY), BAND_NAMES,
                           tuple(sorted(PARAM_QUERY_TERMS.items())),
                           tuple(sorted(BAND_QUERY_INTENT.items())),
                           tuple(sorted(KIND_QUERY_TERMS.items())))
