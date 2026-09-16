"""Measure what a 7B load actually PEAKS at, instead of bracketing it.

WHY THIS EXISTS
---------------
`explanation.MIN_FREE_VRAM_MIB` was set from two data points -- a segfault at
6561 MiB free and a success at 6721 -- and the constant's own comment says the
value chosen inside that band is "slightly optimistic". Nobody had watched the
card DURING a load, so the peak had never been measured, only inferred from
whether the process survived.

WHAT IT MEASURES, AND WHY IN THAT UNIT
--------------------------------------
`peak = free_before - min(free) observed across the run`, sampled from
`nvidia-smi` at 5 Hz. Deliberately the SAME source and the SAME unit the gate
compares against, because that is the only way the answer is directly usable: a
peak in torch's units would need converting through a bookkeeping layer that
`generate.vram_status()` documents as wrong by gigabytes on this WDDM setup.

`torch.cuda.max_memory_allocated/reserved()` are recorded beside it as a
DECOMPOSITION, never as the answer. They see only torch's own allocations, so
they miss the CUDA context entirely.

⚠️ THE SAMPLER RUNS IN THE PARENT
---------------------------------
A load that does not fit does not raise, it SEGFAULTS -- no exception, no
unwinding, nothing in the child runs afterwards. A sampler started by the child
is therefore orphaned by exactly the outcome most worth recording, and keeps
writing to a file nobody will close. Measured: three leaked `nvidia-smi`
processes across three failed runs.

So the parent owns the sampler and the trace file. The child reports its own
phase boundaries as epoch timestamps when it survives; when it does not, the
whole span is the load, which is the only window a crash has anyway. **A crash
is a measurement here**, and this is what makes it readable.

⚠️ ONE LOAD PER PROCESS
-----------------------
`load_generator()` is never unloaded -- `explanation.py` caches it in a module
global and `model_runtime` documents that releasing it dies 0xC0000409. N loads
in one process would mean N models on an 8 GB card, so every repeat is a fresh
subprocess and `--once` is the inner half of that handoff.

WHAT IT CHANGES
---------------
Nothing. It writes one report and loads the model the way the service would.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime

from .. import config as C
from ..common import log

REPORT_JSON = C.RPT_TOOL_VRAM_PROBE

#: Sampling period for the driver poll. 5 Hz against a load that takes 28-45 s
#: is 150-225 samples -- dense enough to catch a transient lasting a fraction of
#: a second, and cheap enough that the sampler is not itself load. One
#: long-running `nvidia-smi`, NOT a spawn per sample: a spawn costs ~50-100 ms
#: here, which would sit inside the window it is trying to measure.
SAMPLE_MS = 200

#: nvidia-smi's own timestamp format, so samples can be bucketed into the load
#: window and the generation window rather than reported as one blur.
TS_FMT = "%Y/%m/%d %H:%M:%S.%f"


def _start_sampler(path):
    """`nvidia-smi` streaming timestamped free-VRAM to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,memory.free",
         "--format=csv,noheader,nounits", "-lms", str(SAMPLE_MS)],
        stdout=handle, stderr=subprocess.DEVNULL, text=True)
    proc._pm_handle = handle
    return proc


def _stop_sampler(proc, path) -> list[tuple[float, int]]:
    """Stop it and read back `(epoch_seconds, free_mib)`. Idempotent."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    if not proc._pm_handle.closed:
        proc._pm_handle.close()

    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        ts, _, free = line.partition(",")
        if not free.strip():
            continue
        try:
            out.append((datetime.strptime(ts.strip(), TS_FMT).timestamp(), int(free)))
        except ValueError:
            continue  # a partially written final line is not a failure
    return out


def _window(samples, t0: float | None, t1: float | None) -> dict:
    """Summarise the samples inside a phase. Reports `n` so a window that caught
    nothing is visible as such rather than as a missing key."""
    if t0 is None or t1 is None:
        return {"n": 0, "min_free_mib": None, "max_free_mib": None}
    got = [mib for when, mib in samples if t0 <= when <= t1]
    if not got:
        return {"n": 0, "min_free_mib": None, "max_free_mib": None}
    return {"n": len(got), "min_free_mib": min(got), "max_free_mib": max(got)}


# ---------------------------------------------------------------- the child


def once(sample: int, record_index: int, with_evidence: bool) -> dict:
    """One load and one generation, in a process that has never touched CUDA.

    Imports are function-local on purpose: `free_before` has to be read before
    torch initialises a context, or the context is inside the figure the peak is
    being measured against.

    Reports epoch timestamps rather than durations so the parent can bucket its
    own samples into these phases.
    """
    from ..core import generate as G

    before = G.vram_status()

    from ..core import explain as E
    from ..stages.s18_explain import policy
    from ..stages.s19_generate import sample_records

    pol = policy(with_evidence=with_evidence)
    recs = sample_records(pol, sample)
    if not recs:
        raise RuntimeError("no records cleared the sufficiency floor; nothing to generate")
    rec = recs[record_index % len(recs)]

    t_load0 = time.time()
    gen = G.load_generator()
    t_load1 = time.time()
    after_load = G.vram_status()

    import torch
    torch_peak = {"max_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2),
                  "max_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2)}

    emb = gen.model.get_input_embeddings().weight
    placement = {"embed_device": str(emb.device), "embed_dtype": str(emb.dtype),
                 "embed_mib": round(emb.numel() * emb.element_size() / 1024**2)}

    t_gen0 = time.time()
    blk = E.explain(rec, pol, generator=gen, generator_name=C.LLM_MODEL_ID)
    t_gen1 = time.time()
    after_gen = G.vram_status()

    # PM-LOG-003: the prose never leaves this process. The hash is what makes
    # "byte-identical after the change" checkable without storing patient text.
    text = blk["text"]
    return {
        "free_before_mib": before["free_mib"],
        "free_after_load_mib": after_load["free_mib"],
        "free_after_generate_mib": after_gen["free_mib"],
        "total_mib": before["total_mib"],
        "vram_source": before["source"],
        "resident_mib": before["free_mib"] - after_load["free_mib"],
        "t_load": [t_load0, t_load1],
        "t_generate": [t_gen0, t_gen1],
        "load_seconds": round(t_load1 - t_load0, 1),
        "generate_seconds": round(t_gen1 - t_gen0, 1),
        "torch": torch_peak,
        "placement": placement,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text_chars": len(text),
        "provenance": gen.provenance,
    }


# ---------------------------------------------------------------- the parent


def _one_run(i: int, sample: int, record_index: int, with_evidence: bool) -> dict:
    """Sample the card in this process while a child does the load."""
    trace = C.BUILD / "scratch" / f"vram_probe_{int(time.time() * 1000)}.csv"
    sampler = _start_sampler(trace)
    time.sleep(1.0)  # a baseline before the child touches anything
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pipeline.tools.vram_probe", "--once",
             "--sample", str(sample), "--record-index", str(record_index)]
            + ([] if with_evidence else ["--no-evidence"]),
            capture_output=True, text=True)
        samples = _stop_sampler(sampler, trace)
    finally:
        _stop_sampler(sampler, trace)

    span_min = min((m for _, m in samples), default=None)
    span_max = max((m for _, m in samples), default=None)

    if proc.returncode != 0 or not proc.stdout.strip():
        # A CRASH IS A MEASUREMENT. It is the only evidence that narrows the
        # requirement from below, so it is recorded with its trace rather than
        # raised. `Segmentation fault` arrives on the shell's stderr, not the
        # child's, so returncode is the only reliable signal.
        trace.unlink(missing_ok=True)
        return {"run": i, "ok": False,
                "error": f"exit {proc.returncode}",
                "start_free_mib": span_max, "min_free_mib": span_min,
                "consumed_before_death_mib": (span_max - span_min)
                if span_max is not None else None,
                "n_samples": len(samples),
                "stderr_tail": proc.stderr.strip()[-600:]}

    row = json.loads(proc.stdout.strip().splitlines()[-1])
    row["run"], row["ok"] = i, True
    row["load_window"] = _window(samples, *row.pop("t_load"))
    row["generate_window"] = _window(samples, *row.pop("t_generate"))
    row["n_samples"] = len(samples)
    lo = row["load_window"]["min_free_mib"]
    row["peak_mib"] = (row["free_before_mib"] - lo) if lo is not None else None
    row["peak_over_resident_mib"] = ((row["peak_mib"] - row["resident_mib"])
                                     if row["peak_mib"] is not None else None)
    trace.unlink(missing_ok=True)
    return row


def main(runs: int = 5, sample: int = 4, record_index: int = 0,
         with_evidence: bool = True, tag: str = "") -> None:
    rows = []
    for i in range(1, runs + 1):
        log(f"run {i}/{runs} -- fresh process, one load")
        row = _one_run(i, sample, record_index, with_evidence)
        rows.append(row)
        if row["ok"]:
            log(f"  peak {row['peak_mib']} MiB   resident {row['resident_mib']} MiB   "
                f"load {row['load_seconds']}s   gen {row['generate_seconds']}s   "
                f"embed on {row['placement']['embed_device']}")
        else:
            log(f"  [red]{row['error']}[/red] -- consumed "
                f"{row['consumed_before_death_mib']} MiB before dying, "
                f"floor {row['min_free_mib']} MiB")

    ok = [r for r in rows if r["ok"] and r.get("peak_mib") is not None]
    peaks = [r["peak_mib"] for r in ok]
    residents = [r["resident_mib"] for r in ok]
    hashes = sorted({r["text_sha256"] for r in ok})
    spread = (max(peaks) - min(peaks)) if len(peaks) > 1 else 0

    report = {
        "tag": tag or "untagged",
        "embed_device": C.LLM_EMBED_DEVICE,
        "model": C.LLM_MODEL_ID, "quantisation": C.LLM_QUANT,
        "n_ok": len(ok), "n_failed": len(rows) - len(ok),
        "peak_mib": {"max": max(peaks) if peaks else None,
                     "min": min(peaks) if peaks else None,
                     "median": statistics.median(peaks) if peaks else None,
                     "spread": spread},
        "resident_mib": {"max": max(residents) if residents else None,
                         "min": min(residents) if residents else None,
                         "spread": (max(residents) - min(residents))
                         if len(residents) > 1 else 0},
        # One hash across every run is the greedy-decoding claim, measured. More
        # than one means the generator is not deterministic and every
        # before/after text comparison is void.
        "text_sha256": hashes,
        "deterministic": len(hashes) == 1 if hashes else None,
        # Read, never hardcoded -- a probe that reports a stale gate beside a
        # fresh peak is worse than one that reports no gate at all.
        "gate_now_mib": C.LLM_MIN_FREE_VRAM_MIB,
        "runs": rows,
    }
    if peaks:
        # The gate has to cover the peak AND the drift between runs: background
        # software moves this card by hundreds of MiB during a 40 s load, and
        # that drift is what made 6561 both a crash and a success.
        report["suggested_gate_mib"] = max(peaks) + max(spread, 128)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, default=str))
    log(f"wrote {REPORT_JSON}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true",
                    help="inner half of the subprocess handoff: one load, JSON to stdout")
    ap.add_argument("-n", "--runs", type=int, default=5)
    ap.add_argument("--sample", type=int, default=4,
                    help="passed to sample_records; seeded, so a fixed value fixes the record")
    ap.add_argument("--record-index", type=int, default=0)
    ap.add_argument("--no-evidence", action="store_true")
    ap.add_argument("--tag", default="", help="names the run in the report")
    a = ap.parse_args()
    if a.once:
        print(json.dumps(once(a.sample, a.record_index, not a.no_evidence)))
    else:
        main(a.runs, a.sample, a.record_index, not a.no_evidence, a.tag)
