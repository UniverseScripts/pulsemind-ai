"""Stage 19 -- generate explanations with a local LLM, and measure them.

s18 built the payload contract, the grounding verifier and a template baseline.
This stage supplies the generator those were built for. explain.explain()
already accepts a `generator` callable, so this is a plug-in rather than a
rewrite.

WHAT IS ACTUALLY BEING MEASURED
-------------------------------
The generator is the easy half. The deliverable is a PAIRED comparison against
the template floor: the same records, both writers, the same checker. Without
the pairing, "the LLM had 3 warnings" is unanchored -- it only means something
beside what the template scored on those same readings.

Zero violations is the ship criterion. This stage does NOT assert it. A
violation here is a finding about the prompt or the model, and an assertion
that aborts the run would destroy the evidence needed to fix it. It is logged
loudly and reported; it is not swept up. What IS asserted is the things that
would be OUR bug: that the sample cleared the sufficiency floor, that every
record carries generator provenance, and that the checker actually ran.

EQUAL ALLOCATION ACROSS BANDS, AND IT SAYS SO
---------------------------------------------
CRITICAL is 3.6% of readings, so a proportional sample of 200 would contain
about seven of the cases where an explanation matters most. Allocation is equal
across the four bands instead. The consequence is stated in the report rather
than left for a reader to infer: per-band rates are population rates, the
overall figure is an unweighted mean across bands and is NOT one.

DUA
---
Generated text quotes per-patient MIMIC values. It goes to build/, never to
reports/, which is tracked. reports/llm_explanations.json carries counts and
codes only -- no text, no evidence strings, no stay_id.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from collections import Counter, defaultdict

import numpy as np

from .. import config as C
from ..core import explain as E
from ..core import generate as G
from ..core import grounding as GR
from ..common import cached_stage, log, stage
from .s18_explain import check, policy, records

REPORT_JSON = C.RPT_S19_LLM


def prompt_hash() -> str:
    """The system prompt is this stage's tunable constant, so it belongs in the
    fingerprint. config cannot import explain (the pure modules take their
    policy as an argument so serving can use them standalone), so the hash is
    appended here instead of being baked into FP_GENERATE."""
    return hashlib.sha256(E.SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]


def sample_records(pol: E.Policy, n: int) -> list[dict]:
    """Seeded, equal allocation across the four bands, one reading per stay.

    One per stay because consecutive readings from the same admission are
    near-duplicates: 200 of them from six patients would measure the model's
    behaviour on six situations, not two hundred.
    """
    per = max(1, n // len(C.BAND_NAMES))
    pool: dict[str, list] = defaultdict(list)
    seen_stay: dict[str, set] = defaultdict(set)
    for r in records():
        if not E.sufficiency(r, pol).ok:
            continue
        b = r["band"]["displayed"]
        if r["stay_id"] in seen_stay[b]:
            continue
        seen_stay[b].add(r["stay_id"])
        pool[b].append(r)

    rng = np.random.default_rng(C.LLM_SEED)
    out = []
    for b in C.BAND_NAMES:
        got = pool[b]
        if not got:
            log(f"  [yellow]no sufficient records in {b}[/yellow]")
            continue
        idx = rng.choice(len(got), size=min(per, len(got)), replace=False)
        out.extend(got[i] for i in sorted(idx))
        if len(got) < per:
            log(f"  [yellow]{b}: only {len(got)} available, wanted {per}[/yellow]")
    return out


def _tally(findings) -> tuple[Counter, Counter]:
    v, w = Counter(), Counter()
    for f in findings:
        (v if f.severity == GR.VIOLATION else w)[f.code] += 1
    return v, w


def evidence_hash(with_evidence: bool = True) -> str:
    """Identity of the retrieval configuration this run generated under.

    Appended at the CALL SITE, exactly like prompt_hash and for the same reason:
    an explanation must never be attachable to a different corpus. Both the
    corpus manifest and the on/off state are in it, so `--no-evidence` and
    `--with-evidence` cannot share a cache entry -- which is the entire point,
    since the pass exists to compare them.
    """
    if not with_evidence or not C.EVIDENCE_MAP_JSON.exists():
        return "no-evidence"
    m = json.loads(C.EVIDENCE_MAP_JSON.read_text(encoding="utf-8"))
    return f"ev:{m.get('corpus_manifest_hash', '?')}:{m.get('schema_version', '?')}"


def main(force: bool = False, sample: int | None = None,
         with_evidence: bool = True) -> None:
    with cached_stage("s19_generate",
                      sources=[C.RECORDS_JSONL],
                      output=C.LLM_EXPLANATIONS_JSONL, force=force,
                      extra=C.FP_GENERATE + (prompt_hash(),
                                             evidence_hash(with_evidence))) as ran:
        if not ran:
            return
        _run(sample, with_evidence)


def _run(sample: int | None = None, with_evidence: bool = True) -> None:
    report: dict = {}
    t0 = time.time()
    pol = policy(with_evidence=with_evidence)
    n_want = sample or C.LLM_SAMPLE
    log(f"retrieval: {evidence_hash(with_evidence)}"
        + (f"  ({len(pol.evidence)} keys)" if pol.evidence else "  (baseline, no RAG)"))

    with stage("Capability check -- toolchain present, and room to load"):
        pf = G.capability_check()
        log(f"  {pf['gpu']}  {pf['compute_capability']}  quant={pf['quantisation']}"
            + (f"  bnb {pf['bitsandbytes']}" if "bitsandbytes" in pf else ""))
        log(f"  VRAM {pf['vram_free_gb']:.2f} GB free of {pf['vram_total_gb']:.2f}")
        # Does NOT load a probe model. Doing that in this process leaves a CUDA
        # context and fragmented segments behind, and the 7B load then fails on
        # fragmentation rather than capacity -- which is exactly what happened.
        # The full probe is `--preflight`, a separate command and process.
        #
        # MEASURED, in MiB and from the driver -- the same unit and source the
        # gate compares against. Peak 6239 MiB across three runs; evidence in
        # `tools/vram_probe.py` and `reports/tool_vram_probe.json`.
        assert pf["vram_free_mib"] >= C.LLM_MIN_FREE_VRAM_MIB, (
            f"only {pf['vram_free_mib']} MiB VRAM free of "
            f"{pf['vram_total_mib']}; the 7B peaks at ~6239 MiB and this stage "
            f"needs {C.LLM_MIN_FREE_VRAM_MIB}. Close whatever is holding the GPU, or "
            f"set PM_LLM_QUANT=none with a 3B model.")

    with stage(f"Load {C.LLM_MODEL_ID}"):
        gen = G.load_generator()
        import torch
        log(f"  {gen.provenance['quantisation']} / {gen.provenance['dtype']} "
            f"on {gen.provenance['device']}  "
            f"VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        log(f"  prompt hash {prompt_hash()}  seed {C.LLM_SEED}  greedy")

    with stage(f"Sample {n_want} readings, equal across bands"):
        recs = sample_records(pol, n_want)
        by_band = Counter(r["band"]["displayed"] for r in recs)
        log("  " + "  ".join(f"{b} {by_band[b]}" for b in C.BAND_NAMES)
            + f"   total {len(recs)}")
        # Assertion 1: the gate is upstream, and the sample must respect it.
        assert all(E.sufficiency(r, pol).ok for r in recs), (
            "a record below the sufficiency floor reached the generator")

    with stage("Generate, verify, and pair against the template floor"):
        rows = []
        ev_hash = evidence_hash(with_evidence)
        ctx_n = act_n = 0
        v_llm, w_llm, v_base, w_base = Counter(), Counter(), Counter(), Counter()
        per_band = defaultdict(lambda: {"n": 0, "llm_v": 0, "llm_w": 0,
                                        "base_v": 0, "base_w": 0})
        llm_len, base_len = [], []
        for i, r in enumerate(recs, 1):
            blk = E.explain(r, pol, generator=gen,
                            generator_name=C.LLM_MODEL_ID)
            text = blk["text"]
            f_llm = check(r, text, pol)
            base = E.baseline(r, pol)
            f_base = check(r, base, pol)

            a, b = _tally(f_llm); v_llm += a; w_llm += b
            c, d = _tally(f_base); v_base += c; w_base += d
            band = r["band"]["displayed"]
            pb = per_band[band]
            pb["n"] += 1
            pb["llm_v"] += sum(a.values()); pb["llm_w"] += sum(b.values())
            pb["base_v"] += sum(c.values()); pb["base_w"] += sum(d.values())
            llm_len.append(len(text)); base_len.append(len(base))
            ctx_n += len(blk["guideline_context"])
            act_n += len(blk["suggested_actions"])

            rows.append({
                "schema_version": C.LLM_SCHEMA_VERSION,
                "record_schema_version": r["schema_version"],
                "provenance": r["provenance"],
                "generator_provenance": gen.provenance,
                "stay_id": r["stay_id"], "charttime": r["charttime"],
                "band": band, "status": blk["status"],
                "generator": blk["generator"],
                "text": text,
                "findings": [f.to_json() for f in f_llm],
                "baseline_text": base,
                "baseline_findings": [f.to_json() for f in f_base],
                # Assembled by code, not written by the model. Recorded on the
                # line so verify.py can re-check byte identity against the
                # corpus without re-running retrieval.
                "guideline_context": blk["guideline_context"],
                "suggested_actions": blk["suggested_actions"],
                "evidence_hash": ev_hash,
            })
            if i % 25 == 0 or i == len(recs):
                rate = np.mean(gen.latencies) if gen.latencies else 0.0
                log(f"  {i}/{len(recs)}   {rate:.1f}s/record   "
                    f"violations so far: llm {sum(v_llm.values())} / "
                    f"base {sum(v_base.values())}")

        with gzip.open(C.LLM_EXPLANATIONS_JSONL, "wt", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        # Assertion 2: provenance must travel with every generated line, or an
        # explanation could be attached to output from another model.
        assert all(x["generator_provenance"]["model"] == C.LLM_MODEL_ID
                   for x in rows), "a row lost its generator provenance"

    with stage("Aggregates -> reports/llm_explanations.json (no per-patient data)"):
        lat = np.array(gen.latencies) if gen.latencies else np.array([0.0])
        n = len(rows)
        tv, tb = sum(v_llm.values()), sum(v_base.values())
        log(f"  {'':14}{'violations':>12}{'warnings':>11}{'median chars':>14}")
        log(f"  {'template':14}{tb:>12,}{sum(w_base.values()):>11,}"
            f"{np.median(base_len):>14,.0f}")
        log(f"  {'LLM':14}{tv:>12,}{sum(w_llm.values()):>11,}"
            f"{np.median(llm_len):>14,.0f}")
        if tv:
            log(f"  [red]{tv} violation(s) from the LLM[/red] -- "
                f"{dict(v_llm)}")
            log("  [red]a finding about the prompt or the model. Iterate the "
                "prompt; do not relax the check.[/red]")
        else:
            log("  [green]zero violations -- the LLM contradicted no record "
                "it was given[/green]")
        for b in C.BAND_NAMES:
            p = per_band[b]
            if p["n"]:
                log(f"  {b:<9} n={p['n']:<4} llm {p['llm_v']}v/{p['llm_w']}w   "
                    f"template {p['base_v']}v/{p['base_w']}w")

        report.update(
            schema_version=C.LLM_SCHEMA_VERSION,
            generated=n,
            sample_allocation=dict(by_band),
            sampling=("equal allocation across bands, one reading per stay, "
                      "seeded. Per-band rates are population rates; the overall "
                      "figure is an unweighted mean across bands and is not."),
            generator_provenance=gen.provenance,
            preflight=pf,
            prompt_sha256_16=prompt_hash(),
            retrieval=dict(
                evidence_hash=ev_hash,
                enabled=bool(pol.evidence),
                keys_available=len(pol.evidence or {}),
                context_passages_attached=ctx_n,
                action_statements_attached=act_n,
                note=("guideline_context and suggested_actions are ASSEMBLED "
                      "from the frozen evidence map, never generated. The model "
                      "sees guideline_context only, and is told not to restate "
                      "it; suggested_actions never enters the prompt.")),
            llm=dict(violations=dict(v_llm), warnings=dict(w_llm),
                     total_violations=tv, total_warnings=sum(w_llm.values()),
                     median_chars=float(np.median(llm_len))),
            template=dict(violations=dict(v_base), warnings=dict(w_base),
                          total_violations=tb,
                          total_warnings=sum(w_base.values()),
                          median_chars=float(np.median(base_len))),
            per_band={b: dict(per_band[b]) for b in C.BAND_NAMES if per_band[b]["n"]},
            latency=dict(median_s=float(np.median(lat)),
                         p90_s=float(np.percentile(lat, 90)),
                         total_s=float(lat.sum()),
                         projected_full_set_hours=float(
                             np.median(lat) * 58765 / 3600)),
            ship_criterion="zero violations; reported, not asserted, so a "
                           "failing run still yields the evidence to fix it",
            seconds=round(time.time() - t0, 1))
        REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"  {REPORT_JSON.name} written   "
            f"({np.median(lat):.1f}s/record, full set would be "
            f"{np.median(lat) * 58765 / 3600:.0f} h)")


def recheck() -> None:
    """Re-run the grounding checker over saved generations, without the model.

    The checker changes more often than the generator does, and generation is
    greedy from a fixed seed -- re-running it would reproduce the same text at
    the cost of an hour of GPU. Rescoring saved output is the honest way to
    measure a checker change, and it keeps the comparison paired: the template
    text is rescored in the same pass, by the same code.
    """
    rows = []
    with gzip.open(C.LLM_EXPLANATIONS_JSONL, "rt", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    recs = {(r["stay_id"], r["charttime"]): r for r in records()}
    # Rescore under the retrieval configuration the text was GENERATED under,
    # not under whatever is on disk now. Rescoring RAG output without evidence
    # would report every quoted threshold as a fabricated number and read as a
    # catastrophic regression that never happened.
    was_rag = any(row.get("guideline_context") for row in rows)
    pol = policy(with_evidence=was_rag)
    log(f"rechecking {len(rows)} rows with evidence="
        f"{'on' if was_rag else 'off'} (as generated)")

    v_llm, w_llm, v_base, w_base = Counter(), Counter(), Counter(), Counter()
    per_band = defaultdict(lambda: {"n": 0, "llm_v": 0, "llm_w": 0,
                                    "base_v": 0, "base_w": 0})
    for row in rows:
        rec = recs[(row["stay_id"], row["charttime"])]
        f_llm = check(rec, row["text"], pol)
        f_base = check(rec, row["baseline_text"], pol)
        row["findings"] = [f.to_json() for f in f_llm]
        row["baseline_findings"] = [f.to_json() for f in f_base]
        a, b = _tally(f_llm); v_llm += a; w_llm += b
        c, d = _tally(f_base); v_base += c; w_base += d
        pb = per_band[row["band"]]
        pb["n"] += 1
        pb["llm_v"] += sum(a.values()); pb["llm_w"] += sum(b.values())
        pb["base_v"] += sum(c.values()); pb["base_w"] += sum(d.values())

    with gzip.open(C.LLM_EXPLANATIONS_JSONL, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")

    rep = json.loads(REPORT_JSON.read_text()) if REPORT_JSON.is_file() else {}
    tv, tb = sum(v_llm.values()), sum(v_base.values())
    rep["llm"] = {**rep.get("llm", {}), "violations": dict(v_llm),
                  "warnings": dict(w_llm), "total_violations": tv,
                  "total_warnings": sum(w_llm.values())}
    rep["template"] = {**rep.get("template", {}), "violations": dict(v_base),
                       "warnings": dict(w_base), "total_violations": tb,
                       "total_warnings": sum(w_base.values())}
    rep["per_band"] = {b: dict(per_band[b]) for b in C.BAND_NAMES if per_band[b]["n"]}
    rep["rechecked"] = ("findings rescored from saved text; generation not "
                        "re-run (greedy + fixed seed, so it would be identical)")
    rep["prompt_sha256_16"] = prompt_hash()
    REPORT_JSON.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    log(f"  rescored {len(rows)} generations without the model")
    log(f"  {'':10}{'violations':>12}{'warnings':>11}")
    log(f"  {'template':10}{tb:>12}{sum(w_base.values()):>11}")
    log(f"  {'LLM':10}{tv:>12}{sum(w_llm.values()):>11}")
    if tv:
        log(f"  [red]{tv} violation(s): {dict(v_llm)}[/red]")
    else:
        log("  [green]zero violations -- ship criterion met[/green]")


def show(k: int) -> None:
    """Print generated text beside the template for the same reading. Whether
    the contract says enough to explain a reading is a judgement, not an
    assertion; this is where it gets made."""
    shown = 0
    with gzip.open(C.LLM_EXPLANATIONS_JSONL, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            print("=" * 78)
            print(f"band {r['band']}   findings: llm {len(r['findings'])} / "
                  f"template {len(r['baseline_findings'])}")
            print("\n--- LLM ---\n" + r["text"])
            print("\n--- template ---\n" + r["baseline_text"])
            for blk, title in (("guideline_context", "guideline context"),
                               ("suggested_actions", "suggested actions")):
                for e in r.get(blk) or ():
                    grade = (f" [{e['strength']}, {e['certainty']} certainty]"
                             if e.get("strength") else "")
                    print(f"\n--- {title} ({e['parameter']}){grade} ---")
                    print(f"  \"{e['quote']}\"\n  -- {e['citation']}")
            for f in r["findings"]:
                print(f"\n  [{f['severity']}] {f['code']}: {f['message']}")
            shown += 1
            if shown >= k:
                return


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _ap.add_argument("--force", action="store_true",
                     help="re-generate even when the manifest is current")
    _ap.add_argument("--sample", type=int, metavar="N",
                     help="override LLM_SAMPLE for this run")
    _ap.add_argument("--preflight", action="store_true",
                     help="probe the toolchain with a ~350 MB model and exit")
    _ap.add_argument("--show", type=int, metavar="K",
                     help="print K generated/template pairs from the last run")
    _ap.add_argument("--recheck", action="store_true",
                     help="rescore saved generations with the current checker; "
                          "no model, no GPU, no regeneration")
    _ap.add_argument("--no-evidence", action="store_true",
                     help="generate WITHOUT retrieval -- the paired baseline "
                          "this pass is measured against")
    _a = _ap.parse_args()
    if _a.preflight:
        print(json.dumps(G.preflight(), indent=2))
    elif _a.recheck:
        recheck()
    elif _a.show:
        show(_a.show)
    else:
        main(force=_a.force or bool(_a.sample), sample=_a.sample,
             with_evidence=not _a.no_evidence)
