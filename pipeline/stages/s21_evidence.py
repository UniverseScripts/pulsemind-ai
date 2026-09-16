"""Stage 21 -- the frozen, reviewable evidence map.

WHY THERE IS NO VECTOR DATABASE HERE
------------------------------------
The decisive property of this corpus is not that it is small. It is that
**the query space is closed and enumerable.**

A Pulsemind retrieval query is never free text. It is composed entirely of the
11 frozen parameters, the 4 band names, and kind in {physiology, documentation}.
Nobody types a question. So the ENTIRE set of possible retrieval results can be
materialised once, at build time, reviewed by a human, and frozen -- and
inference becomes a dict lookup with zero VRAM, zero added latency and complete
determinism, which is the same discipline s19 already follows for generation.

That is not skipping retrieval: the literature-standard pipeline runs here, at
build time, to PROPOSE the map, and the review then ACCEPTS it. A small corpus
turns a tuning problem into a review problem -- nobody can audit a cosine
similarity, and anybody can read 57 passages and say whether they belong.

TWO CHANNELS
------------
  context  keyed on parameter (and on kind)        -> definitions, mostly NHSN
  actions  keyed on (band, parameter)              -> graded statements, protocol steps

Band enters ONLY in the actions channel, and that is deliberate: what is worth
suggesting at LOW (weaning readiness) is the wrong thought at CRITICAL.

THE GATE IS TERM OVERLAP, NOT A SCORE
-------------------------------------
A passage is admissible for `peep` only if it mentions PEEP or a listed synonym
(C.PARAM_QUERY_TERMS). On a closed query set that beats a similarity threshold
outright -- it is decidable, reviewable, and it lets "no adequate passage" be an
honest outcome instead of whatever ranked first among candidates that were all
wrong. Ranking then decides between admissible passages; it never admits one.

RANKING
-------
Lexical BM25 (sqlite FTS5, compiled into the stdlib -- no dependency, no
download) is the primary channel. MedRAG measures the best biomedical dense
retriever beating BM25 by 0.01 points on MedCorp (70.06 vs 70.05), so dense is a
cheap second opinion rather than the backbone; if MedCPT will not load, this
stage says so in the report and ships lexical-only.

Type priority -- recommendation > table > prose -- is then applied as a
STRUCTURAL filter on chunk kind rather than as a rerank (ClinicBot,
arXiv 2605.00846): the most citable content has to dominate, and sorting by
similarity does not guarantee that.

NO PATIENT DATA. Guideline text only, so unlike build/, everything this stage
writes can be read, quoted and reviewed freely.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from collections import Counter

from .. import config as C
from ..common import cached_stage, log
from .s20_corpus import corpus_manifest_hash, load_chunks

RE_TOKEN = re.compile(r"[A-Za-z0-9]+")


# --------------------------------------------------------------------------
# The key set -- enumerated, not discovered
# --------------------------------------------------------------------------
def enumerate_keys() -> list[dict]:
    """Every retrieval key the serving layer can ever ask for.

    Derived from config rather than from features.json on purpose: the five
    derived forms of a parameter (`peep_final`, `peep_locf`, `peep_observed`,
    ...) all retrieve the SAME guideline text -- the guideline has nothing to
    say about last-observation-carried-forward -- so the key is the parameter,
    and explain.split_feature() already maps a feature name onto it.
    """
    keys: list[dict] = []
    for p, (label, unit, _dec) in C.PARAM_DISPLAY.items():
        keys.append({"key": f"context:{p}", "channel": "context",
                     "parameter": p, "band": None, "kind": None,
                     "label": label,
                     "terms": list(C.PARAM_QUERY_TERMS.get(p, (label,))),
                     "query": " ".join((label, unit) + C.PARAM_QUERY_TERMS.get(p, ()))})
    for k, terms in C.KIND_QUERY_TERMS.items():
        keys.append({"key": f"context:kind:{k}", "channel": "context",
                     "parameter": None, "band": None, "kind": k,
                     "label": k, "terms": list(terms),
                     "query": " ".join(terms)})
    for band in C.BAND_NAMES:
        intent = C.BAND_QUERY_INTENT[band]
        for p, (label, _unit, _dec) in C.PARAM_DISPLAY.items():
            terms = C.PARAM_QUERY_TERMS.get(p, (label,))
            keys.append({"key": f"action:{band}:{p}", "channel": "actions",
                         "parameter": p, "band": band, "kind": None,
                         "label": label, "terms": list(terms),
                         "query": " ".join(terms + intent)})
    return keys


# --------------------------------------------------------------------------
# Lexical channel
# --------------------------------------------------------------------------
def build_index(chunks: list[dict]) -> sqlite3.Connection:
    """An in-memory FTS5 index over title + text.

    The title is the spliced heading path (MedRAG), indexed alongside the body
    so a chunk that says "the daily minimum" is still findable under the section
    that says what of.
    """
    con = sqlite3.connect(":memory:")
    try:
        con.execute("CREATE VIRTUAL TABLE c USING fts5(title, body, tokenize='porter unicode61')")
    except sqlite3.OperationalError as exc:      # pragma: no cover -- build-dependent
        raise SystemExit(f"s21: sqlite has no FTS5 support: {exc}")
    con.executemany("INSERT INTO c(rowid, title, body) VALUES (?, ?, ?)",
                    [(i, ch["title"], ch["text"]) for i, ch in enumerate(chunks)])
    return con


def _fts_query(terms: list[str]) -> str:
    """FTS5 MATCH expression: every anchor term as a quoted OR'd phrase.

    Quoted because "positive end-expiratory pressure" has to match as a phrase,
    and because a bare hyphen is a NOT operator in FTS5 syntax -- unquoted,
    `auto-PEEP` silently becomes "auto AND NOT peep".
    """
    parts = []
    for t in terms:
        toks = RE_TOKEN.findall(t)
        if toks:
            parts.append('"' + " ".join(toks) + '"')
    return " OR ".join(dict.fromkeys(parts))


def lexical(con: sqlite3.Connection, key: dict, limit: int) -> dict[int, float]:
    expr = _fts_query(key["terms"] + key["query"].split())
    if not expr:
        return {}
    try:
        rows = con.execute(
            "SELECT rowid, bm25(c, 2.0, 1.0) FROM c WHERE c MATCH ? "
            "ORDER BY bm25(c, 2.0, 1.0) LIMIT ?", (expr, limit)).fetchall()
    except sqlite3.OperationalError:
        return {}
    # bm25() is negative and lower is better; flip it so bigger is better.
    return {int(r): -float(s) for r, s in rows}


# --------------------------------------------------------------------------
# Dense channel -- optional by design
# --------------------------------------------------------------------------
class Dense:
    """MedCPT query/article encoders plus its cross-encoder, on CPU.

    Optional on purpose. Everything here is a second opinion worth 0.01 points
    on the only head-to-head measurement available, so a missing download is a
    line in the report, never a failed build.
    """

    def __init__(self) -> None:
        self.ok = False
        self.rerank_ok = False
        self.note = "disabled by config"
        if not C.RAG_DENSE_ENABLED:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            self.torch = torch
            self.qt = AutoTokenizer.from_pretrained(C.RAG_QUERY_ENCODER)
            self.qm = AutoModel.from_pretrained(C.RAG_QUERY_ENCODER).eval()
            self.dt = AutoTokenizer.from_pretrained(C.RAG_DOC_ENCODER)
            self.dm = AutoModel.from_pretrained(C.RAG_DOC_ENCODER).eval()
            self.ok = True
            self.note = f"{C.RAG_QUERY_ENCODER} + {C.RAG_DOC_ENCODER} on {C.RAG_DENSE_DEVICE}"
        except Exception as exc:                 # noqa: BLE001 -- any failure is the same failure
            self.note = f"unavailable: {type(exc).__name__}: {str(exc)[:160]}"
            return
        # The reranker is a SEPARATE try. MedCPT measures average nDCG@10 rising
        # 0.443 -> 0.510 with it across five BEIR biomedical datasets, and here
        # it runs at build time only, so its inference cost is exactly zero --
        # but it is another 419 MB, and losing it must degrade the map rather
        # than fail the stage.
        try:
            from transformers import (AutoModelForSequenceClassification,
                                      AutoTokenizer)
            self.rt = AutoTokenizer.from_pretrained(C.RAG_CROSS_ENCODER)
            self.rm = AutoModelForSequenceClassification.from_pretrained(
                C.RAG_CROSS_ENCODER).eval()
            self.rerank_ok = True
            self.note += f" + {C.RAG_CROSS_ENCODER}"
        except Exception as exc:                 # noqa: BLE001
            self.note += f" (no reranker: {type(exc).__name__})"

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        """Cross-encoder relevance for one query against its candidates."""
        if not self.rerank_ok or not docs:
            return []
        out: list[float] = []
        with self.torch.no_grad():
            for i in range(0, len(docs), 8):
                batch = [[query, d] for d in docs[i:i + 8]]
                enc = self.rt(batch, truncation=True, padding=True,
                              max_length=512, return_tensors="pt")
                out += self.rm(**enc).logits.squeeze(-1).reshape(-1).tolist()
        return out

    def _embed(self, tok, mdl, texts: list[str], maxlen: int):
        import numpy as np
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), 16):
                enc = tok(texts[i:i + 16], truncation=True, padding=True,
                          max_length=maxlen, return_tensors="pt")
                # MedCPT reads the [CLS] representation, not a mean pool.
                v = mdl(**enc).last_hidden_state[:, 0, :]
                v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-9)
                out.append(v.cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 1), dtype="float32")

    def encode_docs(self, chunks: list[dict]):
        return self._embed(self.dt, self.dm,
                           [f"{c['title']}. {c['text']}" for c in chunks], 512)

    def encode_queries(self, queries: list[str]):
        return self._embed(self.qt, self.qm, queries, 64)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------
def anchor_hits(chunk: dict, key: dict) -> int:
    """How many times this passage mentions the thing being asked about.

    Both the admission gate and the weakness flag are built on this one count,
    because it is the only quality signal here a reviewer can independently
    check. A retrieval score cannot be audited; "the passage says PEEP twice"
    can.
    """
    hay = f"{chunk['title']} {chunk['text']}".lower()
    n = 0
    for t in key["terms"]:
        n += len(re.findall(r"(?<![a-z0-9])" + re.escape(t.lower()) + r"(?![a-z0-9])", hay))
    return n


def channel_pool(chunks: list[dict], key: dict) -> list[int]:
    """Which chunks this channel is even allowed to draw from.

    The actions channel may ONLY return chunks tagged `recommendation`. That is
    the structural half of the two-problem answer: an action Pulsemind emits is
    a graded statement somebody published, or it is nothing.
    """
    if key["channel"] == "actions":
        return [i for i, c in enumerate(chunks) if c["kind"] == "recommendation"]
    return list(range(len(chunks)))


RRF_K = 60      # the standard constant; damps the tail without tuning


def select(chunks: list[dict], key: dict, lex: dict[int, float],
           dense_scores: dict[int, float], want: int,
           cross: dict[int, float] | None = None) -> list[dict]:
    """Rank admissible passages by RECIPROCAL RANK FUSION.

    Fusing on rank rather than on score is not a stylistic choice. Averaging a
    min-max-normalised BM25 with a raw MedCPT cosine mixes two incomparable
    scales: BERT [CLS] cosines sit around 0.7-0.9 almost regardless of
    relevance, so switching the dense channel on lifted every fused score above
    any fixed threshold and took the "weak match" count from 11 to 0 without a
    single passage having changed. That is a units bug wearing the costume of an
    improvement. Ranks have no units.
    """
    pool = [i for i in channel_pool(chunks, key) if anchor_hits(chunks[i], key)]
    if not pool:
        return []
    ranks: dict[int, dict[str, int]] = {}
    for name, table in (("lexical", lex), ("dense", dense_scores),
                        ("cross", cross or {})):
        ordered = sorted((i for i in table if i in pool),
                         key=lambda i: -table[i])
        for r, i in enumerate(ordered, 1):
            ranks.setdefault(i, {})[name] = r
    if not ranks:
        return []

    scored = []
    for i, rk in ranks.items():
        fused = sum(1.0 / (RRF_K + r) for r in rk.values())
        scored.append({"idx": i, "score": round(fused, 5), "ranks": rk,
                       "anchor_hits": anchor_hits(chunks[i], key)})

    # Type priority as a STRUCTURAL sort key ahead of score, not a rerank.
    prio = {k: n for n, k in enumerate(C.CORPUS_KIND_PRIORITY)}
    scored.sort(key=lambda s: (prio.get(chunks[s["idx"]]["kind"], 99), -s["score"]))
    return scored[:want]


# --------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------
def main(force: bool = False) -> None:
    manifest = _manifest_hash()
    with cached_stage("s21_evidence",
                      sources=[C.CORPUS_CHUNKS_JSONL],
                      output=C.EVIDENCE_MAP_JSON, force=force,
                      # The corpus manifest hash is appended HERE rather than
                      # sitting in FP_EVIDENCE, for the same reason s19 appends
                      # the prompt hash: the map is a function of the corpus
                      # CONTENT, and content is not recoverable from a constant.
                      extra=C.FP_EVIDENCE + (manifest,)) as ran:
        if not ran:
            return
        _run(manifest)


def _manifest_hash() -> str:
    if not C.CORPUS_REPORT_JSON.exists():
        return "no-corpus"
    rep = json.loads(C.CORPUS_REPORT_JSON.read_text(encoding="utf-8"))
    return rep.get("manifest_hash") or corpus_manifest_hash(rep.get("documents", {}))


def _run(manifest: str) -> None:
    t0 = time.time()
    chunks = load_chunks()
    keys = enumerate_keys()
    log(f"corpus {len(chunks)} chunks  |  keys {len(keys)}  |  manifest {manifest}")

    con = build_index(chunks)
    dense = Dense()
    log(f"dense channel: {dense.note}")

    dmat = None
    if dense.ok:
        with_t = time.time()
        dmat = dense.encode_docs(chunks)
        log(f"  encoded {len(chunks)} chunks in {time.time() - with_t:.0f} s")
        qmat = dense.encode_queries([k["query"] for k in keys])
        sims = qmat @ dmat.T
    else:
        sims = None

    entries: dict[str, dict] = {}
    stats: Counter = Counter()
    for n, key in enumerate(keys):
        want = C.EVIDENCE_ACTIONS_K if key["channel"] == "actions" else C.EVIDENCE_CONTEXT_K
        lex = lexical(con, key, C.RAG_RERANK_TOP_N)
        dsc: dict[int, float] = {}
        if sims is not None:
            row = sims[n]
            top = row.argsort()[::-1][: C.RAG_RERANK_TOP_N]
            dsc = {int(i): float(row[i]) for i in top}
        # Reranked over the UNION of what the two retrievers proposed, and only
        # over what the anchor gate already admits -- a cross-encoder is
        # quadratic in candidates and there is no reason to score passages that
        # cannot be selected.
        cross: dict[int, float] = {}
        if dense.rerank_ok:
            pool = set(channel_pool(chunks, key))
            cands = [i for i in dict.fromkeys(list(lex) + list(dsc))
                     if i in pool and anchor_hits(chunks[i], key)]
            if cands:
                scores = dense.rerank(key["query"],
                                      [f"{chunks[i]['title']}. {chunks[i]['text']}"
                                       for i in cands])
                cross = dict(zip(cands, scores))
        picked = select(chunks, key, lex, dsc, want, cross)

        passages = []
        for p in picked:
            ch = chunks[p["idx"]]
            # Decidable, and therefore reviewable: a passage that names the
            # parameter once is very often naming it in passing.
            weak = p["anchor_hits"] < C.EVIDENCE_MIN_ANCHOR_HITS
            passages.append({
                "chunk_id": ch["chunk_id"], "doc_key": ch["doc_key"],
                "kind": ch["kind"], "strength": ch["strength"],
                "certainty": ch["certainty"],
                "heading_path": ch["heading_path"],
                "citation": ch["citation"],
                # VERBATIM. This exact string is what the assembler emits and
                # what grounding.py checks by byte identity. Nothing downstream
                # may reformat it.
                "text": ch["text"],
                "score": p["score"], "ranks": p["ranks"],
                "anchor_hits": p["anchor_hits"],
                "flag": "weak_match" if weak else None,
            })
        status = ("missing" if not passages
                  else "weak" if all(p["flag"] for p in passages) else "ok")
        stats[f"{key['channel']}_{status}"] += 1
        entries[key["key"]] = {
            "channel": key["channel"], "parameter": key["parameter"],
            "band": key["band"], "kind": key["kind"], "label": key["label"],
            "query": key["query"], "status": status,
            "pool_size": len(channel_pool(chunks, key)),
            "passages": passages,
        }

    payload = {
        "schema_version": C.EVIDENCE_SCHEMA_VERSION,
        "corpus_manifest_hash": manifest,
        "corpus_chunks": len(chunks),
        "dense_channel": dense.note,
        "min_anchor_hits": C.EVIDENCE_MIN_ANCHOR_HITS,
        "fusion": f"reciprocal rank fusion, k={RRF_K}",
        "kind_priority": list(C.CORPUS_KIND_PRIORITY),
        "keys": entries,
    }
    C.EVIDENCE_MAP_JSON.write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                                   encoding="utf-8")
    log(f"  {C.EVIDENCE_MAP_JSON.name}: {len(entries)} keys  "
        f"{dict(sorted(stats.items()))}")

    _write_review(payload, chunks, time.time() - t0)


def _write_review(payload: dict, chunks: list[dict], seconds: float) -> None:
    """reports/evidence_map.md -- the document the retrieval is judged on.

    Every entry is printed IN FULL. With a closed query set this is the real
    evaluation: a hit-rate against a relevance set nobody wrote is a number
    about a guess, and 57 passages is a sitting's work.
    """
    ent = payload["keys"]
    weak = [k for k, v in ent.items() if v["status"] == "weak"]
    missing = [k for k, v in ent.items() if v["status"] == "missing"]
    ok = [k for k, v in ent.items() if v["status"] == "ok"]

    L: list[str] = []
    L.append("# Evidence map -- review document\n")
    L.append(f"- corpus manifest `{payload['corpus_manifest_hash']}`, "
             f"{payload['corpus_chunks']} chunks\n")
    L.append(f"- dense channel: {payload['dense_channel']}\n")
    L.append(f"- **{len(ok)} accepted**, **{len(weak)} weak**, "
             f"**{len(missing)} with no admissible passage**, "
             f"{len(ent)} keys total\n")
    L.append(f"- built in {seconds:.0f} s\n\n")

    L.append("## Needs your eye\n\n")
    if not weak and not missing:
        L.append("Nothing flagged.\n\n")
    else:
        L.append("A key with no admissible passage is emitted with its passage "
                 "**suppressed**, not dropped -- a missing key has to be visible "
                 "in the output rather than absent from it.\n\n")
        if missing:
            L.append("**No admissible passage** (nothing in the corpus mentions "
                     "this parameter in this channel):\n\n")
            for k in missing:
                L.append(f"- `{k}` -- pool {ent[k]['pool_size']} chunks\n")
            L.append("\n")
        if weak:
            L.append(f"**Weak match** (passage names the parameter fewer than "
                     f"{payload['min_anchor_hits']} times, so it is probably "
                     f"mentioning it in passing):\n\n")
            for k in weak:
                p = ent[k]["passages"][0]
                L.append(f"- `{k}` -- {p['anchor_hits']} mention(s) -- "
                         f"{p['citation']} -- \"{p['text'][:110]}...\"\n")
            L.append("\n")

    for channel, title in (("context", "Definitional context"),
                           ("actions", "Suggested actions")):
        L.append(f"## {title}\n\n")
        for k, v in ent.items():
            if v["channel"] != channel:
                continue
            L.append(f"### `{k}`\n\n")
            L.append(f"query: `{v['query']}` | pool: {v['pool_size']} | "
                     f"status: **{v['status']}**\n\n")
            if not v["passages"]:
                L.append("_no admissible passage -- suppressed in output_\n\n")
                continue
            for p in v["passages"]:
                grade = (f" -- **{p['strength']} recommendation, "
                         f"{p['certainty']} certainty**" if p["strength"] else "")
                flag = "  :warning: weak" if p["flag"] else ""
                L.append(f"- **{p['citation']}** ({p['kind']}, rrf {p['score']}, "
                         f"ranks {p['ranks']}, {p['anchor_hits']} mention(s))"
                         f"{grade}{flag}\n")
                if p["heading_path"]:
                    L.append(f"  - section: {' > '.join(p['heading_path'])}\n")
                L.append(f"  - > {p['text']}\n\n")
    C.EVIDENCE_MAP_MD.write_text("".join(L), encoding="utf-8")
    log(f"  {C.EVIDENCE_MAP_MD.name} written ({len(ok)} accepted, "
        f"{len(weak)} weak, {len(missing)} missing)")


def load_map() -> dict:
    if not C.EVIDENCE_MAP_JSON.exists():
        raise SystemExit("s21 has not run -- evidence_map.json is missing")
    return json.loads(C.EVIDENCE_MAP_JSON.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _ap.add_argument("--force", action="store_true",
                     help="rebuild even when the manifest is current -- needed "
                          "after a code change, which no fingerprint covers")
    main(force=_ap.parse_args().force)
