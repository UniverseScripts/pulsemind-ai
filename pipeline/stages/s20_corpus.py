"""Stage 20 -- the guideline corpus: published PDFs into quotable, reviewable chunks.

WHAT THIS STAGE IS FOR
----------------------
Everything downstream quotes this text VERBATIM to a clinician with a citation
attached. That single fact sets every decision here:

  * Character fidelity beats layout fidelity. A reflowed paragraph is readable;
    "cmH O" where the source says "cmH2O" is a clinical error with a citation
    stapled to it saying the CDC wrote it.
  * Extraction damage must be REPAIRED or COUNTED, never averaged over. Three
    specific defects were measured in this corpus and each is handled by name
    below; anything unrecognised is counted into reports/corpus.json so it shows
    up as a number rather than as a plausible-looking sentence.
  * A passage with no attribution is worse than a missing one. C.CORPUS_DOCS is
    the admission list and this stage REFUSES anything not on it.

EXTRACTOR CHOICE -- MEASURED
----------------------------
pypdfium2 for text, pdfplumber for typography, pypdf for structural metadata.
All three were probed against all three PDFs on 2026-08-09 before being chosen.
The reasoning, the numbers and the three extraction defects corrected here are
in config.py beside CORPUS_HYPHEN_DEFAULT_JOIN -- one copy, so they cannot drift.

pdfplumber is still here for one thing it does better: per-character font names
and sizes, which is how headings are found. The two extractors are aligned at
LINE level by stripping every non-alphanumeric character -- which incidentally
makes pdfplumber's glued "AARCCLINICALPRACTICE" and pdfium's spaced
"AARC CLINICAL PRACTICE" compare equal. The match rate is reported, not assumed.

CHUNKING
--------
Structure-aware, with the heading path spliced onto the front of the chunk as
its title (MedRAG, ACL Findings 2024). That is contextual retrieval without
paying an LLM to write the context: a chunk that says "the daily minimum" is
useless on its own and precise under the heading path it came from.

Target size follows the same paper's measured snippets -- 119 words for
StatPearls, the closest analogue to a point-of-care guideline -- not the ~400
that a generic splitter would use.

NO PATIENT DATA
---------------
The corpus is published guideline text. Its outputs may be read, reviewed and
quoted freely; that is the opposite of build/risk_records.jsonl.gz and is why
the review document in reports/ can print passages in full.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

from .. import config as C
from ..common import cached_stage, log

# --------------------------------------------------------------------------
# Extraction repair
# --------------------------------------------------------------------------
# pdfium emits U+FFFE where the source had a hyphen at a line break. It is
# genuinely ambiguous -- "rec|ommendations" wants a join, "ventilator|associated"
# wants to keep the hyphen -- so it is resolved against the corpus's own
# vocabulary rather than guessed.
SOFT_BREAK = "￾"

# The AARC subset font maps '=' to U+00BC. Left alone it reads as "VT 1/4 tidal
# volume"; NFKC would make that worse by expanding it to "1/4". Measured: 30
# occurrences, all in the AARC guideline, 29 of them flanked by spaces.
BAD_EQUALS = "¼"

MATH_ITALIC = re.compile(r"[\U0001D400-\U0001D7FF]")

# GRADE statements. The strength/certainty parenthetical is the whole point --
# an action shipped without its evidence grade is an opinion with a citation.
# The body may not itself contain another "We recommend/suggest". Without that
# guard, a statement that happens to carry no grade swallows the NEXT statement's
# parenthetical and is published under a strength it was never given.
RE_GRADE = re.compile(
    r"We\s+(?P<verb>recommend|suggest)\b"
    r"(?P<body>(?:(?!We\s+(?:recommend|suggest)).){0,600}?)"
    r"\((?P<strength>strong|conditional|weak)\s+recommendation,\s*"
    r"(?P<certainty>very\s+low|low|moderate|high)\s+certainty\)",
    re.IGNORECASE | re.DOTALL)

# Protocol outline items: "A. Tidal Volume: 4 to 12 mL/Kg ...", "3. PEEP 5 to 15
# cm H2O ...". These carry the concrete parameter targets, which is exactly what
# an action keyed on (band, parameter) needs.
RE_OUTLINE = re.compile(r"^\s*(?:[IVXL]+\.|[A-Z]\.|\d+\.)\s+\S")

RE_WS = re.compile(r"[ \t]+")
RE_NONALNUM = re.compile(r"[^a-z0-9]+")


def _norm_key(s: str) -> str:
    """Line identity for cross-extractor matching.

    Stripping every non-alphanumeric is what makes pdfplumber's glued
    "AARCCLINICALPRACTICEGUIDELINES" equal pdfium's spaced form. That is the
    whole reason line-level matching works at all on this corpus.
    """
    return RE_NONALNUM.sub("", s.lower())


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Vocabulary, built for one purpose: resolving the soft break
# --------------------------------------------------------------------------
def build_vocabulary(raw_texts: list[str]) -> Counter:
    """Token counts over the corpus, EXCLUDING fragments adjacent to a break.

    Counting the fragments would let "microbiologi" attest itself as a word and
    then defeat the very rule that is trying to decide whether it is one.
    """
    vocab: Counter = Counter()
    for t in raw_texts:
        # blank out each break and the word on either side of it
        masked = re.sub(r"[A-Za-z]*" + SOFT_BREAK + r"\s*[A-Za-z]*", " ", t)
        for tok in re.findall(r"[A-Za-z][A-Za-z\-]*[A-Za-z]|[A-Za-z]", masked):
            vocab[tok.lower()] += 1
    return vocab


def _resolve_break(a: str, b: str, vocab: Counter) -> tuple[str, str]:
    """Decide join / hyphen for one soft break. Returns (glue, rule_used)."""
    if vocab.get((a + b).lower()):
        return "", "joined_attested"
    if vocab.get((a + "-" + b).lower()):
        return "-", "hyphen_attested"
    # Neither compound is attested. If BOTH halves are real words on their own,
    # this is a compound the corpus happens not to repeat ("value|based",
    # "pressure|ventilation"); if either half is not a word, it is a syllable
    # split ("microbiologi|cally", "var|iations").
    if vocab.get(a.lower()) and vocab.get(b.lower()):
        return "-", "both_halves_are_words"
    return "", "default_join"


def repair(text: str, vocab: Counter, stats: Counter,
           unresolved: list[dict], doc: str, count: bool = True) -> str:
    """Apply every known extraction defect fix. Anything else is left alone.

    `count=False` for the GRADE pass, which re-reads text the line pass has
    already repaired. Both passes must APPLY the repairs -- they run on separate
    raw strings -- but only one may COUNT them, or reports/corpus.json claims
    twice as many defects as the corpus contains.
    """
    if not count:
        stats = Counter()
        unresolved = []

    # 1. the mis-mapped '='
    n = text.count(BAD_EQUALS)
    if n:
        stats["bad_equals_fixed"] += n
        text = text.replace(BAD_EQUALS, "=")

    # 2. dead formula runs. A long row of MATHEMATICAL ITALIC letters is a
    #    formula pdfium could not recover ("SIR = OOOOOOOO (OO)HHHH"); NFKC
    #    would turn it into confident-looking garbage, so it is excised and
    #    counted instead. Isolated ones are real characters and are folded.
    def _kill(m: re.Match) -> str:
        stats["formula_chars_excised"] += len(m.group(0))
        stats["formula_runs_excised"] += 1
        return " [formula not recoverable from PDF] "

    text = re.sub(r"[\U0001D400-\U0001D7FF]{%d,}" % (C.CORPUS_MATH_ITALIC_MAX_RUN + 1),
                  _kill, text)
    if MATH_ITALIC.search(text):
        stats["math_italic_folded"] += len(MATH_ITALIC.findall(text))
        text = "".join(unicodedata.normalize("NFKC", ch) if MATH_ITALIC.match(ch) else ch
                       for ch in text)

    # 3. C0 controls the subset fonts leave behind (AARC page footers carry
    #    \x01 where a bullet belongs). Never silently deleted: they become a
    #    space, so word boundaries survive.
    ctl = sum(1 for ch in text if ord(ch) < 32 and ch not in "\r\n\t")
    if ctl:
        stats["control_chars_spaced"] += ctl
        text = "".join(" " if (ord(ch) < 32 and ch not in "\r\n\t") else ch for ch in text)

    # 4. the soft break
    def _fix(m: re.Match) -> str:
        a, b = m.group(1), m.group(2)
        glue, rule = _resolve_break(a, b, vocab)
        stats[f"hyphen_{rule}"] += 1
        if rule in ("both_halves_are_words", "default_join"):
            unresolved.append({"doc": doc, "left": a, "right": b,
                               "rule": rule, "result": a + glue + b})
        return a + glue + b

    text = re.sub(r"([A-Za-z]+)" + SOFT_BREAK + r"\s*([A-Za-z]+)", _fix, text)
    # any break not between two words (line-end, punctuation) is a plain hyphen
    left = text.count(SOFT_BREAK)
    if left:
        stats["hyphen_bare"] += left
        text = text.replace(SOFT_BREAK, "-")
    return text


# --------------------------------------------------------------------------
# Reading a document
# --------------------------------------------------------------------------
def _pdfium_lines(page) -> list[tuple[str, float]]:
    """Visual lines plus the tallest glyph on each, from character boxes.

    count_chars() and len(get_text_range()) are equal on this corpus, so the
    character index is an exact offset into the text -- no fuzzy alignment.
    """
    tp = page.get_textpage()
    text = tp.get_text_range()
    out: list[tuple[str, float]] = []
    i = 0
    for raw in text.split("\r\n"):
        h = 0.0
        for k in range(i, i + len(raw)):
            try:
                l, b, r, t = tp.get_charbox(k)
            except Exception:      # pdfium returns no box for some glyphs
                continue
            if t - b > h:
                h = t - b
        out.append((raw, h))
        i += len(raw) + 2          # the "\r\n" we split on
    return out


def _pdfplumber_typography(page) -> dict[str, tuple[float, bool, str]]:
    """normalised line text -> (max font size, any bold, modal font name).

    The font name is the signal that actually works on a journal reprint. The
    AARC guideline sets "Introduction" in AdvP6960 at 10pt against AdvP6975 body
    at 10pt: same size, no "Bold" anywhere in the name, and the only difference
    is which subset font the typesetter reached for. Size and the bold flag both
    miss it, which is why every AARC chunk claimed to sit under the author line.
    """
    try:
        words = page.extract_words(extra_attrs=["fontname", "size"], x_tolerance=1.5)
    except Exception:
        return {}
    # Grouped BOTH ways, on purpose. The AARC guideline is a two-column journal
    # reprint where grouping on `top` alone welds the left-column line to the
    # right-column line at the same height into one pdfplumber "line" that no
    # pdfium line can ever match -- that held its match rate at 36%. But
    # splitting at the midpoint unconditionally breaks the single-column
    # documents, whose lines legitimately cross it (NHSN fell 71% -> 27%).
    # Emitting both groupings costs nothing: the dict is keyed by line text, so
    # the extra entries are simply extra chances to match.
    mid = float(page.width) / 2.0
    lines: dict[tuple[float, int], list[dict]] = {}
    for w in words:
        top = round(w["top"], 0)
        lines.setdefault((top, 0 if w["x0"] < mid else 1), []).append(w)
        lines.setdefault((top, 9), []).append(w)      # 9 = whole-width grouping
    out: dict[str, tuple[float, bool, str]] = {}
    for _, ws in lines.items():
        ws.sort(key=lambda w: w["x0"])
        txt = " ".join(w["text"] for w in ws)
        key = _norm_key(txt)
        if not key:
            continue
        size = max(float(w.get("size") or 0) for w in ws)
        bold = any("bold" in str(w.get("fontname", "")).lower() for w in ws)
        fonts = Counter(str(w.get("fontname", "")).split("+")[-1] for w in ws)
        prev = out.get(key)
        if prev is None or size > prev[0]:
            out[key] = (size, bold, fonts.most_common(1)[0][0])
    return out


def read_document(path: Path) -> list[dict]:
    """One entry per page: raw text and per-line typography."""
    import pdfplumber
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    pages: list[dict] = []
    with pdfplumber.open(path) as pl:
        for idx in range(len(doc)):
            lines = _pdfium_lines(doc[idx])
            typo = _pdfplumber_typography(pl.pages[idx]) if idx < len(pl.pages) else {}
            pages.append({"page": idx + 1, "lines": lines, "typo": typo})
    return pages


# --------------------------------------------------------------------------
# Headings
# --------------------------------------------------------------------------
# A table-of-contents line: dot leaders, or a trailing bare page number. These
# are typographically identical to real headings and they poisoned the whole
# heading stack -- every NHSN chunk came out nested under "Table of Contents".
RE_TOC = re.compile(r"\.{3,}|\s\d{1,3}\s*$")
RE_ROMAN = re.compile(r"^\s*[IVXL]+\.\s+\S")


def _looks_like_heading(s: str) -> bool:
    """Shape test, applied BEFORE the typographic one.

    Typography alone is not enough on a journal reprint: glyph heights drift
    enough that mid-sentence body text clears the size threshold. A heading
    starts a phrase -- so a line beginning lower-case ("during NIV and ...") or
    with a bracket ("(SpO2 < 88%) or hyperoxemia ...") is a continuation, and
    both of those shipped as heading paths before this test existed.
    """
    if not s or len(s.split()) > 14 or s.endswith((".", ";", ",")):
        return False
    if RE_TOC.search(s):
        return False
    if not (s[0].isupper() or s[0].isdigit()):
        return False
    if sum(ch.isalpha() for ch in s) < 3:
        return False
    # A table row is typographically indistinguishable from a heading in NHSN --
    # short, no terminal stop, often in the header font -- and shipped section
    # labels like "Time 18:00 19:00 20:00", "QAD No No No Yes Yes" and
    # "5 6 0.70 (70%) VAC". Numeric density is what separates the two.
    if _table_like(s):
        return False
    # a sentence fragment carrying a citation marker is body text
    return not re.search(r"\bet al\b|\d{4};|\bp\s*=|\bn\s*=", s, re.IGNORECASE)


def _table_like(s: str) -> bool:
    """True when a line reads as tabular data rather than prose."""
    toks = s.split()
    if len(toks) < 3:
        return False
    numeric = sum(1 for t in toks if re.fullmatch(r"[-+]?[\d.,:%()/]+", t))
    if numeric / len(toks) >= 0.4:
        return True
    # a run of repeated one-word cells: "No No No Yes Yes Yes" / "-- -- -- --"
    return bool(re.search(r"(?:\b(\S{1,4})\b[ ]+)(?:\S{1,4}[ ]+){3,}", s))


RE_DIGITS = re.compile(r"\d+")


def strip_running_boilerplate(pages: list[dict], stats: Counter) -> None:
    """Drop the running header/footer. Mutates `pages`.

    Every NHSN page opens with "January 2026 Device-associated Module VAE 10-2",
    which then lands inside the first chunk of all 44 pages -- retrieved,
    embedded, and quoted to a clinician as if it were guideline text. Detected
    by recurrence rather than by position, with page numbers masked so
    "10-2" and "10-3" count as the same line.
    """
    if len(pages) < 4:
        return
    seen: Counter = Counter()
    for pg in pages:
        for key in {_norm_key(RE_DIGITS.sub("#", t)) for t, _ in pg["lines"] if t.strip()}:
            seen[key] += 1
    cut = max(3, int(len(pages) * 0.25))
    boiler = {k for k, n in seen.items() if n >= cut and 0 < len(k) <= 90}
    if not boiler:
        return
    dropped = 0
    for pg in pages:
        keep = []
        for t, h in pg["lines"]:
            s = t.strip()
            if not s:
                keep.append((t, h))
                continue
            # A bare page marker ("10-2") survives the recurrence test because
            # masking its digits leaves nothing to key on. It then opens the
            # first chunk of the page as if it were a sentence.
            if len(s.split()) <= 3 and not any(ch.isalpha() for ch in s):
                dropped += 1
                continue
            if _norm_key(RE_DIGITS.sub("#", t)) in boiler:
                dropped += 1
                continue
            keep.append((t, h))
        pg["lines"] = keep
    stats["boilerplate_lines_dropped"] = dropped
    stats["boilerplate_patterns"] = len(boiler)


def _heading_levels(pages: list[dict], role: str, stats: Counter) -> None:
    """Tag each line with a heading level in {0, 1}, or None. Mutates `pages`.

    TWO LEVELS, DELIBERATELY. An n-level hierarchy inferred from font sizes was
    the first attempt and it does not survive contact with these PDFs: ranking
    by a mix of reported font size and measured glyph height put nearly every
    heading in its own level, `del stack[lvl:]` then almost never popped, and
    the stack grew monotonically until every chunk in a 44-page document claimed
    to live under "Table of Contents > Figure". Two levels -- section and
    subsection -- cannot express that failure.

    A PROTOCOL is not read typographically at all. Its structure is explicit
    (I. / II. / III. for sections, A. / B. / 1. for steps) and the steps are
    CONTENT: emitting "D. Rate: 8 to 26 breaths/minute" as a heading path both
    loses the step and mislabels everything after it.
    """
    body_h: Counter = Counter()
    body_size: Counter = Counter()
    body_font: Counter = Counter()
    for pg in pages:
        for txt, h in pg["lines"]:
            if len(txt.strip()) >= 20:            # body-length lines set the baseline
                body_h[round(h, 1)] += len(txt)
                ts = pg["typo"].get(_norm_key(txt))
                if ts:
                    body_size[round(ts[0], 1)] += len(txt)
                    body_font[ts[2]] += len(txt)
    base_h = body_h.most_common(1)[0][0] if body_h else 0.0
    base_size = body_size.most_common(1)[0][0] if body_size else 0.0
    base_font = body_font.most_common(1)[0][0] if body_font else ""
    stats["body_glyph_height"] = base_h
    stats["body_font_size"] = base_size
    stats["body_font"] = base_font

    matched = total = 0
    ranks: Counter = Counter()
    for pg in pages:
        tagged = []
        for txt, h in pg["lines"]:
            s = txt.strip()
            total += 1
            ts = pg["typo"].get(_norm_key(txt))
            if ts:
                matched += 1
            size, bold, font = ts if ts else (0.0, False, "")
            if role == "protocol":
                is_head = bool(RE_ROMAN.match(s))
                rank = 99.0 if is_head else 0.0
            else:
                # `ts is not None` is a REQUIREMENT, not an optimisation. Glyph
                # height alone was the third signal here and it is the noisy
                # one: on dense two-column body text it drifts past any
                # threshold that still catches real headings, and it produced
                # section labels like "29.4% (P = .71).62 Further" and
                # "Joel T Glogowski, and Dean". A line with no font data is
                # left as body text -- an unlabelled chunk beats a mislabelled
                # one, because the label is what gets shown beside the quote.
                is_head = False
                if ts is not None and _looks_like_heading(s):
                    big_font = base_size and size > base_size + 0.4
                    bold_head = bold and base_size and size >= base_size - 0.1
                    other_font = bool(font) and font != base_font
                    is_head = bool(big_font or bold_head or other_font)
                # Rank orders the levels; a differently-fonted heading at body
                # size is a SUBsection, so it must not outrank a larger one.
                rank = round(size + (1.0 if big_font else 0.0), 1) if is_head else 0.0
            if is_head:
                ranks[rank] += 1
            tagged.append({"text": txt, "h": h, "size": size, "font": font,
                           "bold": bold, "head": is_head, "rank": rank})
        pg["tagged"] = tagged
    stats["typography_line_match_rate"] = round(matched / max(total, 1), 4)
    stats["headings_detected"] = sum(ranks.values())

    # Level 0 is the largest heading rank present; everything else is level 1.
    top = max(ranks, default=0.0)
    for pg in pages:
        for ln in pg["tagged"]:
            ln["level"] = None if not ln["head"] else (0 if ln["rank"] >= top else 1)


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def _flush(buf: list[str], meta: dict, out: list[dict], seen: set[str]) -> None:
    text = RE_WS.sub(" ", " ".join(buf)).strip()
    if len(text.split()) < C.CORPUS_CHUNK_MIN_WORDS:
        return
    if text.count("...") >= 3:   # a table of contents is navigation, not content
        return
    key = _norm_key(text)
    if key in seen:                      # the AARC abstract repeats every
        return                           # recommendation the body already has
    seen.add(key)
    meta = dict(meta)
    # Retagged here rather than at flush sites: whether a run of lines is a
    # criteria table is a property of the assembled text, not of the heading it
    # arrived under. Without this the "table" tier of CORPUS_KIND_PRIORITY has
    # no members at all and the ordering is silently two-tier.
    if meta["kind"] == "prose" and _numeric_dense(text):
        meta["kind"] = "table"
    out.append({**meta, "text": text, "words": len(text.split())})


def _numeric_dense(text: str) -> bool:
    toks = text.split()
    if not toks:
        return False
    numeric = sum(1 for t in toks if re.fullmatch(r"[-+]?[\d.,:%()/]+", t))
    return numeric / len(toks) >= 0.25


def chunk_document(path: Path, spec: dict, pages: list[dict],
                   vocab: Counter, stats: Counter,
                   unresolved: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    seen: set[str] = set()
    stack: list[str] = []
    role = spec["role"]

    for pg in pages:
        buf: list[str] = []
        pg["path"] = list(stack)          # heading context as this page opens
        for ln in pg["tagged"]:
            raw = ln["text"].strip()
            if not raw:
                continue
            if ln["head"]:
                _flush(buf, _meta(spec, pg, stack, "prose"), chunks, seen)
                buf = []
                lvl = ln["level"] or 0
                del stack[lvl:]
                stack.append(repair(raw, vocab, stats, unresolved, path.name))
                if len(stack) > len(pg["path"]):
                    pg["path"] = list(stack)   # keep the most specific seen here
                continue
            fixed = repair(raw, vocab, stats, unresolved, path.name)
            # A protocol outline item is a unit: never glue it to its neighbour.
            if role == "protocol" and RE_OUTLINE.match(raw) and buf:
                _flush(buf, _meta(spec, pg, stack, "recommendation"), chunks, seen)
                buf = []
            buf.append(fixed)
            if len(" ".join(buf).split()) >= C.CORPUS_CHUNK_MAX_WORDS:
                kind = "recommendation" if role == "protocol" else "prose"
                _flush(buf, _meta(spec, pg, stack, kind), chunks, seen)
                buf = []
        kind = "recommendation" if role == "protocol" else "prose"
        _flush(buf, _meta(spec, pg, stack, kind), chunks, seen)

    # GRADE statements are extracted as their own chunks, over the page text,
    # because they routinely straddle the line breaks the loop above splits on.
    #
    # The chunk is EXACTLY the graded statement -- from its own "We recommend"
    # to its closing certainty parenthetical -- and nothing else. Anything wider
    # both loses the one-statement-one-action property the actions channel needs
    # and, on this document, swallows the whole abstract: the AARC abstract is a
    # single run-on sentence carrying all fifteen recommendations, so extending
    # back to the previous full stop produced fifteen near-identical chunks.
    if role == "guideline":
        best: dict[str, dict] = {}
        for pg in pages:
            page_text = RE_WS.sub(" ", repair(
                " ".join(t for t, _ in pg["lines"]),
                vocab, stats, unresolved, path.name, count=False))
            for m in RE_GRADE.finditer(page_text):
                text = page_text[m.start():m.end()].strip()
                key = _norm_key(text)
                if len(text.split()) < 6:
                    continue
                cand = {**_meta(spec, pg, pg["path"], "recommendation"),
                        "text": text, "words": len(text.split()),
                        "strength": m.group("strength").lower(),
                        "certainty": re.sub(r"\s+", " ", m.group("certainty").lower())}
                # Every statement also appears verbatim in the abstract on p.1,
                # where it has no section context. Keep the occurrence with the
                # most specific heading path so the citation lands in the body.
                prev = best.get(key)
                if prev is None or len(cand["heading_path"]) > len(prev["heading_path"]):
                    best[key] = cand
        for key, cand in best.items():
            if key in seen:
                continue
            seen.add(key)
            chunks.append(cand)
    for i, ch in enumerate(chunks):
        ch["chunk_id"] = f"{spec['key']}:p{ch['page']:03d}:c{i:04d}"
    return chunks


def _meta(spec: dict, pg: dict, stack: list[str], kind: str) -> dict:
    path = [s for s in stack if s]
    return {
        "doc_key": spec["key"], "doc_title": spec["title"], "doc_role": spec["role"],
        "page": pg["page"], "heading_path": list(path),
        # MedRAG splices the heading path onto the snippet as its title. This is
        # the field that gets embedded and indexed, not `text` alone.
        "title": " > ".join(path) if path else spec["title"],
        "kind": kind, "strength": None, "certainty": None,
        "citation": f"{spec['cite']}, p.{pg['page']}",
    }


# --------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------
def discover() -> tuple[list[tuple[Path, dict]], list[str]]:
    """Registered PDFs, and the names of any that are not registered."""
    found: list[tuple[Path, dict]] = []
    refused: list[str] = []
    for d in (C.RAG_DOCS_DIR, C.RAG_EXTRA_DIR):
        if not d.exists():
            continue
        for p in sorted(d.glob("*.pdf")):
            spec = C.CORPUS_DOCS.get(p.name)
            if spec is None:
                refused.append(str(p))
            else:
                found.append((p, {**spec, "tracked": d == C.RAG_DOCS_DIR}))
    return found, refused


def main(force: bool = False) -> None:
    docs, _ = discover()
    with cached_stage("s20_corpus",
                      sources=[p for p, _ in docs],
                      output=C.CORPUS_CHUNKS_JSONL, force=force,
                      extra=C.FP_CORPUS) as ran:
        if not ran:
            return
        _run()


def _run() -> None:
    t0 = time.time()
    docs, refused = discover()
    if refused:
        # Refused, not ingested. An unattributable passage still gets quoted.
        for r in refused:
            log(f"  [red]REFUSED[/red] not in C.CORPUS_DOCS: {r}")
    if not docs:
        raise SystemExit("s20: no registered PDFs found in "
                         f"{C.RAG_DOCS_DIR} or {C.RAG_EXTRA_DIR}")

    log(f"registered documents: {len(docs)}")
    read: list[tuple[Path, dict, list[dict]]] = []
    for path, spec in docs:
        pages = read_document(path)
        read.append((path, spec, pages))
        log(f"  {path.name:38s} {len(pages):>3d} pages  role={spec['role']}")

    # One vocabulary over the whole corpus -- the soft-break rule needs to see
    # every document before it can decide any of them.
    vocab = build_vocabulary(["\n".join(t for t, _ in pg["lines"])
                              for _, _, pages in read for pg in pages])
    log(f"vocabulary: {len(vocab):,} distinct tokens")

    stats: Counter = Counter()
    unresolved: list[dict] = []
    all_chunks: list[dict] = []
    per_doc: dict[str, dict] = {}

    for path, spec, pages in read:
        dstats: Counter = Counter()
        strip_running_boilerplate(pages, dstats)
        _heading_levels(pages, spec["role"], dstats)
        chunks = chunk_document(path, spec, pages, vocab, dstats, unresolved)
        all_chunks += chunks
        kinds = Counter(c["kind"] for c in chunks)
        graded = sum(1 for c in chunks if c["strength"])
        per_doc[path.name] = {
            "key": spec["key"], "title": spec["title"], "issuer": spec["issuer"],
            "year": spec["year"], "role": spec["role"], "cite": spec["cite"],
            "url": spec["url"], "tracked_in_git": spec["tracked"],
            "sha256": sha256(path), "bytes": path.stat().st_size,
            "pages": len(pages), "chunks": len(chunks),
            "chunks_by_kind": dict(kinds), "graded_statements": graded,
            "words_total": sum(c["words"] for c in chunks),
            "words_median": _median([c["words"] for c in chunks]),
            "extraction": {k: v for k, v in dstats.items()},
        }
        stats.update({k: v for k, v in dstats.items() if isinstance(v, int)})
        log(f"  {path.name:38s} {len(chunks):>4d} chunks  "
            f"{dict(kinds)}  graded={graded}  "
            f"typo-match={dstats['typography_line_match_rate']:.0%}")

    with C.CORPUS_CHUNKS_JSONL.open("w", encoding="utf-8") as fh:
        for ch in all_chunks:
            fh.write(json.dumps(ch, ensure_ascii=False) + "\n")
    log(f"  {C.CORPUS_CHUNKS_JSONL.name}: {len(all_chunks):,} chunks, "
        f"{C.CORPUS_CHUNKS_JSONL.stat().st_size / 1e6:.2f} MB")

    manifest_hash = corpus_manifest_hash(per_doc)
    report = {
        "schema_version": C.CORPUS_SCHEMA_VERSION,
        "manifest_hash": manifest_hash,
        "documents": per_doc,
        "refused_unregistered": refused,
        "totals": {
            "documents": len(per_doc),
            "pages": sum(d["pages"] for d in per_doc.values()),
            "chunks": len(all_chunks),
            "words": sum(c["words"] for c in all_chunks),
            "by_kind": dict(Counter(c["kind"] for c in all_chunks)),
            "graded_statements": sum(1 for c in all_chunks if c["strength"]),
        },
        "extraction_repairs": {k: v for k, v in sorted(stats.items())},
        # Printed in full on purpose. These are the cases where the soft-break
        # rule had no direct evidence and fell back; they are the most likely
        # place for a silently wrong word, so they are reviewable by eye.
        "hyphen_fallbacks": unresolved,
        "chunk_words": {
            "min": min((c["words"] for c in all_chunks), default=0),
            "median": _median([c["words"] for c in all_chunks]),
            "max": max((c["words"] for c in all_chunks), default=0),
        },
        "seconds": round(time.time() - t0, 1),
    }
    C.CORPUS_REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    log(f"  {C.CORPUS_REPORT_JSON.name} written  manifest_hash={manifest_hash}")


def corpus_manifest_hash(per_doc: dict) -> str:
    """Identity of the corpus CONTENT, for the s21 fingerprint.

    Hashing the document hashes rather than the chunk file: the chunk file also
    moves when the chunker changes, and that is already covered by FP_CORPUS.
    """
    payload = json.dumps({k: v["sha256"] for k, v in sorted(per_doc.items())},
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_chunks() -> list[dict]:
    if not C.CORPUS_CHUNKS_JSONL.exists():
        raise SystemExit("s20 has not run -- corpus_chunks.jsonl is missing")
    with C.CORPUS_CHUNKS_JSONL.open(encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def _median(xs: list[int]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return float(s[n // 2]) if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _ap.add_argument("--force", action="store_true",
                     help="re-extract even when the manifest is current -- needed "
                          "after a code change, which no fingerprint covers")
    main(force=_ap.parse_args().force)
