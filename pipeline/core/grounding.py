"""Does this text contradict the record it claims to explain? Pure logic.

DELIBERATELY DOES NOT IMPORT explain.py
---------------------------------------
A checker that shares code with the generator inherits the generator's
assumptions and cannot catch them. Everything here is re-derived from the RECORD:
the set of numbers a sentence is allowed to contain is computed from the
record's own fields, not handed over by the payload builder. The duplication is
the point of the module.

There is no ground truth for a free-text rationale, so nothing here scores
quality. What it does is catch the failures that are mechanically decidable
against this particular record -- which is precisely why s17 emits contributors,
telemetry provenance and attribution totals rather than just a band.

TWO SEVERITIES
--------------
violation  decidable from the record. A generator producing one is wrong, and
           the run should not ship.
warning    heuristic. Worth a human reading, not worth failing a build over.
           Kept separate so the report is signal rather than noise.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ------------------------------------------------------------------ violations
FABRICATED_NUMBER = "FABRICATED_NUMBER"
PARAM_VALUE_MISMATCH = "PARAM_VALUE_MISMATCH"
IMPUTED_QUOTED = "IMPUTED_QUOTED"
BAND_MISMATCH = "BAND_MISMATCH"
TREND_CLAIM = "TREND_CLAIM"
ALTERED_QUOTE = "ALTERED_QUOTE"
UNCITED_QUOTE = "UNCITED_QUOTE"
DEFINITION_CONFLATION = "DEFINITION_CONFLATION"
# -------------------------------------------------------------------- warnings
POSSIBLE_BAND_MISMATCH = "POSSIBLE_BAND_MISMATCH"
POSSIBLE_TREND_CLAIM = "POSSIBLE_TREND_CLAIM"
DOC_AS_PHYSIOLOGY = "DOC_AS_PHYSIOLOGY"
UNSUPPORTED_DRIVER = "UNSUPPORTED_DRIVER"
TREATMENT_LANGUAGE = "TREATMENT_LANGUAGE"

VIOLATION, WARNING = "violation", "warning"

# Backward-looking trajectory language. All 109 features are point-in-time
# values plus staleness -- there is not one delta or slope among them -- so any
# claim about which way something has been moving is unsupported by construction.
#
# The distinction enforced here is TENSE, NOT TOPIC. The label IS a 6 h forward
# risk, so "risk of deterioration" is the band's meaning and must never be
# flagged; "has been deteriorating" is a claim about the past that nothing in
# the record can support. Hence `deteriorat(ing|ed)` and not `deterioration`.
# ⚠️ Do NOT add `since admission`. `hours_admit_to_icu` is a real feature and a
# frequent top contributor, so quoting it is correct; every genuine trend case
# ("falling since admission") is already caught by the directional verb.
_TREND = re.compile(
    r"\b(ris(ing|en)|rose|climb(ing|ed)|fall(ing|en)|fell|dropp(ing|ed)"
    r"|worsen(ing|ed)|improv(ing|ed|ement)|deteriorat(ing|ed)|escalat(ing|ed)"
    r"|progressively|gradually|trending|trajectory|steadily"
    r"|up from|down from|over the (last|past))\b",
    re.I)
# Same idea, weaker evidence: these are ordinary words for attribution direction
# ("PEEP increases the estimate") as often as for change over time, so they are
# reported rather than failed.
_TREND_SOFT = re.compile(r"\b(increas(e|es|ed|ing)|decreas(e|es|ed|ing)|trend"
                         r"|higher than|lower than|no longer)\b", re.I)
_TREATMENT = re.compile(
    r"\b(administer|titrat(e|ing|ion)|dos(e|ing|age)|mg\b|mcg\b|initiate"
    r"|prescrib(e|ing)|suction(ing)?|recruitment manoeuvre|recruitment maneuver"
    r"|prone|extubat(e|ion)|intubat(e|ion)|sedat(e|ion)|antibiotic|wean(ing)?"
    r"|bronchodilator|diuretic|escalate care)\b", re.I)
# Language that frames a parameter as a fact about ITS DOCUMENTATION rather than
# a physiological claim. Mentioning EtCO2 to say it was never charted is not
# narrating charting as physiology, and is not an unsupported driver either --
# both checks below stand down when this is nearby.
_CHARTING = re.compile(
    r"\b(chart(ed|ing)?|document(ed|ation)?|record(ed|ing)?|monitor(ed|ing)?"
    r"|measur(ed|ement)|time since|how often|frequency|not observed"
    r"|cohort default|carried forward|no value|unavailable|missing)\b", re.I)
_NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)")
# Sentence boundary. Requires whitespace after the punctuation so decimals
# ("4.2 h ago") are not treated as sentence ends.
_SENT = re.compile(r"[.;]\s+")

# Claiming the reading MEETS a surveillance definition. This is its own code
# rather than a special case of anything else because it is the specific false
# statement this corpus invites: NHSN's VAC criteria are literally about PEEP
# and FiO2 rises, which are two of this model's top contributors, so retrieval
# surfaces them constantly. The thresholds are shared PROVENANCE; the statistic
# is not -- the label reads a forward maximum where NHSN reads a two-day daily
# minimum, so "this reading meets the NHSN VAC definition" is false no matter
# how high the score is. Standing rule, recorded in .claude/rules/results.md.
_DEFINITION = re.compile(
    r"\b(?:meets?|met|satisf(?:y|ies|ied)|fulfil(?:s|led)?|qualifies as|consistent with"
    r"|constitutes?|diagnos(?:ed|tic) (?:of|with)|has|represents?)\s+"
    r"(?:the\s+|an?\s+|criteria\s+for\s+|a\s+case\s+of\s+)*"
    r"(?:NHSN\s+)?"
    r"(?:VAE|VAC|IVAC|PVAP|VAP\b"
    r"|ventilator[- ]associated\s+(?:event|condition|pneumonia|complication))",
    re.I)

# How many consecutive words shared with a retrieved passage count as quoting
# rather than coincidence. Below this, ordinary clinical phrasing collides:
# "assessment of PEEP and auto-PEEP" is five words that any correct sentence
# about this record might contain.
_QUOTE_RUN_WORDS = 7
# How much surrounding text must reproduce the passage before a number sitting
# in it counts as part of a quotation rather than as the generator's own figure.
_QUOTE_CONTEXT_WORDS = 5
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class Finding:
    code: str
    severity: str
    message: str
    evidence: str = ""

    def to_json(self) -> dict:
        return {"code": self.code, "severity": self.severity,
                "message": self.message, "evidence": self.evidence}


def _num(x):
    """Numeric or None. Categorical feature values are strings ('Pressure
    Support') and must not be coerced."""
    if isinstance(x, bool) or x is None:
        return None
    return float(x) if isinstance(x, (int, float)) else None


def allowed_numbers(record: dict, display: dict) -> set[float]:
    """Every quantity the record entitles a sentence to state.

    Re-derived here rather than taken from the payload -- see the module
    docstring. Percentage and time-unit forms are included because a readable
    sentence says "11.0%" and "4.2 h", not "0.1101" and "252".
    """
    out: set[float] = set()

    def add(v, *, pct=False, times=False):
        f = _num(v)
        if f is None:
            return
        out.add(f)
        if pct:
            out.add(f * 100)
        if times:
            out.add(f / 60)      # minutes -> hours
            out.add(f / 1440)    # minutes -> days

    b = record["band"]
    add(record["risk"]["calibrated"], pct=True)
    for k in ("observed_rate", "base_rate"):
        add(b[k], pct=True)
    add(b["lift"])
    for e in b.get("envelope") or ():
        add(e, pct=True)
    add(b.get("readings_in_state"))
    add(record["provenance"].get("horizon_hours"))
    add(record.get("documentation_share"), pct=True)
    add(record.get("imputed_share"), pct=True)
    add(record.get("attribution_age_min"), times=True)

    for p, v in record["telemetry"].items():
        # A population_reference value is a cohort statistic. It is NOT a number
        # this text is allowed to state, which is the whole reason the payload
        # withholds it -- so it does not enter the allowed set either.
        if v.get("source") != "population_reference":
            add(v.get("value"))
        add(v.get("age_min"), times=True)

    total = _num(record.get("attribution_total")) or 0.0
    imputed = {p for p, v in record["telemetry"].items()
               if v.get("source") == "population_reference"}
    for c in record["contributors"]:
        p = _param_of(c["feature"], display)
        # `spo2_final` for a population_reference parameter IS the cohort
        # default the telemetry block withholds. Admitting it here would let the
        # one number the payload deliberately hides be quoted and pass -- which
        # is precisely the failure this module exists to prevent.
        if not (p in imputed and _is_value_form(c["feature"], p)):
            add(c.get("value"))
        if total > 0:
            add(abs(c["contribution"]) / total, pct=True)
    add(len(record["contributors"]))
    return out


def _text_numbers(text: str) -> list[tuple[float, int, int, bool]]:
    """(value, decimals, position, is_percentage) for every number in the text.

    The percentage flag matters because a share and a value can be the same
    digits next to the same parameter name: "SpO2 (lowers the score, 5% of it)"
    is an attribution, "SpO2 5" would be a reading. An attached '%' is the one
    structural signal separating them.
    """
    out = []
    for m in _NUMBER.finditer(text):
        raw = m.group(1).replace(",", "")
        dec = len(raw.split(".")[1]) if "." in raw else 0
        try:
            out.append((float(raw), dec, m.start(),
                        text[m.end():m.end() + 1] == "%"))
        except ValueError:
            pass
    return out


def _matches(value: float, dec: int, allowed: set[float]) -> bool:
    """A stated number matches if any allowed quantity rounds to it at the
    precision the text chose to use."""
    return any(round(a, dec) == value for a in allowed)


def _sentence_at(text: str, pos: int) -> str:
    """The sentence containing `pos`.

    Both heuristic checks below ask "is this parameter being discussed as
    physiology, or as a fact about its charting?", and that frame is set by the
    sentence, not by a character radius. A disclosure listing ten unavailable
    parameters puts "No charted value" further than any fixed window from the
    entries in the middle of the list.
    """
    start, end = 0, len(text)
    for m in _SENT.finditer(text):
        if m.end() > pos:
            end = m.start() + 1
            break
        start = m.end()
    return text[start:end]


def _param_spans(text: str, display: dict) -> list[tuple[str, int, int]]:
    """(param, start, end) for each display label appearing in the text.

    Longest label first so 'I:E inspiratory part' is not shadowed by a shorter
    label that happens to be a substring of it.
    """
    spans = []
    for p, (label, _u, _d) in sorted(
            display.items(), key=lambda kv: -len(kv[1][0])):
        for m in re.finditer(re.escape(label), text, re.I):
            if not any(s <= m.start() < e for _p, s, e in spans):
                spans.append((p, m.start(), m.end()))
    return sorted(spans, key=lambda t: t[1])


def _word_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0).lower(), m.start(), m.end()) for m in _WORD.finditer(text)]


def quoted_spans(text: str, evidence) -> list[dict]:
    """Character ranges of `text` that reproduce a retrieved passage.

    A run of at least _QUOTE_RUN_WORDS consecutive words shared with a passage.
    Nothing fuzzy: the words must match in order, so the model cannot earn quote
    status by paraphrasing, and the exemptions below cannot be talked into.
    """
    out: list[dict] = []
    tw = _word_spans(text)
    for ev in evidence or ():
        ew = [w for w, _s, _e in _word_spans(ev.get("text", ""))]
        if len(ew) < _QUOTE_RUN_WORDS:
            continue
        # every start position in the passage, by first word, for a linear scan
        index: dict[str, list[int]] = {}
        for j, w in enumerate(ew):
            index.setdefault(w, []).append(j)
        i = 0
        while i < len(tw):
            best = 0
            for j in index.get(tw[i][0], ()):
                n = 0
                while (i + n < len(tw) and j + n < len(ew)
                       and tw[i + n][0] == ew[j + n]):
                    n += 1
                best = max(best, n)
            if best >= _QUOTE_RUN_WORDS:
                out.append({"start": tw[i][1], "end": tw[i + best - 1][2],
                            "words": best, "evidence": ev})
                i += best
            else:
                i += 1
    return out


def _in_quote(pos: int, spans: list[dict]) -> bool:
    return any(s["start"] <= pos < s["end"] for s in spans)


def check(record: dict, text: str, *, display: dict,
          band_names: tuple[str, ...], evidence=()) -> list[Finding]:
    """Every decidable disagreement between `text` and `record`.

    `evidence` is the retrieved passages attached to this reading, passed AS
    DATA exactly like `display` and `band_names` -- this module still imports
    nothing from explain.py, and still re-derives everything from the record.

    WHAT EVIDENCE CHANGES, AND WHAT IT DELIBERATELY DOES NOT
    -------------------------------------------------------
    Inside a span that reproduces a retrieved passage verbatim, the generator is
    a CONDUIT rather than an author: "PEEP 5 to 15 cm H2O" published by the AARC
    is not a claim about this patient's PEEP, and checks 1, 2, 5, 6 and 7 would
    each raise a false accusation against it. Those checks stand down inside such
    a span and nowhere else.

    `allowed_numbers` is NOT widened, and that is the load-bearing decision.
    Line 293 below tests `not _matches(v, dec, own) and not _matches(v, dec,
    allowed)` -- one set feeding two checks -- so admitting 5 globally as a
    guideline PEEP threshold would simultaneously make "SpO2 5" legal. Scoping
    the exemption to located spans instead keeps every number the model writes in
    its OWN prose subject to the original rule, whether or not some guideline
    happens to contain it.

    The exemption cannot be gamed, because the span has to match published text
    word for word to exist at all.
    """
    f: list[Finding] = []
    if not text or not text.strip():
        return [Finding("EMPTY_TEXT", VIOLATION, "no explanation text")]

    allowed = allowed_numbers(record, display)
    nums = _text_numbers(text)
    quotes = quoted_spans(text, evidence)

    # 1 -- numbers that are in no field of the record.
    for v, dec, pos, _pct in nums:
        if _in_quote(pos, quotes):
            continue
        if not _matches(v, dec, allowed):
            f.append(Finding(
                FABRICATED_NUMBER, VIOLATION,
                f"{v:g} appears in no field of this record",
                text[max(0, pos - 40):pos + 20].strip()))

    # 2 -- a number bound to a parameter must be that parameter's number.
    # Bound = the first number after the label and before the next label, within
    # a window. Conservative on purpose: a missed binding is a missed check, a
    # wrong binding is a false accusation.
    # A parameter NAME inside a verbatim quotation is the guideline's word, not
    # the generator binding a value to this patient. Dropped from `spans` here
    # rather than skipped in each of checks 2, 5 and 6 -- one exclusion, three
    # checks, no chance of them disagreeing about what counts as quoted.
    spans = [s for s in _param_spans(text, display) if not _in_quote(s[1], quotes)]
    tel = record["telemetry"]
    for i, (p, _s, e) in enumerate(spans):
        nxt = spans[i + 1][1] if i + 1 < len(spans) else len(text)
        window = min(nxt, e + 40)
        near = [(v, dec, pos, pct) for v, dec, pos, pct in nums if e <= pos < window]
        if not near:
            continue
        v, dec, pos, is_pct = near[0]
        entry = tel.get(p, {})
        label = display[p][0]
        if entry.get("source") == "population_reference":
            # The record holds no value for this patient, so a number attached
            # to the name would be a cohort statistic stated as an observation.
            #
            # Numbers do legitimately follow a parameter name without being its
            # value: "SpO2 (lowers the score, 5% of it)" is a share, and an
            # explanation that withholds the value correctly puts the share
            # first. So the test is not "is this number in the record" -- the
            # withheld cohort default can collide with a real quantity
            # elsewhere -- but "is this number THE WITHHELD ONE", which is
            # exact and known.
            withheld = _num(entry.get("value"))
            if withheld is not None and not is_pct and _matches(v, dec, {withheld}):
                f.append(Finding(
                    IMPUTED_QUOTED, VIOLATION,
                    f"{label} has no charted value for this patient; {v:g} is "
                    f"the cohort default the model substituted",
                    text[_s:min(len(text), window)].strip()))
            elif not _matches(v, dec, allowed):
                f.append(Finding(
                    IMPUTED_QUOTED, VIOLATION,
                    f"{label} has no charted value for this patient, but the "
                    f"text states {v:g}",
                    text[_s:min(len(text), window)].strip()))
            continue
        own = set()
        for key, times in (("value", False), ("age_min", True)):
            n = _num(entry.get(key))
            if n is None:
                continue
            own.add(n)
            if times:
                own.update((n / 60, n / 1440))
        # Shares and rates legitimately follow a parameter name too
        # ("PEEP 5 cmH2O (raises the score, 21% of it)"), so a number that is
        # allowed globally is not evidence of a mis-binding.
        if own and not _matches(v, dec, own) and not _matches(v, dec, allowed):
            # Report the VALUE, not min(own): `own` also holds the age, and a
            # freshly measured parameter has age 0, so the old message read
            # "SpO2 is 0 in the record" for a parameter that was 97%.
            actual = _num(entry.get("value"))
            f.append(Finding(
                PARAM_VALUE_MISMATCH, VIOLATION,
                (f"{label} is {actual:g} in the record, text states {v:g}"
                 if actual is not None
                 else f"{label} has no value in the record, text states {v:g}"),
                text[_s:min(len(text), window)].strip()))

    # 3 -- the band named must be the band displayed.
    shown = record["band"]["displayed"]
    for name in band_names:
        if name == shown:
            continue
        if re.search(rf"\b{name}\b", text):          # ALL CAPS -- unambiguous
            f.append(Finding(BAND_MISMATCH, VIOLATION,
                             f"text names {name}; the record's band is {shown}",
                             name))
        elif re.search(rf"\b{name}\b", text, re.I):  # 'high FiO2' is not a band
            f.append(Finding(POSSIBLE_BAND_MISMATCH, WARNING,
                             f"text contains '{name.lower()}' and the band is "
                             f"{shown}; may or may not be naming the band",
                             name.lower()))

    # 4 -- trajectory claims, which no feature in this model can support.
    for m in _TREND.finditer(text):
        f.append(Finding(TREND_CLAIM, VIOLATION,
                         f"'{m.group(0)}' is a claim about change over time; "
                         f"the model has no trend inputs",
                         text[max(0, m.start() - 40):m.end() + 20].strip()))
    if not _TREND.search(text):
        for m in _TREND_SOFT.finditer(text):
            f.append(Finding(POSSIBLE_TREND_CLAIM, WARNING,
                             f"'{m.group(0)}' may be describing change over "
                             f"time rather than attribution direction",
                             text[max(0, m.start() - 40):m.end() + 20].strip()))

    # 5 -- a documentation feature narrated as physiology.
    doc_params = {p for c in record["contributors"] if c["kind"] == "documentation"
                  for p in [_param_of(c["feature"], display)] if p}
    phys_params = {p for c in record["contributors"] if c["kind"] == "physiology"
                   for p in [_param_of(c["feature"], display)] if p}
    for p, s, e in spans:
        if p in doc_params and p not in phys_params:
            sent = _sentence_at(text, s)
            if not _CHARTING.search(sent):
                f.append(Finding(
                    DOC_AS_PHYSIOLOGY, WARNING,
                    f"{display[p][0]} enters this score only as a documentation "
                    f"feature, but is mentioned without any charting framing",
                    sent.strip()))

    # 6 -- a driver the model did not use.
    used = {p for c in record["contributors"]
            for p in [_param_of(c["feature"], display)] if p}
    for p, s, e in spans:
        sent = _sentence_at(text, s)
        if p not in used and not _CHARTING.search(sent):
            f.append(Finding(
                UNSUPPORTED_DRIVER, WARNING,
                f"{display[p][0]} is named but is not among this reading's "
                f"contributors",
                sent.strip()))

    # 7 -- treatment language. Reported, never blocked: recommendations are a
    # separate black-box concern that does not influence the assessment, so this
    # exists to make their presence countable rather than to prevent them.
    # Exempt inside a quotation: a graded recommendation says "assess" and
    # "wean" because that is what a recommendation is.
    for m in _TREATMENT.finditer(text):
        if _in_quote(m.start(), quotes):
            continue
        f.append(Finding(TREATMENT_LANGUAGE, WARNING,
                         f"'{m.group(0)}' is intervention language",
                         text[max(0, m.start() - 40):m.end() + 20].strip()))

    # 8 -- a quotation that is not quite the passage.
    #
    # THE failure mode of a quote-only design, and the reason it needs its own
    # code. A fabricated number is caught by check 1 -- but only while it stays
    # fabricated. Drift a threshold onto a value the record DOES contain, inside
    # otherwise faithful guideline prose, and every existing check passes: the
    # number is allowed, the binding is consistent, and the sentence now carries
    # a citation asserting a professional society published it.
    #
    # Detected by POSITIONAL ALIGNMENT rather than by sentence overlap. A
    # drifted number breaks the word run by definition -- that is what makes it
    # drift -- so it never sits inside one of the spans above. A sentence-level
    # overlap test was the first attempt and it has a structural hole: the
    # sentence splitter cuts on ". ", so drifting a list enumerator turns
    # "2. Pulse oximetry should ..." into the fragment "... states: 9." whose
    # overlap with the passage is near zero, and 92 of 124 mutations walked
    # straight through it.
    #
    # Aligning the WORDS AROUND each number has no such hole: if the context
    # either side of a number reproduces the passage but the number does not,
    # the number was altered, wherever it sits and whatever the punctuation
    # around it. This is ClinicBot's "numeric thresholds must match retrieved
    # values exactly via string matching", generalised from table cells to prose.
    for ev in evidence or ():
        src = ev.get("text", "")
        if not src:
            continue
        ew = [w for w, _s, _e in _word_spans(src)]
        src_num_at = {j: w for j, w in enumerate(ew) if _NUMBER.fullmatch(w)}
        if not src_num_at:
            continue
        tw = _word_spans(text)
        for i, (w, s, e) in enumerate(tw):
            if not _NUMBER.fullmatch(w):
                continue
            for j, sw in src_num_at.items():
                if sw == w:
                    continue                      # faithfully reproduced
                left = right = 0
                while (i - left - 1 >= 0 and j - left - 1 >= 0
                       and tw[i - left - 1][0] == ew[j - left - 1]):
                    left += 1
                while (i + right + 1 < len(tw) and j + right + 1 < len(ew)
                       and tw[i + right + 1][0] == ew[j + right + 1]):
                    right += 1
                if left + right < _QUOTE_CONTEXT_WORDS:
                    continue
                f.append(Finding(
                    ALTERED_QUOTE, VIOLATION,
                    f"{w} sits in {left + right} words reproducing "
                    f"{ev.get('citation', 'a retrieved passage')} where that "
                    f"passage reads {sw}",
                    text[max(0, s - 60):min(len(text), e + 60)].strip()))
                break

    # 9 -- a quotation with nothing saying where it came from.
    for q in quotes:
        cite = q["evidence"].get("citation", "")
        window = text[max(0, q["start"] - 200):min(len(text), q["end"] + 200)]
        if cite and not _cite_present(window, cite):
            f.append(Finding(
                UNCITED_QUOTE, VIOLATION,
                f"{q['words']} consecutive words reproduce a retrieved passage "
                f"with no attribution nearby; it should cite {cite}",
                text[q["start"]:q["end"]].strip()))

    # 10 -- claiming the reading meets a surveillance definition.
    for m in _DEFINITION.finditer(text):
        f.append(Finding(
            DEFINITION_CONFLATION, VIOLATION,
            f"'{m.group(0).strip()}' asserts this reading meets a surveillance "
            f"definition; NHSN supplied the thresholds only, and the label "
            f"inverts the statistic (forward maximum vs two-day daily minimum)",
            text[max(0, m.start() - 40):m.end() + 30].strip()))
    return f


def _cite_present(window: str, citation: str) -> bool:
    """Is this citation acknowledged nearby?

    Matched on the issuing body rather than on the full string: a clinician
    reading "the AARC guideline recommends" knows the source, and demanding
    "AARC CPG, Respir Care 2024;69(8):1042-1054" verbatim inside three sentences
    of prose would fail every correct attribution.
    """
    toks = [t for t in _WORD.findall(citation) if len(t) > 3 and not t.isdigit()]
    return any(re.search(rf"\b{re.escape(t)}\b", window, re.I) for t in toks[:3])


def _is_value_form(feature: str, param: str | None) -> bool:
    """Does this feature carry the parameter's VALUE, rather than describe its
    charting? `spo2_observed = 0` is true whether or not the patient was ever
    measured; `spo2_final` is the number itself.

    Re-derived here rather than imported from explain.py -- if both modules were
    wrong in the same way, the check would agree with the generator and catch
    nothing. That independence is the whole point of this file.
    """
    if param is None:
        return False
    tail = feature[len(param):]
    return tail in ("_final", "_locf")


def _param_of(feature: str, display: dict) -> str | None:
    """Frozen parameter behind a feature name, or None.

    Independent of explain.split_feature by design -- if both were wrong in the
    same way the check would agree with the generator and catch nothing.
    """
    for p in display:
        if feature == p or feature.startswith(p + "_"):
            return p
    return None


def worst(findings: list[Finding]) -> str | None:
    if any(x.severity == VIOLATION for x in findings):
        return VIOLATION
    return WARNING if findings else None
