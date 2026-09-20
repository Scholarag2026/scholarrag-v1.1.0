"""Build the blind annotation package for the HSS claim-verification set.

The sheet must be opaque and blind: item ids must not spell out their own construction
category, the rubric must not map categories to labels, wrong-paper items must not display the
true source sentence as the "cited paper" passage, and no_full_text items must not point at
cached text that the label claims does not exist.

This script builds an opaque, blind sheet from scratch:

- ids are shuffled to ann-01..ann-30 with a fixed seed, and the item_id/category/expected-label
  mapping is written ONLY to hss_annotation_key_v2.csv, never to the sheet, the index or the
  instructions;
- for every item that carries a cited DOI, the passage shown is independently retrieved by
  scoring 1-3 sentence windows of the cited paper's own cached text against the claim (Jaccard
  overlap on lowercased alphabetic tokens minus stopwords) and keeping the best window if its
  score clears a threshold, so the passage is not read off the construction record;
  the withheld-evidence items get an explicit "Evidence withheld" marker instead of a path;
- the wrong_paper items are treated by this same procedure against the DOI they are actually
  attributed to in the sheet (chunk_doi), not the DOI the sentence really came from, so their
  passage is either a genuine (non-)match against the cited paper or the explicit no-match
  marker, never the claim sentence itself.

Inputs (read only):
  evaluation/claims/hss_claims.jsonl        one JSON object per item: item_id, rule, expected,
                                             claim, original_sentence, alteration, source_doi,
                                             chunk_doi, chunk_index, title, authors, built_at
  evaluation/claims/hss_sources.json        list of {doi, title, year, journal, ...}
  evaluation/claims/data/hss_fulltext/*.json  {doi, title, authors, source_url, fetched_at,
                                             n_chars, chunks: [{section, text}, ...]}
                                             filename = doi with '.' -> '-', '/' -> '_', + '.json'

Outputs (written under evaluation/claims/annotation/):
  hss_annotation_key_v2.csv        ann_id, item_id, construction_category, expected_label, built_at
  hss_annotation_sheet_v2.csv      the blind sheet (UTF-8 BOM)
  source_texts_index_v2.json       {ann_id: {cited_paper_doi, paths: [...]}}
  INSTRUCTIONS_v2.md               rubric for the annotator session, no category hints
  annotator2_A.csv, annotator2_B.csv, annotator2_C.csv   identical copies of the blind sheet

Run with: python build_sheet_v2.py
"""

from __future__ import annotations

import csv
import json
import os
import random
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLAIMS_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))  # evaluation/claims
CLAIMS_JSONL = os.path.join(CLAIMS_DIR, "hss_claims.jsonl")
SOURCES_JSON = os.path.join(CLAIMS_DIR, "hss_sources.json")
FULLTEXT_DIR = os.path.join(CLAIMS_DIR, "data", "hss_fulltext")
FULLTEXT_REL_DIR = "evaluation/claims/data/hss_fulltext"  # for the index file, repo-relative

SEED = 20260905 * 7
MATCH_THRESHOLD = 0.15
MAX_PASSAGE_CHARS = 700
NO_MATCH_MARKER = "No matching passage found in the cited paper (best overlap below threshold)"
WITHHELD_MARKER = "Evidence withheld: no passage or paper text is available for this item"

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with", "as",
    "is", "are", "was", "were", "be", "been", "being", "that", "this", "these", "those", "it",
    "its", "by", "from", "which", "who", "whom", "their", "they", "he", "she", "his", "her",
    "we", "our", "you", "your", "i", "not", "no", "do", "does", "did", "has", "have", "had",
    "will", "would", "can", "could", "should", "may", "might", "must", "than", "then", "so",
    "such", "also", "into", "about", "over", "under", "between", "both", "each", "more",
    "most", "other", "some", "any", "all", "if", "because", "while", "when", "where", "what",
    "how", "there", "here", "them", "us", "one", "two", "s", "t",
}

TOKEN_RE = re.compile(r"[a-zA-Z]+")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
WS_RE = re.compile(r"\s+")


def tokenize(text):
    toks = TOKEN_RE.findall(text.lower())
    return {t for t in toks if t not in STOPWORDS and len(t) > 1}


def load_claims():
    items = []
    with open(CLAIMS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_sources():
    with open(SOURCES_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {r["doi"]: r for r in records}


def doi_to_filename(doi):
    return doi.replace(".", "-").replace("/", "_") + ".json"


_FULLTEXT_CACHE = {}


def load_fulltext_sentences(doi):
    """Return the list of (roughly) sentence strings for a cited paper's cached text."""
    if doi in _FULLTEXT_CACHE:
        return _FULLTEXT_CACHE[doi]
    path = os.path.join(FULLTEXT_DIR, doi_to_filename(doi))
    if not os.path.isfile(path):
        _FULLTEXT_CACHE[doi] = []
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    parts = [chunk.get("text", "") for chunk in data.get("chunks", [])]
    full_text = " ".join(parts)
    full_text = WS_RE.sub(" ", full_text).strip()
    sentences = [s.strip() for s in SENT_SPLIT_RE.split(full_text) if s.strip()]
    _FULLTEXT_CACHE[doi] = sentences
    return sentences


def best_passage(claim_text, doi):
    """Score 1-3 sentence windows of the cited paper's cached text against the claim.

    Returns (passage_text_or_marker, best_score).
    """
    sentences = load_fulltext_sentences(doi)
    if not sentences:
        return NO_MATCH_MARKER, 0.0

    claim_tokens = tokenize(claim_text)
    if not claim_tokens:
        return NO_MATCH_MARKER, 0.0

    best_score = -1.0
    best_window_text = None
    n = len(sentences)
    for window_size in (1, 2, 3):
        for start in range(0, n - window_size + 1):
            window = sentences[start:start + window_size]
            window_text = " ".join(window)
            window_tokens = tokenize(window_text)
            if not window_tokens:
                continue
            union = claim_tokens | window_tokens
            if not union:
                continue
            overlap = len(claim_tokens & window_tokens) / len(union)
            if overlap > best_score:
                best_score = overlap
                best_window_text = window_text

    if best_window_text is None or best_score < MATCH_THRESHOLD:
        return NO_MATCH_MARKER, max(best_score, 0.0)

    passage = best_window_text
    if len(passage) > MAX_PASSAGE_CHARS:
        cut = passage[:MAX_PASSAGE_CHARS]
        last_space = cut.rfind(" ")
        if last_space > 0:
            cut = cut[:last_space]
        passage = cut.rstrip() + " ..."
    return passage, best_score


def surname(full_name):
    parts = full_name.strip().split()
    return parts[-1] if parts else full_name.strip()


def first_author_year(authors, year):
    surnames = [surname(a) for a in authors] if authors else []
    if not surnames:
        base = "Unknown"
    elif len(surnames) == 1:
        base = surnames[0]
    elif len(surnames) == 2:
        base = f"{surnames[0]} & {surnames[1]}"
    else:
        base = f"{surnames[0]} et al."
    year_str = str(year) if year is not None else "n.d."
    return f"{base}, {year_str}"


def main():
    items = load_claims()
    assert len(items) == 30, f"expected 30 claim items, found {len(items)}"
    sources = load_sources()

    item_ids = [it["item_id"] for it in items]
    assert len(set(item_ids)) == 30, "item_id values are not unique"

    shuffled = list(item_ids)
    random.Random(SEED).shuffle(shuffled)
    ann_id_by_item = {item_id: f"ann-{i+1:02d}" for i, item_id in enumerate(shuffled)}

    by_item_id = {it["item_id"]: it for it in items}

    key_rows = []
    sheet_rows = []
    index_map = {}

    n_passage = 0
    n_no_match = 0
    n_withheld = 0

    for item_id, ann_id in ann_id_by_item.items():
        item = by_item_id[item_id]
        chunk_doi = item.get("chunk_doi")
        source_doi = item.get("source_doi")
        cited_doi = chunk_doi if chunk_doi else source_doi
        withheld = chunk_doi is None

        key_rows.append({
            "ann_id": ann_id,
            "item_id": item_id,
            "construction_category": item["rule"],
            "expected_label": "|".join(item["expected"]),
            "built_at": item.get("built_at", ""),
        })

        source_rec = sources.get(cited_doi, {})
        year = source_rec.get("year")
        fay = first_author_year(item.get("authors") or [], year)

        if withheld:
            passage = WITHHELD_MARKER
            n_withheld += 1
            paths = []
        else:
            passage, score = best_passage(item["claim"], cited_doi)
            if passage == NO_MATCH_MARKER:
                n_no_match += 1
            else:
                n_passage += 1
            paths = [f"{FULLTEXT_REL_DIR}/{doi_to_filename(cited_doi)}"]

        sheet_rows.append({
            "ann_id": ann_id,
            "claim_text": item["claim"],
            "cited_paper_title": item.get("title", ""),
            "cited_paper_doi": cited_doi,
            "cited_paper_first_author_year": fay,
            "cited_paper_passage": passage,
            "annotator_label": "",
            "annotator_confidence": "",
            "annotator_note": "",
            "annotator_name": "",
            "annotated_on": "",
        })

        index_map[ann_id] = {"cited_paper_doi": cited_doi, "paths": paths}

    # Sort output rows by ann_id (ann-01..ann-30) so the shuffle only governs the id
    # assignment, not a visible ordering artifact.
    key_rows.sort(key=lambda r: int(r["ann_id"].split("-")[1]))
    sheet_rows.sort(key=lambda r: int(r["ann_id"].split("-")[1]))

    # --- write key (withheld mapping) ---
    key_path = os.path.join(BASE_DIR, "hss_annotation_key_v2.csv")
    with open(key_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "ann_id", "item_id", "construction_category", "expected_label", "built_at"
            ],
        )
        w.writeheader()
        w.writerows(key_rows)

    # --- write blind sheet (UTF-8 BOM) ---
    sheet_fields = [
        "ann_id", "claim_text", "cited_paper_title", "cited_paper_doi",
        "cited_paper_first_author_year", "cited_paper_passage", "annotator_label",
        "annotator_confidence", "annotator_note", "annotator_name", "annotated_on",
    ]

    def write_sheet(path):
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sheet_fields)
            w.writeheader()
            w.writerows(sheet_rows)

    sheet_path = os.path.join(BASE_DIR, "hss_annotation_sheet_v2.csv")
    write_sheet(sheet_path)
    for label in ("A", "B", "C"):
        write_sheet(os.path.join(BASE_DIR, f"annotator2_{label}.csv"))

    # --- write source texts index ---
    index_path = os.path.join(BASE_DIR, "source_texts_index_v2.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_map, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    # --- write instructions ---
    instructions_path = os.path.join(BASE_DIR, "INSTRUCTIONS_v2.md")
    write_instructions(instructions_path)

    print("items:", len(sheet_rows))
    print("with a passage shown:", n_passage)
    print("with the no-match marker:", n_no_match)
    print("withheld:", n_withheld)


def write_instructions(path):
    text = """# Annotation instructions (round 2)

## Purpose

This is an independent check of the labels used in a claim-verification evaluation. Each row in
hss_annotation_sheet_v2.csv (or your copy annotator2_A.csv / annotator2_B.csv /
annotator2_C.csv) shows a claim sentence, the paper it is cited to, and a passage that was
retrieved from that paper's text. Your job is to read the claim and the passage and assign one
label per row. You are one of three independent annotation sessions checking the same 30 items;
do not open the other two sessions' files, and do not open any results, summary, key or
provenance file in this folder.

## Labels

Use exactly one of these four labels for every row, written in the annotator_label column:

- verified: the cited paper's text supports the claim as stated.
- needs_nuance: the paper supports only a weaker, narrower or conditional version; the claim
  overstates, generalises or drops a qualifier.
- unsupported: the cited paper contradicts the claim, or does not address it, so the citation
  does not support it.
- no_full_text: the item shows the marker "Evidence withheld" and no passage or paper text is
  available for it; apply this label in that case regardless of how plausible the claim sounds.

## Rule

Judge only from the cited_paper_passage column and, when a path is listed for that row in
source_texts_index_v2.json, the cited paper's own cached text at that path. If the passage says
"No matching passage found", open the cited paper's text and search for the claim's key terms
before deciding. Do not use prior knowledge about the claim's subject matter. Do not open any
file that is not listed for the row you are judging.

## What to fill in

For each row, fill in:

- annotator_label: one of the four labels above.
- annotator_confidence: high, medium, or low.
- annotator_note: a short note (a sentence or two) explaining the decision, ideally quoting the
  part of the passage or paper text that drove it.
- annotator_name: your session identifier.
- annotated_on: the date you completed the row.

Leave every other column exactly as given; do not edit ann_id, claim_text,
cited_paper_title, cited_paper_doi, cited_paper_first_author_year or
cited_paper_passage.

## Time estimate

Working through all 30 rows carefully, including opening the cited paper's text where the rule
above requires it, should take about 1 to 2 hours.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
