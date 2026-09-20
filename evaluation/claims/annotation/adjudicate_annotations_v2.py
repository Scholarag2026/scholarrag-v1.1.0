"""Adjudicate the round-2 blind triple annotation of the HSS claim-verification set.

Reads ``hss_annotation_completed_v2.csv`` (the merge of annotator2_A/B/C.csv produced by
``aggregate_annotations_v2.py``) and writes ``hss_annotation_adjudicated_v2.csv`` with one
final label per item.

Adjudication trigger: an item is adjudicated when the three sessions are not unanimous, or
when the majority label falls outside the expected-label set recorded in the construction key.
Two items met the trigger (ann-22 and ann-23, both non-unanimous). Three further
items whose annotator notes misdescribe the cited paper (ann-04, ann-20, ann-30) were re-read
against the cached paper text as well; the unanimous label survived in all three, so their
``source`` stays ``unanimous`` and the correction is recorded in the ``rationale`` column.

Every final label below was decided by reading the cited paper's cached text in
``evaluation/claims/data/hss_fulltext/``; the decisive sentence is quoted in the rationale.
The paper text is the authority, not the construction key.

Run with: python adjudicate_annotations_v2.py
"""

from __future__ import annotations

import csv
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
COMPLETED = os.path.join(HERE, "hss_annotation_completed_v2.csv")
ADJUDICATED = os.path.join(HERE, "hss_annotation_adjudicated_v2.csv")

OUT_COLUMNS = [
    "ann_id",
    "item_id",
    "final_label",
    "source",
    "rationale",
    "evidence_mode",
    "construction_category",
    "expected_label",
    "agrees_with_construction",
]

# Items the adjudicator re-decided against the paper text after a non-unanimous vote.
ADJUDICATED_LABEL = {
    "ann-22": "unsupported",
    "ann-23": "unsupported",
}

WITHHELD_RATIONALE = (
    'The item shows the marker "Evidence withheld: no passage or paper text is available for '
    'this item", so no passage and no paper text were presented for it. All three sessions '
    "abstained, which is the response the rubric requires whenever evidence is withheld, however "
    "plausible the claim sounds."
)

RATIONALE = {
    "ann-01": WITHHELD_RATIONALE,
    "ann-02": WITHHELD_RATIONALE,
    "ann-16": WITHHELD_RATIONALE,
    "ann-18": WITHHELD_RATIONALE,
    "ann-26": WITHHELD_RATIONALE,
    "ann-03": (
        'Kim and Weng state "Second most common strategy is classroom observation (n=17), '
        'followed by artifacts/documents ... (n=7), and focus group discussion (n=6)", which '
        "matches every figure in the claim. The only change is students to learners, so the "
        "paper supports the claim as stated."
    ),
    "ann-04": (
        "Shirvan and Talebzadeh 2020, on the signature dynamics of FLCA and FLE, contains no "
        "occurrence of PsyCap, psychological capital, Masten, optimism or persistence, and its "
        'single hit for "Resilien" is the reference-list title "Resilient, overcontrolled, and '
        'undercontrolled boys: Three replicable personality types". The paper does not address '
        "psychological capital at all, so the citation cannot support the claim; the notes by A "
        "and C, which report zero hits for resilience, are inaccurate as written, though the "
        "conclusion they draw stands."
    ),
    "ann-05": (
        'The paper reads "There are students who simply do not know where their instructors '
        'stand (14%)" and gives the same 12% figure for ChatGPT use not having been discussed in '
        "class. Only students becomes learners, so the paper supports the claim as stated."
    ),
    "ann-06": (
        "The paper reports that \"individuals' who had higher levels of PsyCap had more levels of "
        'ideal L2-self and language learning experience", whereas the claim asserts lower levels '
        "of PsyCap. The direction of the association is reversed, so the cited paper contradicts "
        "the claim."
    ),
    "ann-07": (
        "The paper states that attractor states arise where \"the systems self-organize into more "
        'preferable states (Hiver, 2015; Juarrero, 1999)", whereas the claim says less preferable '
        "states. The polarity is reversed, so the cited paper contradicts the claim."
    ),
    "ann-08": (
        "The cited review by Kim and Weng contains no occurrence of enjoyment, anxiety, gesture, "
        "movement or nonverbal, so it says nothing about body movements signalling emotion. The "
        "cited paper does not address the claim, so the citation does not support it."
    ),
    "ann-09": (
        'The paper reads "can improve the overall quality of L2 writing in a future study", with '
        "the claim substituting investigation for study. The recommendation to use a holistic "
        "scale to evaluate intelligent writing assistants such as Grammarly is stated in the "
        "paper as the claim states it."
    ),
    "ann-10": (
        'The paper contains the sentence "Japanese socio-cultural environments can be restrictive '
        "or even dismissive of NESTs, resulting in ambivalence in their beliefs about their "
        'professional identities and competencies" verbatim. The claim reproduces that sentence, '
        "so the paper supports it as stated."
    ),
    "ann-11": (
        'The paper reads "yet it might make our work more interesting, engaging, and innovative", '
        "whereas the claim renders this as less interesting, engaging, and innovative. The "
        "valence is reversed, so the cited paper contradicts the claim."
    ),
    "ann-12": (
        'The paper reports as its MANOVA result that "students who have higher PsyCap are more '
        'willing to communicate in English in comparison to students with lower levels of PsyCap". '
        "Only students becomes learners, so the paper supports the claim as stated."
    ),
    "ann-13": (
        "The cited paper, on PsyCap and willingness to communicate, contains no occurrence of "
        '"predictive text", "corrective feedback", "Frankenberg" or even the stem "technolog". '
        "The cited paper does not address the claim, so the citation does not support it."
    ),
    "ann-14": (
        'The paper recommends Grammarly "as a means to increase lexical variation in their '
        'writing", whereas the claim says decrease lexical variation. The reversal also '
        "contradicts the claim's own clause that lexical richness was enhanced, so the cited "
        "paper contradicts the claim."
    ),
    "ann-15": (
        "The paper contains \"Empirical research is also needed to investigate students' "
        "perceptions of AI (Chan & Hu, 2023) and effective practices and models of AI "
        'implementation in English language teaching" verbatim. The claim reproduces that '
        "sentence, so the paper supports it as stated."
    ),
    "ann-17": (
        'The paper contains "A look at the distribution shows a skew towards higher values for '
        'FLE and lower values for FLCA" verbatim. The claim reproduces that sentence, so the '
        "paper supports it as stated."
    ),
    "ann-19": (
        'The paper reads "the narratives of one best fit candidate for each archetype are '
        'presented in the results", with the claim substituting findings for results. That '
        "substitution renames a section and changes no reported fact, so the paper supports the "
        "claim as stated."
    ),
    "ann-20": (
        'The paper contains the decisive sentence verbatim: "In accordance with this approach, '
        "the L2 Learning Experience can be defined as the perceived quality of the learners' "
        'engagement with various aspects of the language learning process." Annotator B\'s note, '
        'that the paper reads "may be defined" and that the claim\'s "can" merely matches the '
        "paper's \"may\", misreads the source, because the abstract carries \"may be defined\" "
        'while the body carries the exact "can be defined" sentence the claim copies; the label '
        "verified is unchanged."
    ),
    "ann-21": (
        'The paper reports that "FLE was significantly higher with the Main Teacher", whereas the '
        "claim says significantly lower. The direction is reversed, so the cited paper "
        "contradicts the claim."
    ),
    "ann-22": (
        'The paper states "The results revealed an effect size of 0.81 which can be interpreted '
        "as a large effect of the independent variable (CL) on the dependent variable (speaking "
        'skill)", and the string 1.62 does not occur anywhere in its cached text. The claim '
        "reports a different value for the same statistic inside the same sentence frame, which "
        "is a contradiction of the reported number rather than an overstated or unqualified "
        "version of it, so unsupported is correct and annotator C's needs_nuance understates the "
        "error."
    ),
    "ann-23": (
        'The paper states "Seventeen participants (18%) reported using it to write the whole '
        'assignment for them, which is certainly a cause for concern", and 36.1% is the paper\'s '
        'separate figure for "writing a part of the assignment". The claim attaches the paper\'s '
        "count of seventeen to a percentage the paper assigns to a different and incompatible "
        "category, so the cited paper contradicts the claim and unsupported is correct rather "
        "than needs_nuance."
    ),
    "ann-24": (
        'The paper reads "Most studies used one-on-one oral interviews with their participants", '
        "whereas the claim says Few studies. The quantifier is reversed, so the cited paper "
        "contradicts the claim."
    ),
    "ann-25": (
        "The cited Grammarly freewriting study contains no occurrence of ChatGPT, Seventeen or "
        '"entire assignment". The cited paper does not address the claim, so the citation does '
        "not support it."
    ),
    "ann-27": (
        'The paper reports that the comparison "showed a significant difference between the '
        'results of the experimental and control groups", whereas the claim inserts no before '
        "significant difference. The finding is negated, so the cited paper contradicts the claim."
    ),
    "ann-28": (
        'The paper states that "NESTs need to expend significant cognitive labor on their '
        'socio-cultural and linguistic tasks compared to their NNEST colleagues", whereas the '
        "claim says no significant cognitive labor. The finding is negated, so the cited paper "
        "contradicts the claim."
    ),
    "ann-29": (
        'The paper contains "both t-tests and effect sizes showed no significant improvements for '
        'both the experimental group and the control group" verbatim. The claim reproduces that '
        "sentence, so the paper supports it as stated."
    ),
    "ann-30": (
        "The cited trioethnography describes only its own procedure, that the three authors' "
        'discussions "were recorded and transcribed in full", and it contains no occurrence of '
        '"translated into English" and no survey of interview practice across EFL studies. '
        "Annotator B's note that the paper contains no instance of transcription is inaccurate, "
        "since transcribed does occur, but the paper still does not address the claim, so "
        "unsupported stands."
    ),
}


def main():
    with open(COMPLETED, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 30:
        raise SystemExit("expected 30 rows, found %d" % len(rows))

    out_rows = []
    for row in rows:
        ann_id = row["ann_id"]
        unanimous = row["unanimous"] == "True"
        if ann_id in ADJUDICATED_LABEL:
            final_label = ADJUDICATED_LABEL[ann_id]
            source = "adjudicated"
        elif unanimous:
            final_label = row["majority_label"]
            source = "unanimous"
        else:
            final_label = row["majority_label"]
            source = "majority"
        expected = set(row["expected_label"].split("|"))
        out_rows.append(
            {
                "ann_id": ann_id,
                "item_id": row["item_id"],
                "final_label": final_label,
                "source": source,
                "rationale": RATIONALE[ann_id],
                "evidence_mode": row["evidence_mode"],
                "construction_category": row["construction_category"],
                "expected_label": row["expected_label"],
                "agrees_with_construction": str(final_label in expected),
            }
        )

    with open(ADJUDICATED, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)

    labels = Counter(r["final_label"] for r in out_rows)
    sources = Counter(r["source"] for r in out_rows)
    agree = sum(1 for r in out_rows if r["agrees_with_construction"] == "True")
    print("rows:", len(out_rows))
    print("final labels:", dict(labels))
    print("sources:", dict(sources))
    print("final vs construction: %d/%d = %.3f" % (agree, len(out_rows), agree / len(out_rows)))
    bookkeeping_ok = all(
        (r["source"] == "adjudicated") == (src["unanimous"] != "True")
        for r, src in zip(out_rows, rows)
    )
    print("source matches unanimity:", bookkeeping_ok)
    print("em dash in output:", any("\u2014" in r["rationale"] for r in out_rows))


if __name__ == "__main__":
    main()
