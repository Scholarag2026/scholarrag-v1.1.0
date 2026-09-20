"""Adjudicate a round-3 blind triple annotation of a claim-verification set.

Reads ``hss_annotation_completed_v3.csv`` (the merge of annotator3_A/B/C.csv produced by
``aggregate_annotations_v3.py``) and writes ``hss_annotation_adjudicated_v3.csv``, one final
label per item, plus the two empty countersignature columns ``countersigned_by`` and
``countersigned_at`` that the user fills in at the end of the revision. ``--completed``,
``--out`` and ``--decisions`` point the same code at another set; the real-claims set is
run that way.

Adjudication trigger, key arm off: an item is adjudicated when the three sessions are not
unanimous. A unanimous blind majority stands as the label without reference to any
construction key. The HSS round-3 set has no non-unanimous item, so the built-in
``ADJUDICATED_LABEL`` is empty and ``n_adjudicated`` is 0 for it.

Every row still carries a rationale that was written by reading the cited paper's cached
text and quoting the decisive sentence from it. For the HSS set that text is in
``evaluation/claims/data/hss_test_v3_fulltext/`` and each quote was checked to occur
verbatim in it. The paper text is the authority, not the construction key:
``agrees_with_construction`` is reported as a diagnostic and no label is moved toward the
key.

The key is optional. When the completed sheet carries ``construction_category`` and
``expected_label`` (that is, the aggregator was given a key with expected labels), the
construction columns and the agreement count are written; without them those cells are
empty and the agreement line is not printed. The real-claims set has no expected labels at
all, because its claims and papers are real, so those cells stay empty there.

``--decisions`` replaces the two dictionaries below with the ones in a json file, so a set
other than the HSS one can be adjudicated without editing this file. The json carries an
``adjudicated_label`` object (ann_id to one of the four statuses, one entry per
non-unanimous item) and a ``rationale`` object (ann_id to the rationale text, one entry per
item). Without the flag the HSS dictionaries built into this file are used and the HSS
output is unchanged.

Deterministic: no randomness, no network calls, no LLM/paid API calls. Standard library
only.

Run with:
    python adjudicate_annotations_v3.py
    python adjudicate_annotations_v3.py --completed <csv> --out <csv> [--aggregate <json>]
        [--decisions <json>]
"""

from __future__ import annotations

import argparse
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_COMPLETED = os.path.join(HERE, "hss_annotation_completed_v3.csv")
DEFAULT_OUT = os.path.join(HERE, "hss_annotation_adjudicated_v3.csv")

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
    "countersigned_by",
    "countersigned_at",
]

LABELS = {"verified", "needs_nuance", "unsupported", "no_full_text"}

# Items the adjudicator decided against the cached paper text after a non-unanimous vote.
# The HSS test set had none: all 60 items were unanimous across sessions A, B and C.
ADJUDICATED_LABEL: dict[str, str] = {}

# One rationale per item, each written by reading the cited paper's cached text and
# quoting the decisive sentence from it. Every quote below was checked to occur
# verbatim in that cached text.

RATIONALE = {
    "ann-01": (
        "The cited paper is \"Potential sources of foreign language learning boredom: A Q "
        "methodology study\". Its cached text contains no occurrence of \"fluctuation\", "
        "\"boredom level\" or \"data collection tool\", and its closest sentence, \"To this aim, "
        "Mariusz Kruk, Mirosław Pawlak, Majid Elahi Shirvan, Tahereh Taherian, Elham Yazdanmehr "
        "38 a Q method, which shares features of both qualitative and quantitative research "
        "approaches, was used to explore 37 Iranian English as a foreign language (EFL) learners’ "
        "perceptions of potential sources of boredom in the classroom.\", shares 21% of the "
        "claim's content words and is about a different subject. The cited paper does not state "
        "the claim, so the citation does not support it."
    ),
    "ann-02": (
        "The cited paper reads \"Finally, the question asking for suggestions from the students "
        "revealed many useful ideas which may help to improve the language education in Turkey.\" "
        "The claim differs only in \"learners demonstrated\" for \"students revealed\", which "
        "does not change what the sentence asserts, so the paper supports the claim as stated."
    ),
    "ann-03": (
        "The cited paper reads \"This shows that participants who loaded significantly on Factor "
        "1 felt that having an unsupportive teacher was a much more important source of boredom "
        "than others who shared the views that emerged in Factors 2 and 3.\" The claim writes "
        "\"less\" where the paper writes \"more\", so the paper states a different comparative "
        "direction for the same entity and the same measure and contradicts the claim."
    ),
    "ann-04": (
        "The cited paper reads \"The results indicated significant differences across the groups "
        "on how writers constructed their authorial stance with interactional metadiscourse "
        "markers.\" The claim differs only in \"findings\" for \"results\", which does not change "
        "what the sentence asserts, so the paper supports the claim as stated."
    ),
    "ann-05": (
        "The cited paper contains the sentence verbatim: \"The relationship between the superior "
        "colonizer and the inferior colonized promoted by colonialism is further consolidated by "
        "discriminating narratives and practices that segregate and minimize subjectivities, "
        "knowledges, and autonomies.\" The claim reproduces it word for word, so the paper states "
        "the claim as written."
    ),
    "ann-06": (
        "The item shows the marker \"Evidence withheld: no passage or paper text is available for "
        "this item\", and source_texts_index_v3.json lists no path for it, so neither a passage "
        "nor the paper text was available to the sessions. All three abstained, which is the "
        "response the rubric requires whenever evidence is withheld, however plausible the claim "
        "sounds."
    ),
    "ann-07": (
        "The cited paper reads \"Following increased global mobility, there is high demand for "
        "teachers in SFI and there is an urgent need for the development of teacher education for "
        "this group.\" The claim writes \"decreased\" where the paper writes \"increased\", so "
        "the paper states a different direction for the same entity and the same measure and "
        "contradicts the claim."
    ),
    "ann-08": (
        "The item shows the marker \"Evidence withheld: no passage or paper text is available for "
        "this item\", and source_texts_index_v3.json lists no path for it, so neither a passage "
        "nor the paper text was available to the sessions. All three abstained, which is the "
        "response the rubric requires whenever evidence is withheld, however plausible the claim "
        "sounds."
    ),
    "ann-09": (
        "The cited paper reads \"Consequently, the inter-rater reliability was found 90%, which "
        "is above the satisfactory rate.\" The claim writes \"99%\" where the paper writes "
        "\"90%\", so the paper states a different value for the same entity and the same measure "
        "and contradicts the claim."
    ),
    "ann-10": (
        "The cited paper reads \"Moreover, thanks to the information provided by the teacher "
        "throughout the interview, we know that ChatGPT was effective in helping the teacher "
        "design most teaching tasks for her classes of different ages and levels.\" The claim "
        "writes \"few\" where the paper writes \"most\", so the paper states a different "
        "quantifier for the same entity and the same measure and contradicts the claim."
    ),
    "ann-11": (
        "The cited paper reads \"Still, 65% of all revisions were successful, amounting up to 86% "
        "in the collaborative context that led to improvement of the text.\" The claim writes "
        "\"89%\" where the paper writes \"65%\", so the paper states a different value for the "
        "same entity and the same measure and contradicts the claim."
    ),
    "ann-12": (
        "The cited paper reads \"Cronbach’s alpha for all research structures has been "
        "calculated, and it is found that the composite reliability and Cronbach’s alpha values "
        "for all variables are above 0.7 and show the appropriate reliability of the model.\" The "
        "claim writes \"1.0\" where the paper writes \"0.7\", so the paper states a different "
        "value for the same entity and the same measure and contradicts the claim."
    ),
    "ann-13": (
        "The cited paper reads \"We found the primary feature or advantage of the peer review "
        "process was the sense of empathy and resonance generated from the participants’ reading "
        "of the reviewed writings.\" The claim differs only in \"observed\" for \"found\", which "
        "does not change what the sentence asserts, so the paper supports the claim as stated."
    ),
    "ann-14": (
        "The cited paper is \"Exploring the Relationship Between Language Learning Strategies, "
        "Academic Achievement, Grade Level, and Gender\". Its cached text contains no occurrence "
        "of \"transcrib\", \"whole class interaction\", \"negotiation of meaning\" or "
        "\"communicative role\", and its closest sentence, \"On the other hand, indirect language "
        "learning strategies are grouped into affective (related to the ability to identify "
        "feelings and discuss them, as well as to the use of positive self-encouragement), "
        "metacognitive (which include good management of the learning process through planning "
        "tasks and evaluating accomplishments, etc.), and social strategies (which involve an "
        "interaction with other students as a significant component of the learning process "
        "through asking for help or clarification) (Oxford, 1990; 2003).\", shares 18% of the "
        "claim's content words and is about a different subject. The cited paper does not state "
        "the claim, so the citation does not support it."
    ),
    "ann-15": (
        "The item shows the marker \"Evidence withheld: no passage or paper text is available for "
        "this item\", and source_texts_index_v3.json lists no path for it, so neither a passage "
        "nor the paper text was available to the sessions. All three abstained, which is the "
        "response the rubric requires whenever evidence is withheld, however plausible the claim "
        "sounds."
    ),
    "ann-16": (
        "The cited paper reads \"By the same token, there have been studies of the curricular "
        "bases of reforms and guidelines designed mainly to achieve better results in the "
        "quality, effectiveness, and efficiency of teaching.\" The claim writes \"worse\" where "
        "the paper writes \"better\", so the paper states a different direction for the same "
        "entity and the same measure and contradicts the claim."
    ),
    "ann-17": (
        "The cited paper reads \"The findings indicated a significant impact of stress on English "
        "language teachers’ well-being, supporting previous studies that showed the negative "
        "effects of stress on various psychological aspects (Acheson & Nelson, 2020; De Costa et "
        "al., 2020; Hall, 2017).\" The claim writes \"positive\" where the paper writes "
        "\"negative\", so the paper states a different polarity for the same entity and the same "
        "measure and contradicts the claim."
    ),
    "ann-18": (
        "The cited paper is \"Translanguaging-as-Resource: University ESL Instructors’ Language "
        "Orientations and Attitudes Toward Translanguaging\". Its cached text contains no "
        "occurrence of \"irresistib\", \"undoubted\" or \"authority of teacher feedback\", and "
        "its closest sentence, \"According to García, Johnson, and Selĵ er (2017), the speciﬁ c "
        "core components of a translanguaging pedagogy include (a) a trans- languaging stance, "
        "which is the belief that the diverse language practices of TESL CANADA JOURNAL/REVUE "
        "TESL DU CANADA 25 VOLUME 36, ISSUE 1, 2019 students are valuable resources that should "
        "be used in the classroom; (b) a translanguaging design, which involves the design of "
        "strategic plans (e.g., lesson plans, assessments) that are informed by students’ diverse "
        "language practices; and (c) translanguaging shifts, which require the ability to make "
        "moment-by-moment changes to the lessons according to students’ needs.\", shares 11% of "
        "the claim's content words and is about a different subject. The cited paper does not "
        "state the claim, so the citation does not support it."
    ),
    "ann-19": (
        "The cited paper reads \"Furthermore, the result showed that teachers’ information and "
        "data literacy could positively predict teachers’ subjective norms.\" The claim differs "
        "only in \"Moreover,\" for \"Furthermore,\"; \"demonstrated\" for \"showed\"; "
        "\"instructors'\" for \"teachers'\"; \"instructors'\" for \"teachers'\", none of which "
        "changes what the sentence asserts, so the paper supports the claim as stated."
    ),
    "ann-20": (
        "The cited paper states \"The students with a low overall GPA (2.5-3.4) revealed somewhat "
        "different preferences in their strategy usage.\" That sentence supports the claim's main "
        "finding. The claim also adds \"using eye-tracking software\", and the cached text of the "
        "cited paper contains no occurrence of \"tracking\", \"eye-tracking\" or \"software\", so "
        "the added detail is not stated by the paper."
    ),
    "ann-21": (
        "The cited paper states \"Examples of the use of such approaches can be found in research "
        "projects carried out in the Languages School at Universidad de Antioquia or the Faculty "
        "of Human Sciences at Universidad Nacional de Colombia.\" That sentence supports the "
        "claim's main finding. The claim also adds \"in a bilingual classroom\", and the cached "
        "text of the cited paper contains no occurrence of \"bilingual\" or \"classroom\", so the "
        "added detail is not stated by the paper."
    ),
    "ann-22": (
        "The cited paper reads \"Although the College has historically functioned as a community "
        "college, the SOE is part of a 4-year baccalaureate degree program in which students earn "
        "a number of teaching endorsements, including English for Speakers of Other Languages "
        "(ESOL).\" The claim writes \"5-year\" where the paper writes \"4-year\", so the paper "
        "states a different value for the same entity and the same measure and contradicts the "
        "claim."
    ),
    "ann-23": (
        "The item shows the marker \"Evidence withheld: no passage or paper text is available for "
        "this item\", and source_texts_index_v3.json lists no path for it, so neither a passage "
        "nor the paper text was available to the sessions. All three abstained, which is the "
        "response the rubric requires whenever evidence is withheld, however plausible the claim "
        "sounds."
    ),
    "ann-24": (
        "The cited paper contains the sentence verbatim: \"The rest of the options show a lack of "
        "significant differences between the groups with p-values >.05 and very small effect size "
        "values.\" The claim reproduces it word for word, so the paper states the claim as "
        "written."
    ),
    "ann-25": (
        "The cited paper states \"We decided to exclude articles that did not report results of a "
        "research study or that had limitations in terms of providing accurate information about "
        "data collection, data analysis, participants, or other essential components of the "
        "research process.\" That sentence supports the claim's main finding. The claim also adds "
        "\"using a mobile application\", and the cached text of the cited paper contains no "
        "occurrence of \"mobile\" or \"application\", so the added detail is not stated by the "
        "paper."
    ),
    "ann-26": (
        "The cited paper contains the sentence verbatim: \"However, during the course, the "
        "participants developed an intricate view of language assessment.\" The claim reproduces "
        "it word for word, so the paper states the claim as written."
    ),
    "ann-27": (
        "The cited paper contains the sentence verbatim: \"Level two considered an analysis of "
        "the main findings or outcomes as reported in the studies to identify changes in teacher "
        "professional development associated with the implementation of specific initiatives.\" "
        "The claim reproduces it word for word, so the paper states the claim as written."
    ),
    "ann-28": (
        "The cited paper reads \"Group-level analysis revealed that learners experienced "
        "significantly higher focus and interest during tasks performed in video-chat mode than "
        "text-chat mode.\" The claim writes \"lower\" where the paper writes \"higher\", so the "
        "paper states a different direction for the same entity and the same measure and "
        "contradicts the claim."
    ),
    "ann-29": (
        "The cited paper reads \"According to her experience, ChatGPT provides creative and easy "
        "solutions for children in the classroom, something that is not commonly found in "
        "textbooks.\" The claim differs only in \"observed\" for \"found\", which does not change "
        "what the sentence asserts, so the paper supports the claim as stated."
    ),
    "ann-30": (
        "The cited paper states \"Most of the participants reported that they had enrolled in the "
        "English Language teacher education program because of their interest in learning "
        "English.\" That sentence supports the claim's main finding. The claim also adds \"using "
        "a mobile application\", and the cached text of the cited paper contains no occurrence of "
        "\"mobile\" or \"application\", so the added detail is not stated by the paper."
    ),
    "ann-31": (
        "The cited paper states \"Interestingly, we found evidence to suggest that materials used "
        "in designing language assessments were a pivotal factor that influences the pre-service "
        "teachers’ enterprise of design.\" That sentence supports the claim's main finding. The "
        "claim also adds \"using a mobile application\", and the cached text of the cited paper "
        "contains no occurrence of \"mobile\" or \"application\", so the added detail is not "
        "stated by the paper."
    ),
    "ann-32": (
        "The cited paper reads \"Revisions of both categories were deemed unnecessary by the "
        "coders in two thirds of the cases, however, cohesion revision led to improvement in four "
        "out of five cases, whereas structural changes did so only for 45%.\" The claim differs "
        "only in \"nevertheless,\" for \"however,\", which does not change what the sentence "
        "asserts, so the paper supports the claim as stated."
    ),
    "ann-33": (
        "The item shows the marker \"Evidence withheld: no passage or paper text is available for "
        "this item\", and source_texts_index_v3.json lists no path for it, so neither a passage "
        "nor the paper text was available to the sessions. All three abstained, which is the "
        "response the rubric requires whenever evidence is withheld, however plausible the claim "
        "sounds."
    ),
    "ann-34": (
        "The cited paper reads \"Contrary to laudable prior research findings, the majority of "
        "participants considered online review ineffective.\" The claim writes \"minority\" where "
        "the paper writes \"majority\", so the paper states a different quantifier for the same "
        "entity and the same measure and contradicts the claim."
    ),
    "ann-35": (
        "The cited paper reads \"Among the demographic variables, only gender indicated a "
        "significant difference between English language teachers, with women scoring higher on "
        "the PERMA scale than men.\" The claim differs only in \"instructors,\" for "
        "\"teachers,\", which does not change what the sentence asserts, so the paper supports "
        "the claim as stated."
    ),
    "ann-36": (
        "The cited paper reads \"The data frequently shows that, before the course, the "
        "participants thought that language assessment was about grades and/or tests.\" The claim "
        "writes \"rarely\" where the paper writes \"frequently\", so the paper states a different "
        "quantifier for the same entity and the same measure and contradicts the claim."
    ),
    "ann-37": (
        "The cited paper states \"It reports a study which examined the changes in the levels of "
        "boredom experienced by 13 English majors in four EFL classes and the factors accounting "
        "for such changes.\" That sentence supports the claim's main finding. The claim also adds "
        "\"using eye-tracking software\", and the cached text of the cited paper contains no "
        "occurrence of \"eye\", \"tracking\" or \"software\", so the added detail is not stated "
        "by the paper."
    ),
    "ann-38": (
        "The cited paper contains the sentence verbatim: \"All of the participants highly "
        "stressed the importance of being fully competent in the language, to know about the "
        "language, and to know effective language teaching strategies.\" The claim reproduces it "
        "word for word, so the paper states the claim as written."
    ),
    "ann-39": (
        "The cited paper reads \"The motivation of bored students is lower since they are "
        "disengaged from school subjects or tasks, which results in their inability to "
        "concentrate or simply manifest interest and joy (Pekrun & Linnenbrink-Garcia, 2012).\" "
        "The claim writes \"higher\" where the paper writes \"lower\", so the paper states a "
        "different direction for the same entity and the same measure and contradicts the claim."
    ),
    "ann-40": (
        "The cited paper reads \"It is worth mentioning the four participants were selected from "
        "a pool of 23 students, which indicates their anonymous reviews were not typically from "
        "each other.\" The claim writes \"32\" where the paper writes \"23\", so the paper states "
        "a different value for the same entity and the same measure and contradicts the claim."
    ),
    "ann-41": (
        "The cited paper states \"In these studies, we found that interaction in the form of "
        "group tasks with student-student interaction only rarely took place.\" That sentence "
        "supports the claim's main finding. The claim also adds \"among heritage speakers\", and "
        "the cached text of the cited paper contains no occurrence of \"heritage\" or "
        "\"speaker\", so the added detail is not stated by the paper."
    ),
    "ann-42": (
        "The cited paper states \"A content analysis of 32 interviews revealed four factors that "
        "accounted for changes in engagement during tasks: task design (e.g., task familiarity), "
        "task process (e.g., instances of collaboration), task condition (e.g., communication "
        "mode), and learner factors (e.g., perceptions of proficiency).\" That sentence supports "
        "the claim's main finding. The claim also adds \"among heritage speakers\", and the "
        "cached text of the cited paper contains no occurrence of \"heritage\", so the added "
        "detail is not stated by the paper."
    ),
    "ann-43": (
        "The cited paper is \"Towards Understanding Teacher Mentoring, Learner WCF Beliefs, and "
        "Learner Revision Practices Through Peer Review Feedback: A Sociocultural Perspective\". "
        "Its cached text contains no occurrence of \"stressor\", \"perception of stress\" or "
        "\".397\", and its closest sentence, \"When we checked the survey item that explored "
        "participants’ perception about learning from the peers’ writings, only HY reported she "
        "did learn from the reviewed writings.\", shares 12% of the claim's content words and is "
        "about a different subject. The cited paper does not state the claim, so the citation "
        "does not support it."
    ),
    "ann-44": (
        "The cited paper contains the sentence verbatim: \"With regard to how GELT can be added "
        "to such a classroom, two general paths seem to be possible.\" The claim reproduces it "
        "word for word, so the paper states the claim as written."
    ),
    "ann-45": (
        "The cited paper is \"Whole class interaction in the adult L2-classroom\". Its cached "
        "text contains no occurrence of \"text-chat\", \"text chat\" or \"speculated\", and its "
        "closest sentence, \"She argued that output has a consciousness-raising function that "
        "helps learners become aware of the gaps in their interlanguage and test their hypotheses "
        "about the L2.\", shares 19% of the claim's content words and is about a different "
        "subject. The cited paper does not state the claim, so the citation does not support it."
    ),
    "ann-46": (
        "The cited paper is \"Dynamic engagement in second language computer-mediated "
        "collaborative writing tasks: Does communication mode matter?\". Its cached text contains "
        "no occurrence of \"statement\", \"representative of each factor\" or \"rated "
        "significantly higher\", and its closest sentence, \"Group-level analysis revealed that "
        "learners experienced significantly higher focus and interest during tasks performed in "
        "video-chat mode than text-chat mode.\", shares 14% of the claim's content words and is "
        "about a different subject. The cited paper does not state the claim, so the citation "
        "does not support it."
    ),
    "ann-47": (
        "The cited paper is \"The role of university teachers’ 21st-century digital competence in "
        "their attitudes toward ICT integration in higher education: Extending the theory of "
        "planned behavior\". Its cached text contains no occurrence of \"member check\" or "
        "\"transcript\", and its closest sentence, \"18 no.2 Participants The current study’s "
        "participants’ sample was selected during the emergency remote language teaching by "
        "utilizing purposeful sampling of university teach- ers to provide rich information for "
        "the study goal.\", shares 17% of the claim's content words and is about a different "
        "subject. The cited paper does not state the claim, so the citation does not support it."
    ),
    "ann-48": (
        "The cited paper reads \"Rather, language teaching can be viewed as a dynamic process "
        "that engages students’ multiple meaning-making resources (Mazak, 2017).\" The claim "
        "differs only in \"learners'\" for \"students'\", which does not change what the sentence "
        "asserts, so the paper supports the claim as stated."
    ),
    "ann-49": (
        "The cited paper reads \"Such findings indicate that as the students’ use of memory and "
        "affective strategies increases, their EFL achievement tends to decrease, whereas their "
        "greater use of cognitive strategies causes an increase in their EFL achievement.\" The "
        "claim writes \"decreases\" where the paper writes \"increases\", so the paper states a "
        "different direction for the same entity and the same measure and contradicts the claim."
    ),
    "ann-50": (
        "The cited paper reads \"The data were analyzed descriptively; the results showed that "
        "most teachers are not satisfied with the coursebooks, crowded classes, quite limited "
        "class hours, and unmotivated students.\" The claim writes \"few\" where the paper writes "
        "\"most\", so the paper states a different quantifier for the same entity and the same "
        "measure and contradicts the claim."
    ),
    "ann-51": (
        "The item shows the marker \"Evidence withheld: no passage or paper text is available for "
        "this item\", and source_texts_index_v3.json lists no path for it, so neither a passage "
        "nor the paper text was available to the sessions. All three abstained, which is the "
        "response the rubric requires whenever evidence is withheld, however plausible the claim "
        "sounds."
    ),
    "ann-52": (
        "The cited paper reads \"The internationalization of higher education—driven by "
        "political, economic, and sociocultural dimensions (Maringe, 2010)—has resulted in a "
        "steady increase in the number of linguistically and culturally diverse students in "
        "Canadian educational institutions.\" The claim writes \"decrease\" where the paper "
        "writes \"increase\", so the paper states a different direction for the same entity and "
        "the same measure and contradicts the claim."
    ),
    "ann-53": (
        "The item shows the marker \"Evidence withheld: no passage or paper text is available for "
        "this item\", and source_texts_index_v3.json lists no path for it, so neither a passage "
        "nor the paper text was available to the sessions. All three abstained, which is the "
        "response the rubric requires whenever evidence is withheld, however plausible the claim "
        "sounds."
    ),
    "ann-54": (
        "The cited paper states \"In the present study, PQ Method (Schmolck, 2002) was employed "
        "to produce the initial by-person correlation matrix.\" That sentence supports the "
        "claim's main finding. The claim also adds \"using eye-tracking software\", and the "
        "cached text of the cited paper contains no occurrence of \"eye\", \"eye-tracking\" or "
        "\"software\", so the added detail is not stated by the paper."
    ),
    "ann-55": (
        "The cited paper states \"The activities were reported useful (e.g., increased GELT "
        "awareness, positive attitudes toward GELT, and willingness to learn more about GELT) to "
        "help teachers understand GELT even though the concepts were not taught explicitly.\" "
        "That sentence supports the claim's main finding. The claim also adds \"during the winter "
        "semester\", and the cached text of the cited paper contains no occurrence of \"winter\" "
        "or \"semester\", so the added detail is not stated by the paper."
    ),
    "ann-56": (
        "The cited paper is \"Well-being and the Perception of Stress among EFL University "
        "Teachers in Saudi Arabia\". Its cached text contains no occurrence of \"compensation\", "
        "\"metacognitive\" or \"insignificant\", and its closest sentence, \"Recently, research "
        "inspired by positive psychology (MacIntyre et al., 2019; Williams et al., 2016) produced "
        "a range of studies on the professional lives of teachers (Day & Gu, 2014; Hiver & "
        "Dörnyei, 2017), including the role of teachers’ emotional and social intelligences in "
        "learning processes (Elias & Arnold, 2006) and in teacher well-being (Day & Gu, 2009; "
        "Gkonou & Mercer, 2017).\", shares 29% of the claim's content words and is about a "
        "different subject. The cited paper does not state the claim, so the citation does not "
        "support it."
    ),
    "ann-57": (
        "The cited paper reads \"A cross-genre analysis of the findings reveals that the total "
        "number of interpersonal metadiscourse features (normalized per 1000 words) is higher in "
        "BR sub-corpus (76.26) than in RA sub-corpus (44.09).\" The claim writes \"1370\" where "
        "the paper writes \"1000\", so the paper states a different value for the same entity and "
        "the same measure and contradicts the claim."
    ),
    "ann-58": (
        "The cited paper contains the sentence verbatim: \"When exposed to a perspective of "
        "dynamic bilingualism—one that visibly challenged prior experiences and "
        "beliefs—throughout their time in class, TCs began to speak of enhanced understanding of "
        "bilingualism and increased feelings of empowerment (both for themselves and others).\" "
        "The claim reproduces it word for word, so the paper states the claim as written."
    ),
    "ann-59": (
        "The cited paper contains the sentence verbatim: \"The results demonstrated a significant "
        "difference between the scores of two tests both in control and DI groups, which suggests "
        "that both traditional instruction and DI gave rise to an increment in students’ L2 "
        "overall achievement.\" The claim reproduces it word for word, so the paper states the "
        "claim as written."
    ),
    "ann-60": (
        "The item shows the marker \"Evidence withheld: no passage or paper text is available for "
        "this item\", and source_texts_index_v3.json lists no path for it, so neither a passage "
        "nor the paper text was available to the sessions. All three abstained, which is the "
        "response the rubric requires whenever evidence is withheld, however plausible the claim "
        "sounds."
    ),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed", default=DEFAULT_COMPLETED)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--aggregate",
        default=None,
        help="optional aggregate json, cross-checked against the adjudication trigger",
    )
    parser.add_argument(
        "--decisions",
        default=None,
        help=(
            "optional decisions json for a set other than the HSS one, with an "
            "'adjudicated_label' object and a 'rationale' object keyed by ann_id; without it "
            "the HSS round-3 dictionaries built into this file are used"
        ),
    )
    return parser.parse_args(argv)


def load_decisions(path):
    """Return (adjudicated_label, rationale) read from a decisions json."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    rationale = payload.get("rationale")
    if not isinstance(rationale, dict) or not rationale:
        raise ValueError(f"{path} carries no non-empty 'rationale' object")
    adjudicated = payload.get("adjudicated_label", {})
    if not isinstance(adjudicated, dict):
        raise ValueError(f"{path} carries an 'adjudicated_label' that is not an object")
    return adjudicated, rationale


def load_completed(path):
    with open(path, encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


def adjudicate(rows, *, aggregate=None, adjudicated_label=None, rationale=None):
    """Return (out_rows, stats). Raises when the rationales do not cover the input."""
    adjudicated_label = ADJUDICATED_LABEL if adjudicated_label is None else adjudicated_label
    rationale = RATIONALE if rationale is None else rationale
    ann_ids = [row["ann_id"] for row in rows]
    missing = [a for a in ann_ids if a not in rationale]
    if missing:
        raise ValueError(f"no rationale for {missing}")
    stray = [a for a in adjudicated_label if a not in ann_ids]
    if stray:
        raise ValueError(f"adjudicated label for unknown item {stray}")
    bad = sorted({v for v in adjudicated_label.values()} - LABELS)
    if bad:
        raise ValueError(f"adjudicated label outside the four statuses: {bad}")

    has_key = "construction_category" in rows[0] and "expected_label" in rows[0]
    triggered = [r["ann_id"] for r in rows if str(r.get("unanimous", "")).lower() != "true"]
    uncovered = [a for a in triggered if a not in adjudicated_label]
    if uncovered:
        raise ValueError(f"non-unanimous items without an adjudicated label: {uncovered}")

    if aggregate is not None:
        # The aggregate lists the flagged items by item_id when it was given a key and by
        # ann_id otherwise, so translate to ann_ids before comparing.
        by_item_id = {r.get("item_id", ""): r["ann_id"] for r in rows if r.get("item_id", "")}
        flagged = sorted(
            by_item_id.get(flag, flag)
            for flag in aggregate.get("items_needing_adjudication", [])
        )
        if flagged != sorted(a for a in adjudicated_label):
            raise ValueError(
                "the aggregate flags "
                f"{flagged} but the adjudicated labels cover {sorted(adjudicated_label)}"
            )

    out_rows = []
    agree = 0
    n_with_key = 0
    disagreements = []
    for row in rows:
        ann_id = row["ann_id"]
        adjudicated = ann_id in adjudicated_label
        final = adjudicated_label[ann_id] if adjudicated else row["majority_label"]
        if final not in LABELS:
            raise ValueError(f"{ann_id}: final label {final!r} is not one of the four statuses")
        expected = row.get("expected_label", "") if has_key else ""
        if has_key and expected:
            n_with_key += 1
            agrees = final == expected
            agree += int(agrees)
            if not agrees:
                disagreements.append((ann_id, row.get("item_id", ""), final, expected))
        out_rows.append(
            {
                "ann_id": ann_id,
                "item_id": row.get("item_id", ""),
                "final_label": final,
                "source": "adjudicated" if adjudicated else "unanimous",
                "rationale": rationale[ann_id],
                "evidence_mode": row.get("evidence_mode", ""),
                "construction_category": row.get("construction_category", "") if has_key else "",
                "expected_label": expected,
                "agrees_with_construction": (
                    str(final == expected) if has_key and expected else ""
                ),
                "countersigned_by": "",
                "countersigned_at": "",
            }
        )
    stats = {
        "n_items": len(out_rows),
        "n_adjudicated": len(adjudicated_label),
        "has_key": has_key,
        "n_with_key": n_with_key,
        "construction_agreement": agree,
        "disagreements": disagreements,
    }
    return out_rows, stats


def write_csv(path, out_rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)


def main(argv=None):
    args = parse_args(argv)
    rows = load_completed(args.completed)
    aggregate = None
    if args.aggregate:
        with open(args.aggregate, encoding="utf-8") as handle:
            aggregate = json.load(handle)
    adjudicated_label = None
    rationale = None
    if args.decisions:
        adjudicated_label, rationale = load_decisions(args.decisions)
    out_rows, stats = adjudicate(
        rows,
        aggregate=aggregate,
        adjudicated_label=adjudicated_label,
        rationale=rationale,
    )
    write_csv(args.out, out_rows)

    print(f"Wrote {args.out}")
    print(f"Items: {stats['n_items']}")
    print(f"Adjudicated (non-unanimous): {stats['n_adjudicated']}")
    counts: dict[str, int] = {}
    for row in out_rows:
        counts[row["final_label"]] = counts.get(row["final_label"], 0) + 1
    print("Final label distribution: " + ", ".join(
        f"{label} {counts[label]}" for label in sorted(counts)
    ))
    if stats["has_key"] and stats["n_with_key"]:
        print(
            "Adjudicated label equals the construction label: "
            f"{stats['construction_agreement']}/{stats['n_with_key']}"
        )
        if stats["disagreements"]:
            for ann_id, item_id, final, expected in stats["disagreements"]:
                print(f"  {ann_id} ({item_id}): adjudicated {final}, construction {expected}")
        else:
            print("  no disagreement between the adjudicated labels and the construction labels")
    return stats


if __name__ == "__main__":
    main()
