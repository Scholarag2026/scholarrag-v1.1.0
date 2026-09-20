import { useMutation } from "@tanstack/react-query";
// Relative, not the "@/" alias: this file has pure per-criterion stage helpers
// (see resetCriteriaFlags/toggleCriterionFlag/criteriaStagesFromFlags below) and
// they are imported directly by frontend/tests/smart-search-stages.test.mjs under
// `node --experimental-strip-types`, which -- unlike the Next.js build -- has no
// resolver for the "@/*" tsconfig path alias. A relative import resolves identically
// under both.
import { apiFetch } from "../lib/api";
import type { SecondPassStageLike } from "../lib/provenance";

export interface ScopeRefineResponse {
  is_specific_enough: boolean;
  clarifying_question: string | null;
  options: string[];
  refined_topic: string | null;
  inclusion_criteria: string[];
  exclusion_criteria: string[];
}

/**
 * Web of Science venue filter (Stage 1 of screening).
 * - "auto": apply only when a journal list has been imported (default)
 * - "on":   require it (the API answers 409 when no list is imported)
 * - "off":  skip Stage 1 and send every record to LLM screening
 */
export type WosFilter = "auto" | "on" | "off";

export interface SmartSearchPaper {
  doi: string | null;
  title: string;
  authors: { name: string }[];
  year: number | null;
  journal_name: string | null;
  journal_issn?: string | null;
  citation_count: number | null;
  abstract: string | null;
  source_api: string;
  external_id: string | null;
  full_text_url: string | null;
  wos_indexed?: boolean | null;
  wos_collection?: string | null;
  wos_categories?: string[] | null;
  /** Reason the screener gave for including this record (v1.1.0). */
  screening_reason?: string | null;
  /** Protocol-eligibility decision: "INCLUDE" | "EXCLUDE" | "NEEDS_REVIEW". */
  screening_status?: string | null;
  /** Criterion id (e.g. "E3") an EXCLUDE or a demoted NEEDS_REVIEW is anchored to. */
  screening_criterion?: string | null;
  /** Verbatim quote from the shown title/abstract backing the criterion, if any. */
  screening_quote?: string | null;
  /** Why a NEEDS_REVIEW was flagged this way: "full_text_criterion" | "cut_abstract" |
   * "unanchored_exclude" | "unquoted_criterion" | "no_abstract", or "" when no guard
   * attribution applies to this decision. */
  screening_guard_reason?: string | null;
  /** On an INCLUDE, the full-text inclusion criterion ids the shown
   * text does not already confirm; [] otherwise. */
  screening_to_confirm?: string[] | null;
}

//
// Stage-aware criteria: a criterion the screener cannot decide from a title and
// abstract alone (e.g. "administered the LASSI") is marked "full_text" via a per-line
// "Needs full text" checkbox in the Scope Ready card. The three helpers below are pure
// and hold all of that bookkeeping, so the panel component only ever calls them -- and
// so they can be tested without rendering React (see
// frontend/tests/smart-search-stages.test.mjs).

/** The two stages a criterion can be marked at (`SmartSearchRequest.CRITERION_STAGES`
 * on the backend). Backward compatible: an empty stages array means every criterion is
 * "abstract", the pre-S8 default. */
export type CriterionStage = "abstract" | "full_text";

/** A fresh, all-unchecked flag array for a criteria list of the given length. Called
 * wherever the criteria array itself is replaced (scope refinement resolving, going
 * back to an earlier answer, resetting the form), so a stale flag can never outlive
 * the criterion it described. */
export function resetCriteriaFlags(length: number): boolean[] {
  return new Array(length).fill(false);
}

/** Flips exactly the flag at `index`, immutably (a fresh array; `flags` is untouched).
 * The panel's per-line checkbox `onChange` calls this with its own index. */
export function toggleCriterionFlag(flags: boolean[], index: number): boolean[] {
  return flags.map((flag, i) => (i === index ? !flag : flag));
}

/** Turns the panel's checkbox flags into the wire shape `SmartSearchRequest` validates:
 * `[]` when nothing is checked (every criterion stays "abstract", the smallest and most
 * backward-compatible payload), otherwise a list exactly parallel to `flags`. */
export function criteriaStagesFromFlags(flags: boolean[]): CriterionStage[] {
  if (flags.length === 0 || flags.every((flag) => !flag)) return [];
  return flags.map((flag) => (flag ? "full_text" : "abstract"));
}

export interface SmartSearchCriteria {
  query: string;
  inclusion_criteria: string[];
  exclusion_criteria: string[];
  /** Parallel to inclusion_criteria: "abstract" | "full_text" per criterion. */
  inclusion_criteria_stages?: CriterionStage[];
  exclusion_criteria_stages?: CriterionStage[];
  wos_filter: WosFilter;
}

/** PRISMA-compatible record counts; every field is optional so older results still render. */
export interface SmartSearchFlow {
  identified?: number;
  duplicates_removed?: number;
  stage1_screened?: number;
  stage1_excluded?: number;
  stage2_screened?: number;
  stage2_excluded?: number;
  /** Reached Stage 2 but the shown text could not decide it. */
  needs_review?: number;
  unscreened?: number;
  included?: number;
  rounds?: number;
  stop_reason?: string | null;
  /** NEEDS_REVIEW counts by guard reason, plus "undecidable"/"no_abstract". A dict, not
   * a flow row: FLOW_KEYS in lib/provenance.ts stays at seven. */
  needs_review_by_reason?: Record<string, number>;
}

export interface ScreeningRecordEntry {
  title: string;
  doi: string | null;
  year: number | null;
  journal: string | null;
  journal_issn: string | null;
  openalex_id: string | null;
  stage: "wos" | "llm";
  /** Protocol-eligibility decision: "" when the record never reached the LLM
   * screener (e.g. a Stage 1 exclusion or an unscreened record). */
  status?: string;
  criterion?: string;
  quote?: string;
  /** Why a NEEDS_REVIEW record was flagged; "" for every other outcome. */
  needs_review_reason?: string;
  reason: string;
}

export interface ScreenerProvenance {
  model_configured?: string | null;
  model_reported?: string[];
  system_fingerprints?: string[];
  temperature?: number | null;
  prompt_version?: string | null;
  calls?: number;
  input_tokens?: number | null;
  output_tokens?: number | null;
}

/** One query a round issued, and how it did (`backend/app/services/smart_search.py`). */
export interface SmartSearchRoundQuery {
  query: string;
  /** How many papers this query returned. */
  returned: number;
  /** Of those, how many survived DOI/title dedup into this round's own paper set. */
  new_unique: number;
}

/** One round's own log line (`provenance.rounds`): every query it
 * issued, and what that round's screening did with the result. */
export interface SmartSearchRoundLogEntry {
  round: number;
  queries: SmartSearchRoundQuery[];
  /** Papers that got a screening decision this round. */
  screened: number;
  /** New INCLUDE papers this round added. */
  included_new: number;
  /** New NEEDS_REVIEW papers this round flagged. */
  needs_review_new: number;
}

/** The two stop-rule thresholds a run applied (`provenance.settings`). */
export interface SmartSearchSettings {
  min_rounds?: number;
  dry_round_patience?: number;
}

export interface SmartSearchResult {
  papers: SmartSearchPaper[];
  total_included?: number;
  total_scanned?: number;
  rounds?: number;
  elapsed_minutes?: number;
  stop_reason?: string | null;
  /** False when Stage 1 (venue filter) was skipped — by choice or because no list is imported. */
  stage1_applied?: boolean;
  criteria?: SmartSearchCriteria;
  flow?: SmartSearchFlow;
  excluded?: ScreeningRecordEntry[];
  /** Full paper dicts flagged NEEDS_REVIEW: rendered with the same PaperCard the
   * included list uses, behind the "Flagged for review" filter chip. */
  needs_review?: SmartSearchPaper[];
  unscreened?: ScreeningRecordEntry[];
  provenance?: {
    screener?: ScreenerProvenance;
    query_generator?: { model_configured?: string | null; prompt_version?: string | null };
    /** The min-rounds/dry-round-patience stop rule this run applied. */
    settings?: SmartSearchSettings;
    /** One entry per round actually run, each with its own query log. */
    rounds?: SmartSearchRoundLogEntry[];
    /** The job-level inclusion-only second-pass stage, run once for the whole job after
     * every round's own search loop ends, never attributed to any one round -- see
     * lib/provenance.ts's own secondPassStageSummary. */
    screener_second_pass?: SecondPassStageLike;
  };
}

export function useRefineScope() {
  return useMutation<
    ScopeRefineResponse,
    Error,
    { query: string; previous_answers: { question: string; answer: string }[] }
  >({
    mutationFn: (body) =>
      apiFetch("/scope/refine", {
        method: "POST",
        body: JSON.stringify(body),
      }),
  });
}

export function useTriggerSmartSearch() {
  return useMutation<
    { task_id: string },
    Error,
    {
      projectId: string;
      query: string;
      inclusion_criteria?: string[];
      exclusion_criteria?: string[];
      inclusion_criteria_stages?: CriterionStage[];
      exclusion_criteria_stages?: CriterionStage[];
      wos_filter?: WosFilter;
    }
  >({
    mutationFn: ({ projectId, ...body }) =>
      apiFetch(`/projects/${projectId}/smart-search`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
  });
}
