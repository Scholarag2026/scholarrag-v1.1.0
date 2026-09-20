import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch, fetchWithAuth } from "@/lib/api";

// ── Types ────────────────────────────────────────────────────────────────────

/**
 * One atomic assertion the v2 claim-verification prompt decomposed a claim into, with
 * the source's verdict on it (`backend/app/schemas/fulltext.py::ClaimAssertion`).
 * Optional/absent on reports produced by the v1 prompt.
 */
export interface ClaimAssertion {
  text: string;
  /** "population" | "intervention" | "quantity" | "direction" | "scope" | "attribution" | "other" */
  kind: string;
  /** "supported" | "contradicted" | "absent" */
  verdict: string;
  claim_value?: string | null;
  source_value?: string | null;
  quote?: string | null;
  /**
   * The proposition the sentence exists to state, its main predicate over its main
   * subject. Exactly one assertion is marked `true` by the v3 prompt; all others are
   * peripheral. Optional so an older cached report (built before this field existed)
   * still parses.
   */
  central?: boolean;
  /** What the assertion is about (the "principal entity" the v3 prompt reasons
   * over). Optional/absent on older cached reports. */
  entity?: string | null;
  /** What is being measured or compared, paired with `entity` for the v3
   * contradiction and centrality checks. Optional/absent on older cached reports. */
  measure?: string | null;
  /** Every verbatim span composed to reach this assertion's verdict (v3's
   * `supported` composition rule can span more than one chunk). Optional/absent on
   * older cached reports. */
  quotes?: string[];
}

export interface ClaimVerification {
  claim_text: string;
  paper_id: string;
  status: "verified" | "unsupported" | "needs_nuance" | "no_full_text" | "error";
  evidence_quote: string | null;
  explanation: string;
  suggested_revision: string | null;
  /** Model name the provider reported for this call (v1.1.0). */
  model_reported?: string | null;
  /**
   * One entry per atomic assertion the v2 prompt decomposed the claim into.
   * Optional so an older cached report (built before this field existed) still parses.
   */
  assertions?: ClaimAssertion[];
  /**
   * The status the model returned before any code guard ran. `null` when no
   * guard-relevant model call was made; absent on older cached reports. Compare against
   * `status` to tell whether a code guard, not the model, produced the final verdict.
   */
  model_status?: string | null;
  /**
   * Machine-readable slugs for every guard that fired, in guard order, e.g.
   * "attribution_mismatch", "quote_not_verbatim", "numeric_not_in_source",
   * "numeric_outside_cited_passage", "no_full_text_with_chunks". Optional/empty on
   * older cached reports.
   */
  machine_reasons?: string[];
  /**
   * Machine-readable slugs from guards that never change `status` (e.g.
   * "numeric_not_in_source", "centrality_unmarked"). Always a sibling of
   * `machine_reasons`, never one of its members. Optional/empty on older cached
   * reports.
   */
  diagnostics?: string[];
  /**
   * Peripheral assertions the v3 precedence rule marks absent, in prose, for a
   * `needs_nuance` claim. Optional/empty on older cached reports or on claims of a
   * different status.
   */
  unstated_details?: string[];
  /**
   * Every verbatim span the model used across all assertions. `evidence_quote`
   * keeps its own name, type and position -- the service sets it to this list's first
   * span, so it never breaks an existing reader. Optional/empty on older cached reports.
   */
  evidence_quotes?: string[];
  /**
   * The full sentence the claim was cut from. Equal to `claim_text` when no
   * citation-link proposition narrowed the claim to a shorter clause of that sentence.
   * Set by the service, never by the model; absent on an older cached report.
   */
  claim_sentence?: string | null;
  /** Title of the paper the claim cites, if resolved. */
  paper_title?: string | null;
}

export interface ClaimVerificationProvenance {
  model_configured?: string | null;
  model_reported?: string[];
  temperature?: number | null;
  prompt_version?: string | null;
  calls?: number;
  input_tokens?: number | null;
  output_tokens?: number | null;
}

export interface ClaimVerificationReport {
  draft_id: string;
  verifications: ClaimVerification[];
  verified_count: number;
  unsupported_count: number;
  nuance_count: number;
  abstract_only_count: number;
  error_count: number;
  /** Claims a human should rewrite. Verification never edits the draft (decision D4). */
  needs_rewrite: number;
  /** Claims a human should remove. Verification never edits the draft (decision D4). */
  needs_removal: number;
  /** Which model/temperature/prompt version produced the verdicts (v1.1.0). */
  provenance?: ClaimVerificationProvenance | null;
  /** Share (0..1) of claims whose cited paper had full text available (v1.1.0). */
  full_text_coverage?: number;
  /** The job that produced this report; enables the claim-record CSV/JSON export. */
  task_id?: string | null;
  /**
   * Verifications with at least one assertion verdict "contradicted". Optional
   * so an older cached report (built before this field existed) still parses.
   */
  contradicted_count?: number;
  /** Verifications whose final status a code guard changed from the model's. */
  guarded_count?: number;
  /**
   * How citations in the draft were resolved, independent of how they were rendered.
   * Optional/absent on an older cached report, or one produced by `verify_user_edits`.
   */
  citation_coverage?: CitationCoverage | null;
  /**
   * `true` once the standalone action (`verify-and-heal`) has healed this draft -- it
   * removes an unsupported sentence rather than only reporting it. `null`/absent on an
   * older report, or on a report from `verify-edits`, which does not heal.
   */
  healed?: boolean | null;
  /** Removal/drop counts `finalize_draft_document` returned when it healed the draft. */
  finalize_stats?: Record<string, number> | null;
  /**
   * The one report of the final text: every claim that survives in the saved, healed
   * draft, all of them verified by construction. The product shows only this view --
   * no separate before/after report -- whenever it is present; an older report without
   * it has no healed view to show.
   */
  final_report?: FinalClaimReport | null;
}

/** The one report of the final text (`ClaimVerificationReport.final_report` above):
 * the claims that survive in the saved, healed draft. */
export interface FinalClaimReport {
  verifications: ClaimVerification[];
  verified_count: number;
}

/**
 * How the citations in a draft were resolved, independent of how they were rendered.
 * Additive/optional so a report built before this field existed, or one from
 * `verify_user_edits` (which only ever sees plain strings, never a Tiptap document),
 * still parses.
 */
export interface CitationCoverage {
  found: number;
  linked: number;
  sent_to_verifier: number;
  unresolved: number;
  unresolved_citations: string[];
  by_source: Record<string, number>;
}

export interface PasteFulltextRequest {
  text: string;
}

// ── Hooks ────────────────────────────────────────────────────────────────────

export function useTriggerAcquireFullTexts() {
  const queryClient = useQueryClient();
  return useMutation<
    { task_id: string },
    Error,
    { projectId: string }
  >({
    mutationFn: ({ projectId }) =>
      apiFetch(`/projects/${projectId}/acquire-full-texts`, {
        method: "POST",
      }),
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({
        queryKey: ["projects", projectId, "papers"],
      });
    },
  });
}

/**
 * Upload a full-text PDF for a specific paper.
 * Uses raw fetch (not apiFetch) because FormData requires the browser
 * to set the Content-Type with the multipart boundary automatically.
 */
export function useUploadFulltext() {
  const queryClient = useQueryClient();
  return useMutation<
    unknown,
    Error,
    { paperId: string; projectId?: string; file: File }
  >({
    mutationFn: async ({ paperId, file }) => {
      const formData = new FormData();
      formData.append("file", file);

      const API_URL =
        process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

      const res = await fetchWithAuth(`${API_URL}/papers/${paperId}/upload-fulltext`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const error = await res
          .json()
          .catch(() => ({ detail: res.statusText }));
        throw new Error(error.detail || error.message || res.statusText);
      }

      return res.json();
    },
    onSuccess: (_, { projectId }) => {
      if (projectId) {
        queryClient.invalidateQueries({
          queryKey: ["projects", projectId, "papers"],
        });
      }
    },
  });
}

export function usePasteFulltext() {
  const queryClient = useQueryClient();
  return useMutation<
    unknown,
    Error,
    { paperId: string; projectId?: string; text: string }
  >({
    mutationFn: ({ paperId, text }) =>
      apiFetch(`/papers/${paperId}/paste-fulltext`, {
        method: "POST",
        body: JSON.stringify({ text }),
      }),
    onSuccess: (_, { projectId }) => {
      if (projectId) {
        queryClient.invalidateQueries({
          queryKey: ["projects", projectId, "papers"],
        });
      }
    },
  });
}

export function useTriggerVerifyClaims() {
  const queryClient = useQueryClient();
  return useMutation<
    { task_id: string },
    Error,
    { projectId: string; draftId: string }
  >({
    mutationFn: ({ projectId, draftId }) =>
      apiFetch(`/projects/${projectId}/verify-and-heal`, {
        method: "POST",
        body: JSON.stringify({ draft_id: draftId }),
      }),
    onSuccess: (_, { projectId, draftId }) => {
      queryClient.invalidateQueries({
        queryKey: ["drafts", draftId, "claim-verification"],
      });
      // verify-and-heal heals the draft itself (removes an unsupported sentence rather
      // than only reporting it), so the editor's own copy of the draft is stale the
      // moment this call succeeds.
      queryClient.invalidateQueries({ queryKey: ["drafts", draftId] });
      queryClient.invalidateQueries({
        queryKey: ["projects", projectId, "papers"],
      });
    },
  });
}

export function useClaimVerification(draftId: string) {
  return useQuery<ClaimVerificationReport>({
    queryKey: ["drafts", draftId, "claim-verification"],
    queryFn: () => apiFetch(`/drafts/${draftId}/claim-verification`),
    enabled: !!draftId,
    retry: false,
  });
}
