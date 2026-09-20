import { useMutation } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import { useTaskPolling } from "@/hooks/use-task-polling";

export interface VerificationCheck {
  check_type: string;
  status: string;
  message: string;
  details: Record<string, unknown> | null;
}

/** Derived from the record checks only (doi_exists, metadata_match). */
export type RecordStatus = "pass" | "warning" | "fail" | "no_doi";

export interface PaperVerificationResult {
  paper_id: string;
  doi: string | null;
  title: string;
  overall_status: RecordStatus | string;
  /** Record checks: doi_exists, metadata_match. */
  checks: VerificationCheck[];
  /**
   * Contextual indicators (wos_indexed, recency, citation_count) with status
   * info | note | skipped. They never change overall_status. Optional for older results.
   */
  indicators?: VerificationCheck[];
}

export interface VerificationReport {
  total: number;
  passed: number;
  failed: number;
  warnings: number;
  results: PaperVerificationResult[];
}

export function useVerifyReferences(projectId: string) {
  return useMutation<
    { task_id: string },
    Error,
    { paper_ids?: string[] }
  >({
    mutationFn: (data) =>
      apiFetch(`/projects/${projectId}/verify-references`, {
        method: "POST",
        body: JSON.stringify({ paper_ids: data.paper_ids || [] }),
      }),
  });
}

export function useTaskResult(taskId: string | null) {
  return useTaskPolling<{
    status: string;
    progress: number;
    progress_message: string | null;
    result: VerificationReport | null;
    error: string | null;
  }>({ taskId, kind: "verification" });
}
