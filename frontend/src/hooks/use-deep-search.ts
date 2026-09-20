import { useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";

// ── Types ────────────────────────────────────────────────────────────────────

export interface DeepSearchRequest {
  query: string;
  year_from?: number | null;
  year_to?: number | null;
  min_citations?: number | null;
  max_rounds?: number;
}

export interface RoundMetrics {
  round: number;
  strategy: string;
  new_papers: number;
  queries: string[];
  papers_expanded?: number | null;
  stopped: boolean;
}

export interface CoverageMetrics {
  total_scanned: number;
  total_unique: number;
  rounds: RoundMetrics[];
  sources: Record<string, number>;
  yield_curve: number[];
  confidence: string;
}

export interface SeedExpandRequest {
  dois?: string[];
  titles?: string[];
}

export interface FieldFoundationsRequest {
  topic: string;
  research_questions?: string[];
}

export interface FoundationalWork {
  suggested_title: string;
  suggested_authors: string[];
  suggested_year: number;
  why_essential: string;
  verified: boolean;
  matched_paper: Record<string, unknown> | null;
}

// ── Hooks ────────────────────────────────────────────────────────────────────

export function useTriggerDeepSearch() {
  const queryClient = useQueryClient();
  return useMutation<
    { task_id: string },
    Error,
    { projectId: string } & DeepSearchRequest
  >({
    mutationFn: ({ projectId, ...body }) =>
      apiFetch(`/projects/${projectId}/deep-search`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({
        queryKey: ["projects", projectId, "papers"],
      });
    },
  });
}

export function useTriggerSeedExpand() {
  const queryClient = useQueryClient();
  return useMutation<
    { task_id: string },
    Error,
    { projectId: string } & SeedExpandRequest
  >({
    mutationFn: ({ projectId, ...body }) =>
      apiFetch(`/projects/${projectId}/seed-expand`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({
        queryKey: ["projects", projectId, "papers"],
      });
    },
  });
}

export function useTriggerFieldFoundations() {
  const queryClient = useQueryClient();
  return useMutation<
    { task_id: string },
    Error,
    { projectId: string } & FieldFoundationsRequest
  >({
    mutationFn: ({ projectId, ...body }) =>
      apiFetch(`/projects/${projectId}/field-foundations`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({
        queryKey: ["projects", projectId, "papers"],
      });
    },
  });
}
