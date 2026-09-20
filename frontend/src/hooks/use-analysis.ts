import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";

// Types

export interface PaperAnalysis {
  id: string;
  project_id: string;
  paper_id: string;
  quality_score: number;
  relevance_score: number | null;
  key_findings: string[];
  methodology: string;
  methodology_rigor: "high" | "medium" | "low";
  limitations: string[];
  theories_used: string[];
  sample_info: string | null;
  paper_title: string;
  paper_year: number | null;
  paper_doi: string | null;
}

export interface ResearchGap {
  gap_type: "unexplored" | "under_explored" | "controversial" | "methodological";
  description: string;
  evidence: string[];
  severity: "high" | "medium" | "low";
}

export interface Position {
  view: string;
  supporters: string[];
}

export interface Controversy {
  topic: string;
  positions: Position[];
}

export interface SuggestedQuestion {
  question: string;
  rationale: string;
  methodology_hint: string;
  related_gap_descriptions: string[];
}

export interface GapReport {
  gaps: ResearchGap[];
  controversies: Controversy[];
  suggested_questions: SuggestedQuestion[];
  theoretical_landscape: string[];
  summary: string;
}

// Hooks

export function usePaperAnalyses(projectId: string) {
  return useQuery<{ analyses: PaperAnalysis[]; total: number }>({
    queryKey: ["projects", projectId, "paper-analyses"],
    queryFn: () => apiFetch(`/projects/${projectId}/paper-analyses`),
    enabled: !!projectId,
  });
}

export function useGapReport(projectId: string) {
  // The backend returns 200 with a `null` body when no report has been generated yet
  // (it used to 404). Type the payload as nullable so consumers must branch on content.
  return useQuery<GapReport | null>({
    queryKey: ["projects", projectId, "gap-report"],
    queryFn: () => apiFetch<GapReport | null>(`/projects/${projectId}/gap-report`),
    enabled: !!projectId,
    retry: false,
  });
}

export function useAnalyzeQuality() {
  const queryClient = useQueryClient();
  return useMutation<{ task_id: string }, Error, { projectId: string; force?: boolean }>({
    mutationFn: ({ projectId, force = false }) =>
      apiFetch(`/projects/${projectId}/analyze-quality`, {
        method: "POST",
        body: JSON.stringify({ force }),
      }),
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({ queryKey: ["projects", projectId, "paper-analyses"] });
    },
  });
}

export function useAnalyzeGaps() {
  const queryClient = useQueryClient();
  return useMutation<{ task_id: string }, Error, { projectId: string }>({
    mutationFn: ({ projectId }) =>
      apiFetch(`/projects/${projectId}/analyze-gaps`, {
        method: "POST",
        body: JSON.stringify({}),
      }),
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({ queryKey: ["projects", projectId, "gap-report"] });
    },
  });
}
