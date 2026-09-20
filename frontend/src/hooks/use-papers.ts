// frontend/src/hooks/use-papers.ts
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";

export interface Paper {
  id: string;
  doi: string | null;
  title: string;
  authors: { name: string }[];
  year: number | null;
  journal_name: string | null;
  citation_count: number | null;
  abstract: string | null;
  source_api: string;
  full_text_url: string | null;
  is_wos_indexed: boolean | null;
  wos_collection: string | null;
  wos_categories: string | null;
  created_at: string;
  metadata: {
    fulltext_status?: "acquired" | "abstract_only";
    fulltext_source?: string;
    fulltext_char_count?: number;
    deep_analysis?: Record<string, unknown>;
  } | null;
}

export interface ProjectPaper {
  id: string;
  project_id: string;
  paper_id: string;
  paper: Paper;
  relevance_score: number | null;
  user_notes: string | null;
  tags: Record<string, unknown> | null;
  added_at: string;
}

export function useProjectPapers(projectId: string) {
  return useQuery<{ papers: ProjectPaper[]; total: number }>({
    queryKey: ["projects", projectId, "papers"],
    queryFn: () => apiFetch(`/projects/${projectId}/papers?limit=500`),
    enabled: !!projectId,
  });
}

export function useAddPaper(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (paperData: {
      doi?: string | null;
      title: string;
      authors: { name: string }[];
      year?: number | null;
      journal_name?: string | null;
      citation_count?: number | null;
      abstract?: string | null;
      source_api: string;
      external_id?: string | null;
      full_text_url?: string | null;
      journal_issn?: string | null;
      wos_collection?: string | null;
      wos_categories?: string | null;
    }) =>
      apiFetch(`/projects/${projectId}/papers`, {
        method: "POST",
        body: JSON.stringify({ paper_data: paperData }),
      }),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ["projects", projectId, "papers"] }),
  });
}

export function useRemovePaper(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (paperId: string) =>
      apiFetch(`/projects/${projectId}/papers/${paperId}`, {
        method: "DELETE",
      }),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ["projects", projectId, "papers"] }),
  });
}

export function useDeepAnalyze(projectId: string) {
  const qc = useQueryClient();
  return useMutation<{ task_id: string }, Error, void>({
    mutationFn: () =>
      apiFetch(`/projects/${projectId}/deep-analyze`, { method: "POST" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["projects", projectId, "papers"] });
    },
  });
}

export function useAcquireFullTexts(projectId: string) {
  const qc = useQueryClient();
  return useMutation<
    { task_id: string },
    Error,
    { paper_ids: string[] }
  >({
    mutationFn: (body) =>
      apiFetch(`/projects/${projectId}/acquire-full-texts`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["projects", projectId, "papers"] });
    },
  });
}
