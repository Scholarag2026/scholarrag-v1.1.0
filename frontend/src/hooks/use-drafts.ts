import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import { pollTaskResult } from "@/lib/task-poll";

export interface DraftVersion {
  id: string;
  version: number;
  change_summary: string | null;
  created_at: string;
}

export interface Draft {
  id: string;
  project_id: string;
  title: string;
  paper_type: string;
  content: Record<string, unknown> | null;
  current_version: number;
  status: string;
  created_at: string;
  updated_at: string;
}

export function useProjectDrafts(projectId: string) {
  return useQuery<{ drafts: Draft[] }>({
    queryKey: ["projects", projectId, "drafts"],
    queryFn: () => apiFetch(`/projects/${projectId}/drafts`),
    enabled: !!projectId,
  });
}

export function useDraft(draftId: string | null) {
  return useQuery<{ draft: Draft; versions: DraftVersion[] }>({
    queryKey: ["drafts", draftId],
    queryFn: () => apiFetch(`/drafts/${draftId}`),
    enabled: !!draftId,
  });
}

export function useCreateDraft(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { title: string; paper_type?: string }) =>
      apiFetch(`/projects/${projectId}/drafts`, {
        method: "POST",
        body: JSON.stringify(data),
      }),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ["projects", projectId, "drafts"] }),
  });
}

export function useUpdateDraft(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      draftId,
      ...data
    }: {
      draftId: string;
      title?: string;
      content?: Record<string, unknown>;
      status?: string;
    }) =>
      apiFetch(`/drafts/${draftId}`, {
        method: "PUT",
        body: JSON.stringify(data),
      }),
    onSuccess: (_data, variables) => {
      qc.invalidateQueries({ queryKey: ["drafts", variables.draftId] });
      qc.invalidateQueries({ queryKey: ["projects", projectId, "drafts"] });
    },
  });
}

export function useDeleteDraft(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (draftId: string) =>
      apiFetch(`/drafts/${draftId}`, { method: "DELETE" }),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ["projects", projectId, "drafts"] }),
  });
}

/** Code-level audit of APA in-text citations against the project library (v1.1.0). */
export interface CitationAudit {
  matched: string[];
  unmatched: string[];
  needs_citation_flags: number;
  total: number;
}

export interface WritingProvenance {
  model_configured?: string | null;
  model_reported?: string[] | string | null;
  temperature?: number | null;
  max_tokens?: number | null;
  prompt_version?: string | null;
}

/**
 * One sentence-to-reference entry from the writer's citation-link map.
 * ``paragraph_index`` is an index into the generated section's own paragraphs, split on
 * blank lines exactly as ``[b for b in content.split("\n\n") if b.strip()]`` does;
 * attaching it onto the matching Tiptap paragraph node is what carries it forward.
 */
export interface CitationLink {
  paragraph_index: number;
  sentence: string;
  keys: string[];
  citation_text: string;
}

export interface GenerateSectionResult {
  content: string;
  section_type: string;
  papers_used: number;
  citation_audit?: CitationAudit | null;
  citation_links?: CitationLink[] | null;
  provenance?: WritingProvenance | null;
}

export function useGenerateSection() {
  return useMutation<
    { task_id: string },
    Error,
    {
      draftId: string;
      section_type: string;
      context?: string;
      language?: string;
      /**
       * The target body length in words, sent to the model plus a hard maximum 25% over
       * it. `undefined` runs generation with no length control at all.
       */
      target_words?: number;
    }
  >({
    mutationFn: ({ draftId, ...data }) =>
      apiFetch(`/drafts/${draftId}/generate`, {
        method: "POST",
        body: JSON.stringify(data),
      }),
  });
}

export interface ComplianceCheck {
  check_type: string;
  status: string;
  message: string;
  details: Record<string, unknown> | null;
}

export interface ComplianceReport {
  overall_status: string;
  checks: ComplianceCheck[];
  total_word_count: number;
  sections_found: string[];
}

export function useCheckCompliance() {
  return useMutation<ComplianceReport, Error, { draftId: string }>({
    mutationFn: async ({ draftId }) => {
      const { task_id } = await apiFetch<{ task_id: string }>(
        `/drafts/${draftId}/check-compliance`,
        { method: "POST", body: JSON.stringify({}) },
      );
      return pollTaskResult<ComplianceReport>(task_id);
    },
  });
}

export function useExportDraft() {
  return useMutation({
    mutationFn: async ({
      draftId,
      format = "docx",
    }: {
      draftId: string;
      format?: string;
    }) => {
      const token = localStorage.getItem("access_token");
      const API_URL =
        process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";
      const res = await fetch(
        `${API_URL}/drafts/${draftId}/export?format=${format}`,
        { headers: { Authorization: `Bearer ${token}` } },
      );
      if (!res.ok) throw new Error("Export failed");
      const blob = await res.blob();
      return { blob, filename: `draft.${format}` };
    },
    onSuccess: ({ blob, filename }) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    },
  });
}
