import { useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch, fetchWithAuth } from "@/lib/api";

// ── Types ────────────────────────────────────────────────────────────────────

export interface ReviewPaper {
  index: number;
  file_name: string;
  title: string;
  authors: { name: string }[];
  year: number | null;
  journal_name: string | null;
  doi: string | null;
  abstract: string | null;
  source_api: string;
  has_full_text: boolean;
}

// ── Hooks ────────────────────────────────────────────────────────────────────

/**
 * Upload article PDF files for a project.
 * Uses raw fetch (not apiFetch) because FormData requires the browser
 * to set the Content-Type with the multipart boundary automatically.
 */
export function useUploadArticles() {
  return useMutation<
    { task_id: string },
    Error,
    { projectId: string; files: File[] }
  >({
    mutationFn: async ({ projectId, files }) => {
      const formData = new FormData();
      files.forEach((file) => formData.append("files", file));

      const API_URL =
        process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

      const res = await fetchWithAuth(
        `${API_URL}/projects/${projectId}/papers/upload`,
        {
          method: "POST",
          body: formData,
        },
      );

      if (!res.ok) {
        const error = await res
          .json()
          .catch(() => ({ detail: "Upload failed" }));
        throw new Error(error.detail || "Upload failed");
      }

      return res.json();
    },
  });
}

export function useConfirmUpload(projectId: string) {
  const qc = useQueryClient();
  return useMutation<
    { imported_count: number },
    Error,
    {
      taskId: string;
      papers: Omit<
        ReviewPaper,
        "file_name" | "source_api" | "has_full_text"
      >[];
    }
  >({
    mutationFn: ({ taskId, papers }) =>
      apiFetch(`/projects/${projectId}/papers/upload-confirm`, {
        method: "POST",
        body: JSON.stringify({ task_id: taskId, papers }),
      }),
    onSuccess: () =>
      qc.invalidateQueries({
        queryKey: ["projects", projectId, "papers"],
      }),
  });
}
