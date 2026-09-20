import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch, fetchWithAuth } from "@/lib/api";

// Types

export interface ColumnInfo {
  name: string;
  dtype: "numeric" | "categorical" | "text" | "datetime" | "boolean";
  missing_count: number;
  missing_pct: number;
  unique_count: number;
  sample_values: string[];
  role?: "independent" | "dependent" | "control" | "participant_id" | "text_data";
}

export interface DatasetUploadResponse {
  id: string;
  project_id: string;
  filename: string;
  row_count: number;
  column_count: number;
  columns: ColumnInfo[];
  available_sheets?: string[];
  selected_sheet?: string;
}

export interface DatasetDetail {
  id: string;
  project_id: string;
  filename: string;
  row_count: number;
  column_count: number;
  columns: ColumnInfo[];
  available_sheets?: string[];
  selected_sheet?: string;
  preview_rows: Record<string, unknown>[];
  created_at: string;
  updated_at: string;
}

export interface DatasetListItem {
  id: string;
  project_id: string;
  filename: string;
  row_count: number;
  column_count: number;
  created_at: string;
}

// Hooks

export function useDatasets(projectId: string) {
  return useQuery<DatasetListItem[]>({
    queryKey: ["projects", projectId, "datasets"],
    queryFn: () => apiFetch(`/projects/${projectId}/datasets`),
    enabled: !!projectId,
  });
}

export function useDatasetDetail(datasetId: string | null) {
  return useQuery<DatasetDetail>({
    queryKey: ["datasets", datasetId],
    queryFn: () => apiFetch(`/datasets/${datasetId}`),
    enabled: !!datasetId,
    retry: false,
  });
}

export function useUploadDataset() {
  const queryClient = useQueryClient();
  return useMutation<
    DatasetUploadResponse,
    Error,
    { projectId: string; file: File; sheet?: string }
  >({
    mutationFn: async ({ projectId, file, sheet }) => {
      const formData = new FormData();
      formData.append("file", file);
      if (sheet) formData.append("sheet", sheet);

      const API_URL =
        process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

      const res = await fetchWithAuth(`${API_URL}/projects/${projectId}/datasets`, {
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
      queryClient.invalidateQueries({
        queryKey: ["projects", projectId, "datasets"],
      });
    },
  });
}

export function useUpdateDataset() {
  const queryClient = useQueryClient();
  return useMutation<
    DatasetDetail,
    Error,
    { datasetId: string; columns?: ColumnInfo[]; selected_sheet?: string }
  >({
    mutationFn: ({ datasetId, ...body }) =>
      apiFetch(`/datasets/${datasetId}`, {
        method: "PUT",
        body: JSON.stringify(body),
      }),
    onSuccess: (data) => {
      queryClient.invalidateQueries({
        queryKey: ["datasets", data.id],
      });
    },
  });
}

export function useDeleteDataset() {
  const queryClient = useQueryClient();
  return useMutation<void, Error, { datasetId: string; projectId: string }>({
    mutationFn: ({ datasetId }) =>
      apiFetch(`/datasets/${datasetId}`, { method: "DELETE" }),
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({
        queryKey: ["projects", projectId, "datasets"],
      });
    },
  });
}
