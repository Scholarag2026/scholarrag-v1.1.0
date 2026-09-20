import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";

// Types

export interface CodeDefinition {
  code: string;
  definition: string;
  examples: string[];
}

export interface Theme {
  theme: string;
  description: string;
  codes: CodeDefinition[];
}

export interface CodebookSchema {
  themes: Theme[];
  created_at?: string;
  updated_at?: string;
}

export interface CodingSegment {
  segment_id: string;
  text: string;
  codes: string[];
  confidence: number;
  source_row: number;
  source_column: string;
  notes?: string;
}

export interface CodingResult {
  codebook: CodebookSchema;
  segments: CodingSegment[];
  total_segments: number;
  reviewed_segments: number;
  uncertainties: CodingUncertainty[];
  annotation_notes: string[];
}

export interface CodingUncertainty {
  segment_id: string;
  text: string;
  candidate_codes: string[];
  reason: string;
}

export interface PerCodeKappa {
  code_name: string;
  /** null when the kappa is mathematically undefined (e.g. neither coder used the code). */
  kappa: number | null;
  agreement_pct: number;
}

export interface DisagreementItem {
  segment_id: string;
  ai_codes: string[];
  human_codes: string[];
}

export interface InterCoderReport {
  /** null when Cohen's kappa is not computable — render distinctly from a real 0.0. */
  overall_kappa: number | null;
  overall_kappa_explanation: string | null;
  agreement_pct: number;
  per_code_kappa: PerCodeKappa[];
  disagreement_segments: DisagreementItem[];
}

// Hooks

export function useQualitativeResults(datasetId: string | null) {
  return useQuery<CodingResult>({
    queryKey: ["datasets", datasetId, "analysis", "qualitative"],
    queryFn: () => apiFetch(`/datasets/${datasetId}/analysis/qualitative`),
    enabled: !!datasetId,
    retry: false,
  });
}

export function useTriggerQualitative() {
  const queryClient = useQueryClient();
  return useMutation<{ task_id: string }, Error, { datasetId: string }>({
    mutationFn: ({ datasetId }) =>
      apiFetch(`/datasets/${datasetId}/analyze/qualitative`, {
        method: "POST",
        body: JSON.stringify({}),
      }),
    onSuccess: (_, { datasetId }) => {
      queryClient.invalidateQueries({
        queryKey: ["datasets", datasetId, "analysis", "qualitative"],
      });
    },
  });
}

export function useUpdateCodebook() {
  const queryClient = useQueryClient();
  return useMutation<CodebookSchema, Error, { datasetId: string; codebook: CodebookSchema }>({
    mutationFn: ({ datasetId, codebook }) =>
      apiFetch(`/datasets/${datasetId}/codebook`, {
        method: "PUT",
        body: JSON.stringify(codebook),
      }),
    onSuccess: (_, { datasetId }) => {
      queryClient.invalidateQueries({
        queryKey: ["datasets", datasetId, "analysis", "qualitative"],
      });
    },
  });
}

export function useSaveCodingSession() {
  const queryClient = useQueryClient();
  return useMutation<
    void,
    Error,
    { datasetId: string; segments: CodingSegment[] }
  >({
    mutationFn: ({ datasetId, segments }) =>
      apiFetch(`/datasets/${datasetId}/coding-session`, {
        method: "PUT",
        body: JSON.stringify({ segments }),
      }),
    onSuccess: (_, { datasetId }) => {
      queryClient.invalidateQueries({
        queryKey: ["datasets", datasetId, "analysis", "qualitative"],
      });
    },
  });
}

export function useInterCoderReport(datasetId: string | null) {
  return useQuery<InterCoderReport>({
    queryKey: ["datasets", datasetId, "inter-coder-reliability"],
    queryFn: () => apiFetch(`/datasets/${datasetId}/inter-coder-reliability`),
    enabled: !!datasetId,
    retry: false,
  });
}

export function useTriggerInterCoder() {
  const queryClient = useQueryClient();
  return useMutation<{ task_id: string }, Error, { datasetId: string }>({
    mutationFn: ({ datasetId }) =>
      apiFetch(`/datasets/${datasetId}/inter-coder-reliability`, {
        method: "POST",
        body: JSON.stringify({}),
      }),
    onSuccess: (_, { datasetId }) => {
      queryClient.invalidateQueries({
        queryKey: ["datasets", datasetId, "inter-coder-reliability"],
      });
    },
  });
}
