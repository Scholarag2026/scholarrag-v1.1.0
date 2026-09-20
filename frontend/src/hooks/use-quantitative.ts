import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";

// Types

export interface AnalysisMethod {
  method: string;
  justification: string;
  assumptions: string[];
  variables: string[];
}

export interface AnalysisPlan {
  methods: AnalysisMethod[];
  overall_strategy: string;
}

export interface NumericStats {
  column: string;
  mean: number;
  median: number;
  std_dev: number;
  min: number;
  max: number;
  skewness: number;
  kurtosis: number;
  n: number;
}

export interface CategoricalStats {
  column: string;
  frequencies: Record<string, number>;
  mode: string;
  n: number;
}

export interface DescriptiveStats {
  numeric: NumericStats[];
  categorical: CategoricalStats[];
}

export interface CorrelationEntry {
  var1: string;
  var2: string;
  r: number;
  p_value: number;
  significant: boolean;
}

export interface AssumptionCheck {
  test_name: string;
  variable: string;
  statistic: number;
  p_value: number;
  met: boolean;
  interpretation: string;
}

export interface ResultPrediction {
  hypothesis: string;
  predicted_outcome: string;
  confidence: "high" | "medium" | "low";
  rationale: string;
}

export interface CodeTemplate {
  language: "python" | "r" | "spss";
  code: string;
  description: string;
}

export interface InterpretationGuide {
  section: string;
  apa_example: string;
  notes: string;
}

export interface AnalysisResults {
  plan: AnalysisPlan;
  descriptive_stats: DescriptiveStats;
  correlations: CorrelationEntry[];
  assumption_checks: AssumptionCheck[];
  result_predictions: ResultPrediction[];
  code_templates: CodeTemplate[];
  interpretation_guides: InterpretationGuide[];
}

// Hooks

export function useQuantitativeResults(datasetId: string | null) {
  return useQuery<AnalysisResults>({
    queryKey: ["datasets", datasetId, "analysis", "quantitative"],
    queryFn: () => apiFetch(`/datasets/${datasetId}/analysis/quantitative`),
    enabled: !!datasetId,
    retry: false,
  });
}

export function useTriggerQuantitative() {
  const queryClient = useQueryClient();
  return useMutation<{ task_id: string }, Error, { datasetId: string }>({
    mutationFn: ({ datasetId }) =>
      apiFetch(`/datasets/${datasetId}/analyze/quantitative`, {
        method: "POST",
        body: JSON.stringify({}),
      }),
    onSuccess: (_, { datasetId }) => {
      queryClient.invalidateQueries({
        queryKey: ["datasets", datasetId, "analysis", "quantitative"],
      });
    },
  });
}
