import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";

// Types for ResearchDesign
export interface MethodologyPlan {
  approach: "qualitative" | "quantitative" | "mixed_methods";
  design_type: string;
  justification: string;
  research_questions: string[];
  theoretical_framework: string;
}

export interface EthicsPlan {
  considerations: string[];
  consent_template: string;
  data_protection: string;
  irb_notes: string;
}

export interface Instrument {
  name: string;
  type: "survey" | "interview" | "observation" | "focus_group" | "document_analysis";
  description: string;
  sample_questions: string[];
  administration: string;
}

export interface SamplingStrategy {
  strategy_type: string;
  target_population: string;
  sample_size: number;
  justification: string;
  recruitment_plan: string;
  inclusion_criteria: string[];
  exclusion_criteria: string[];
}

export interface ValidityPlan {
  internal_validity: string[];
  external_validity: string[];
  reliability_measures: string[];
  triangulation: string;
}

export interface ResearchDesign {
  methodology: MethodologyPlan;
  ethics: EthicsPlan;
  instruments: Instrument[];
  sampling: SamplingStrategy;
  validity: ValidityPlan;
}

// Types for DataCollectionPlan
export interface CollectionPhase {
  phase_number: number;
  name: string;
  description: string;
  steps: string[];
  duration: string;
  deliverables: string[];
}

export interface QualityCheck {
  check_name: string;
  when: string;
  how: string;
  action_if_failed: string;
}

export interface DataStoragePlan {
  storage_method: string;
  backup_strategy: string;
  access_control: string;
  anonymization: string;
  retention_period: string;
}

export interface TimelineItem {
  week: string;
  activity: string;
  milestone: string | null;
}

export interface DataCollectionPlan {
  phases: CollectionPhase[];
  quality_checks: QualityCheck[];
  data_storage: DataStoragePlan;
  timeline: TimelineItem[];
  ethical_reminders: string[];
}

// Hooks
export function useResearchDesign(projectId: string) {
  return useQuery<ResearchDesign>({
    queryKey: ["projects", projectId, "research-design"],
    queryFn: () => apiFetch(`/projects/${projectId}/research-design`),
    enabled: !!projectId,
    retry: false,
  });
}

export function useGenerateDesign() {
  const queryClient = useQueryClient();
  return useMutation<{ task_id: string }, Error, { projectId: string; researchQuestion: string }>({
    mutationFn: ({ projectId, researchQuestion }) =>
      apiFetch(`/projects/${projectId}/research-design`, {
        method: "POST",
        body: JSON.stringify({ research_question: researchQuestion }),
      }),
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({ queryKey: ["projects", projectId, "research-design"] });
    },
  });
}

export function useCollectionPlan(projectId: string) {
  return useQuery<DataCollectionPlan>({
    queryKey: ["projects", projectId, "data-collection-plan"],
    queryFn: () => apiFetch(`/projects/${projectId}/data-collection-plan`),
    enabled: !!projectId,
    retry: false,
  });
}

export function useGenerateCollectionPlan() {
  const queryClient = useQueryClient();
  return useMutation<{ task_id: string }, Error, { projectId: string }>({
    mutationFn: ({ projectId }) =>
      apiFetch(`/projects/${projectId}/data-collection-plan`, {
        method: "POST",
        body: JSON.stringify({}),
      }),
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({ queryKey: ["projects", projectId, "data-collection-plan"] });
    },
  });
}
