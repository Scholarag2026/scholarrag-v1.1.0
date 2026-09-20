import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";

export interface Project {
  id: string;
  title: string;
  description: string | null;
  target_journal: string | null;
  citation_style: string;
  status: string;
  target_journal_guidelines: Record<string, unknown> | null;
  refined_topic: string | null;
  inclusion_criteria: string[] | null;
  exclusion_criteria: string[] | null;
  created_at: string;
  updated_at: string;
}

export function useProjects() {
  return useQuery<{ projects: Project[] }>({
    queryKey: ["projects"],
    queryFn: () => apiFetch("/projects"),
  });
}

export function useProject(id: string) {
  return useQuery<Project>({
    queryKey: ["projects", id],
    queryFn: () => apiFetch(`/projects/${id}`),
    enabled: !!id,
  });
}

export function useCreateProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: {
      title: string;
      description?: string;
      citation_style?: string;
      target_journal?: string;
    }) => apiFetch("/projects", { method: "POST", body: JSON.stringify(data) }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["projects"], exact: true }),
  });
}

export function useUpdateProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      id,
      ...data
    }: {
      id: string;
      title?: string;
      description?: string;
      target_journal?: string;
      citation_style?: string;
    }) =>
      apiFetch(`/projects/${id}`, {
        method: "PUT",
        body: JSON.stringify(data),
      }),
    onSuccess: (_data, variables) => {
      qc.invalidateQueries({ queryKey: ["projects"], exact: true });
      qc.invalidateQueries({ queryKey: ["projects", variables.id], exact: true });
    },
  });
}

export function useDeleteProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch(`/projects/${id}`, { method: "DELETE" }),
    onSuccess: (_data, id) => {
      qc.invalidateQueries({ queryKey: ["projects"], exact: true });
      qc.removeQueries({ queryKey: ["projects", id] });
    },
  });
}
