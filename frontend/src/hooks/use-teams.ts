import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";

export interface TeamMember {
  id: string;
  user_id: string;
  email: string;
  name: string;
  role: string;
  joined_at: string;
}

export interface Team {
  id: string;
  name: string;
  description: string | null;
  created_by: string;
  members: TeamMember[];
  created_at: string;
}

export interface TeamListItem {
  id: string;
  name: string;
  description: string | null;
  member_count: number;
  created_at: string;
}

export function useTeams() {
  return useQuery<TeamListItem[]>({
    queryKey: ["teams"],
    queryFn: () => apiFetch("/teams"),
  });
}

export function useTeam(teamId: string) {
  return useQuery<Team>({
    queryKey: ["teams", teamId],
    queryFn: () => apiFetch(`/teams/${teamId}`),
    enabled: !!teamId,
  });
}

export function useCreateTeam() {
  const queryClient = useQueryClient();
  return useMutation<Team, Error, { name: string; description?: string }>({
    mutationFn: (data) =>
      apiFetch("/teams", { method: "POST", body: JSON.stringify(data) }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["teams"] });
    },
  });
}

export function useAddMember() {
  const queryClient = useQueryClient();
  return useMutation<void, Error, { teamId: string; email: string; role?: string }>({
    mutationFn: ({ teamId, ...data }) =>
      apiFetch(`/teams/${teamId}/members`, { method: "POST", body: JSON.stringify(data) }),
    onSuccess: (_, { teamId }) => {
      queryClient.invalidateQueries({ queryKey: ["teams", teamId] });
    },
  });
}

export function useRemoveMember() {
  const queryClient = useQueryClient();
  return useMutation<void, Error, { teamId: string; userId: string }>({
    mutationFn: ({ teamId, userId }) =>
      apiFetch(`/teams/${teamId}/members/${userId}`, { method: "DELETE" }),
    onSuccess: (_, { teamId }) => {
      queryClient.invalidateQueries({ queryKey: ["teams", teamId] });
    },
  });
}

export function useShareProject() {
  const queryClient = useQueryClient();
  return useMutation<void, Error, { teamId: string; projectId: string }>({
    mutationFn: ({ teamId, projectId }) =>
      apiFetch(`/teams/${teamId}/projects`, { method: "POST", body: JSON.stringify({ project_id: projectId }) }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["teams"] });
    },
  });
}
