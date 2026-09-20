import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";

export interface AdminUser {
  id: string;
  email: string;
  name: string;
  role: string;
  expertise_level: string;
  created_at: string;
  project_count: number;
}

export interface SystemStats {
  total_users: number;
  total_projects: number;
  total_papers: number;
  total_drafts: number;
  total_jobs: number;
}

export function useAdminUsers() {
  return useQuery<AdminUser[]>({
    queryKey: ["admin", "users"],
    queryFn: () => apiFetch("/admin/users"),
    retry: false,
  });
}

export function useAdminStats() {
  return useQuery<SystemStats>({
    queryKey: ["admin", "stats"],
    queryFn: () => apiFetch("/admin/stats"),
    retry: false,
  });
}

export function useUpdateUserRole() {
  const queryClient = useQueryClient();
  return useMutation<void, Error, { userId: string; role: string }>({
    mutationFn: ({ userId, role }) =>
      apiFetch(`/admin/users/${userId}/role`, {
        method: "PATCH",
        body: JSON.stringify({ role }),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "users"] });
    },
  });
}

export function useDeleteUser() {
  const queryClient = useQueryClient();
  return useMutation<void, Error, string>({
    mutationFn: (userId) =>
      apiFetch(`/admin/users/${userId}`, { method: "DELETE" }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "users"] });
    },
  });
}
