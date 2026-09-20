import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";

export interface MonthCount {
  month: string;
  count: number;
}

export interface UserAnalytics {
  total_papers: number;
  total_drafts: number;
  total_searches: number;
  total_exports: number;
  papers_by_month: MonthCount[];
  papers_by_source: Record<string, number>;
}

export function useMyAnalytics() {
  return useQuery<UserAnalytics>({
    queryKey: ["analytics", "me"],
    queryFn: () => apiFetch("/analytics/me"),
    retry: false,
  });
}
