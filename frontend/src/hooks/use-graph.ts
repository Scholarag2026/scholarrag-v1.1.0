import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";

export interface GraphNode {
  id: string;
  title: string;
  authors: string[];
  year: number | null;
  citation_count: number | null;
  quality_score: number | null;
  in_library: boolean;
  doi: string | null;
  abstract: string | null;
  is_seed?: boolean;
  doi_score?: number | null;
  cluster_id?: number | null;  // Louvain community ID
}

export interface GraphEdge {
  source: string;
  target: string;
}

export interface GraphBuildSummary {
  status: string;
  edges_created: number;
  papers_processed: number;
  papers_failed: number;
  citations_unavailable: number;
  error: string | null;
  completed_at: string | null;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  /** Outcome of the most recent build; null/undefined means the graph was never built. */
  last_build?: GraphBuildSummary | null;
}

export interface GraphExpansion {
  new_nodes: GraphNode[];
  new_edges: GraphEdge[];
}

export function useCitationGraph(projectId: string, focusNodeId?: string | null) {
  return useQuery<GraphData>({
    queryKey: ["projects", projectId, "citation-graph"],  // stable key — no focusNodeId!
    queryFn: () => {
      const params = focusNodeId ? `?focus_id=${focusNodeId}` : "";
      return apiFetch(`/projects/${projectId}/citation-graph${params}`);
    },
    enabled: !!projectId,
    // D8: the client-merged expansion set is authoritative until an explicit rebuild.
    // Any background refetch re-ranks against a focus-dependent 80-node cap on the
    // server and silently drops nodes the user already expanded, so the ONLY thing
    // allowed to invalidate this query is a completed build (see graph-tab.tsx).
    staleTime: Infinity,
    // GraphTab is rendered from a view-driven switch, so switching tabs unmounts it.
    // The app-level 30-min gcTime (providers.tsx) would drop the cache entry while
    // away, discarding the client-merged expansion set. One hour keeps expansions
    // through normal tab-switching while remaining a finite escape hatch: if a build
    // completes while GraphTab is unmounted (buildTaskId is component-local state,
    // so its completed-build invalidation never fires), the stale pre-build cache
    // still expires instead of being served forever.
    gcTime: 60 * 60 * 1000,
    refetchOnMount: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    retry: false,
    placeholderData: (prev) => prev, // Show previous data while refetching
  });
}

export function useBuildGraph() {
  // No onSuccess invalidation: a 202 means "build queued", not "graph changed".
  // Refetching here would throw away the user's expanded node set (D8); graph-tab.tsx
  // invalidates once the build job actually reports completed.
  return useMutation<{ task_id: string }, Error, { projectId: string }>({
    mutationFn: ({ projectId }) =>
      apiFetch(`/projects/${projectId}/build-graph`, {
        method: "POST",
        body: JSON.stringify({}),
      }),
  });
}

export function useExpandNode() {
  const queryClient = useQueryClient();
  return useMutation<
    GraphExpansion,
    Error,
    { projectId: string; paperId: string }
  >({
    mutationFn: ({ projectId, paperId }) =>
      apiFetch(`/projects/${projectId}/citation-graph/expand`, {
        method: "POST",
        body: JSON.stringify({ paper_id: paperId }),
      }),
    onSuccess: (expansion, { projectId }) => {
      queryClient.setQueryData<GraphData>(
        ["projects", projectId, "citation-graph"],
        (old) => {
          if (!old) return old;
          const existingIds = new Set(old.nodes.map((n) => n.id));
          const newNodes = expansion.new_nodes.filter(
            (n) => !existingIds.has(n.id),
          );
          const existingEdges = new Set(
            old.edges.map((e) => `${e.source}->${e.target}`),
          );
          const newEdges = expansion.new_edges.filter(
            (e) => !existingEdges.has(`${e.source}->${e.target}`),
          );
          return {
            ...old,
            nodes: [...old.nodes, ...newNodes],
            edges: [...old.edges, ...newEdges],
          };
        },
      );
    },
  });
}
