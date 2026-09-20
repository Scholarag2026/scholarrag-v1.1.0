"use client";

import { Button } from "@/components/ui/button";
import { CitationGraph } from "@/components/citation-graph";
import { EmptyState } from "@/components/empty-state";
import { useBuildGraph, useCitationGraph, type GraphData, type GraphNode } from "@/hooks/use-graph";
import { useProjectPapers } from "@/hooks/use-papers";
import { apiFetch } from "@/lib/api";
import { useGraphNavigation } from "@/hooks/use-graph-navigation";
import { GraphMinimap } from "@/components/graph-minimap";
import { GraphLegend, type ColorMode } from "@/components/graph-legend";
import { ChevronLeft, ChevronRight, Crosshair, Loader2, Maximize2, Search, Share2, X } from "lucide-react";
import { useTranslations } from "next-intl";
import { toast } from "sonner";
import { useQueryClient } from "@tanstack/react-query";
import { useTaskPolling } from "@/hooks/use-task-polling";
import { useState, useEffect, useCallback, useRef } from "react";

interface GraphTabProps {
  projectId: string;
}

export function GraphTab({ projectId }: GraphTabProps) {
  const t = useTranslations("graph");
  const tEmpty = useTranslations("emptyStates");
  const [buildTaskId, setBuildTaskId] = useState<string | null>(null);
  const [focusNodeId, setFocusNodeId] = useState<string | null>(null);
  const { data: graphData, isLoading } = useCitationGraph(projectId, focusNodeId);
  const { data: papersData } = useProjectPapers(projectId);
  const paperCount = papersData?.total ?? 0;
  const buildGraph = useBuildGraph();
  const queryClient = useQueryClient();
  const nav = useGraphNavigation();
  const [colorMode, setColorMode] = useState<ColorMode>("year");
  const [searchQuery, setSearchQuery] = useState("");
  const [minimapNodes, setMinimapNodes] = useState<Array<{ x: number; y: number; data: GraphNode }>>([]);
  const controlsRef = useRef<{ fitAll: () => void; centerOnFocus: () => void; navigateTo: (x: number, y: number) => void } | null>(null);

  const handleFocusChange = useCallback((nodeId: string, nodeTitle: string) => {
    setFocusNodeId(nodeId);
    nav.push(nodeId, nodeTitle);
  }, [nav]);

  // Poll task status while building via the shared polling hook (bounded, backoff, cancel).
  // TaskResponse already serves result + error — read them so a failed or degraded
  // build can be reported (issue #3).
  const { data: taskStatus, timedOut } = useTaskPolling<{
    status: string;
    progress: number;
    progress_message: string | null;
    result: {
      edges_created?: number;
      papers_processed?: number;
      papers_failed?: number;
      citations_unavailable?: number;
    } | null;
    error: string | null;
  }>({
    taskId: buildTaskId,
    kind: "graph_build",
  });

  // When the build reaches a terminal state, tell the user what happened and — only on
  // a completed build — refresh the graph (D8: nothing else may invalidate it).
  useEffect(() => {
    if (!buildTaskId) return;
    if (taskStatus?.status === "completed") {
      const edges = taskStatus.result?.edges_created ?? 0;
      const citesMissing = taskStatus.result?.citations_unavailable ?? 0;
      queryClient.invalidateQueries({
        queryKey: ["projects", projectId, "citation-graph"],
      });
      if (citesMissing > 0) {
        toast.warning(t("buildCompletePartial", { edges, papers: citesMissing }));
      } else if (edges > 0) {
        toast.success(t("buildComplete", { edges }));
      } else {
        toast.warning(t("buildCompleteEmpty"));
      }
      setBuildTaskId(null);
    }
    if (taskStatus?.status === "failed") {
      toast.error(taskStatus.error || t("buildFailed"));
      setBuildTaskId(null);
    }
    if (taskStatus?.status === "cancelled") {
      setBuildTaskId(null);
    }
  }, [
    taskStatus?.status,
    taskStatus?.result,
    taskStatus?.error,
    buildTaskId,
    queryClient,
    projectId,
    t,
  ]);

  const isBuilding =
    !!buildTaskId &&
    !timedOut &&
    taskStatus?.status !== "completed" &&
    taskStatus?.status !== "failed" &&
    taskStatus?.status !== "cancelled";

  const handleAddToLibrary = async (node: GraphNode) => {
    if (!node.doi) return;
    try {
      // The endpoint requires full paper_data ({doi} alone is 400-rejected);
      // build it from the graph node, mirroring use-papers.ts.
      await apiFetch(`/projects/${projectId}/papers`, {
        method: "POST",
        body: JSON.stringify({
          paper_data: {
            doi: node.doi,
            title: node.title,
            authors: node.authors.map((name) => ({ name })),
            year: node.year,
            citation_count: node.citation_count,
            abstract: node.abstract,
            source_api: "openalex",
          },
        }),
      });
      // D8: patch the added node locally instead of invalidating — an invalidation
      // would refetch the server's re-ranked top-N set and silently drop every
      // client-merged node above the cap (see use-graph.ts).
      queryClient.setQueryData<GraphData>(
        ["projects", projectId, "citation-graph"],
        (old) => {
          if (!old) return old;
          return {
            ...old,
            nodes: old.nodes.map((n) =>
              n.id === node.id ? { ...n, in_library: true, is_seed: true } : n,
            ),
          };
        },
      );
      queryClient.invalidateQueries({
        queryKey: ["projects", projectId, "papers"],
      });
    } catch (err) {
      toast.error(
        err instanceof Error && err.message
          ? err.message
          : t("addToLibraryFailed"),
      );
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-6 w-6 animate-spin text-[var(--ds-text-secondary)]" />
      </div>
    );
  }

  const hasGraph =
    graphData && graphData.nodes.length > 0;

  const lastBuild = graphData?.last_build ?? null;
  const buildFailed = !hasGraph && lastBuild?.status === "failed";
  const builtButEmpty =
    !hasGraph &&
    lastBuild?.status === "completed" &&
    lastBuild.edges_created === 0 &&
    !!papersData &&
    paperCount > 0;
  const isDegraded = buildFailed || builtButEmpty;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-[var(--ds-text-heading)]">
            {t("title")}
          </h2>
        </div>
        <Button
          onClick={() =>
            buildGraph.mutate(
              { projectId },
              {
                onSuccess: (data) => {
                  setBuildTaskId(data.task_id);
                  toast.info(t("building"));
                },
                onError: (err) => toast.error(err.message || t("buildFailed")),
              },
            )
          }
          disabled={buildGraph.isPending || isBuilding}
          className="gap-2"
        >
          {buildGraph.isPending || isBuilding ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Share2 className="h-4 w-4" />
          )}
          {isBuilding
            ? taskStatus?.progress_message || t("building")
            : t("buildGraph")}
        </Button>
      </div>

      {nav.history.length > 0 && (
        <div className="flex items-center gap-1 text-sm overflow-x-auto py-1">
          <Button
            variant="ghost" size="icon" className="h-6 w-6 shrink-0"
            disabled={!nav.canGoBack}
            onClick={() => {
              const entry = nav.goBack();
              if (entry) setFocusNodeId(entry.nodeId);
            }}
          >
            <ChevronLeft className="h-3.5 w-3.5" />
          </Button>
          {nav.history.map((entry, i) => (
            <span key={`${entry.nodeId}-${i}`} className="flex items-center gap-1 shrink-0">
              {i > 0 && <span className="text-[var(--ds-text-muted)]">/</span>}
              <button
                className={`hover:underline truncate max-w-[160px] ${
                  i === nav.currentIndex
                    ? "text-[var(--ds-primary)] font-medium"
                    : "text-[var(--ds-text-secondary)]"
                }`}
                title={entry.title}
                onClick={() => {
                  const jumped = nav.jumpTo(i);
                  if (jumped) setFocusNodeId(jumped.nodeId);
                }}
              >
                {entry.title.length > 25 ? entry.title.slice(0, 25) + "..." : entry.title}
              </button>
            </span>
          ))}
          <Button
            variant="ghost" size="icon" className="h-6 w-6 shrink-0"
            disabled={!nav.canGoForward}
            onClick={() => {
              const entry = nav.goForward();
              if (entry) setFocusNodeId(entry.nodeId);
            }}
          >
            <ChevronRight className="h-3.5 w-3.5" />
          </Button>
        </div>
      )}

      {hasGraph ? (
        <div className="relative">
          {/* Toolbar */}
          <div className="absolute top-3 left-3 right-3 z-10 flex items-start justify-between pointer-events-none">
            {/* Color mode tabs */}
            <div className="pointer-events-auto flex rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-card)]/95 p-0.5 gap-0.5 backdrop-blur-sm">
              {([
                { mode: "year" as const, label: t("colorModeYear") },
                { mode: "cluster" as const, label: t("colorModeCluster") },
                { mode: "quality" as const, label: t("colorModeQuality") },
              ]).map(({ mode, label }) => (
                <button
                  key={mode}
                  onClick={() => setColorMode(mode)}
                  className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                    colorMode === mode
                      ? "bg-[var(--ds-primary)] text-white"
                      : "text-[var(--ds-text-secondary)] hover:bg-[var(--ds-bg-subtle)]"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>

            {/* Right side: search + tools */}
            <div className="pointer-events-auto flex items-center gap-2">
              <div className="flex items-center gap-1.5 rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-card)]/95 px-2.5 h-8 backdrop-blur-sm">
                <Search className="h-3.5 w-3.5 text-[var(--ds-text-secondary)]" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder={t("searchPlaceholder")}
                  className="w-36 border-none bg-transparent text-xs outline-none placeholder:text-[var(--ds-text-muted)]"
                />
                {searchQuery && (
                  <button onClick={() => setSearchQuery("")} className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]">
                    <X className="h-3 w-3" />
                  </button>
                )}
              </div>
              <div className="flex rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-card)]/95 p-0.5 gap-0.5 backdrop-blur-sm">
                <Button variant="ghost" size="icon" className="h-7 w-7" title={t("zoomToFit")}
                  onClick={() => controlsRef.current?.fitAll()}>
                  <Maximize2 className="h-3.5 w-3.5" />
                </Button>
                <Button variant="ghost" size="icon" className="h-7 w-7" title={t("centerOnFocus")}
                  onClick={() => controlsRef.current?.centerOnFocus()}
                  disabled={!focusNodeId}>
                  <Crosshair className="h-3.5 w-3.5" />
                </Button>
              </div>
            </div>
          </div>

          {/* Stats bar */}
          <div className="absolute top-12 left-3 z-10 rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-card)]/92 px-3 py-1.5 text-xs text-[var(--ds-text-secondary)] backdrop-blur-sm">
            {t("stats", { nodes: graphData.nodes.length, edges: graphData.edges.length })}
          </div>

          <CitationGraph
            data={graphData}
            projectId={projectId}
            onAddToLibrary={handleAddToLibrary}
            focusNodeId={focusNodeId}
            onFocusChange={handleFocusChange}
            onPositionsUpdate={setMinimapNodes}
            onRegisterControls={(c) => { controlsRef.current = c; }}
            colorMode={colorMode}
            searchQuery={searchQuery}
          />

          <GraphLegend
            colorMode={colorMode}
            yearRange={(() => {
              const years = graphData.nodes.map(n => n.year).filter((y): y is number => y != null);
              return years.length ? [Math.min(...years), Math.max(...years)] as [number, number] : [2000, 2025] as [number, number];
            })()}
            clusterCount={new Set(graphData.nodes.map(n => n.cluster_id).filter(c => c != null)).size}
          />

          {/* Minimap */}
          {minimapNodes.length > 0 && (
            <GraphMinimap
              nodes={minimapNodes}
              focusNodeId={focusNodeId}
              viewport={null}
              onNavigate={(x, y) => controlsRef.current?.navigateTo(x, y)}
              colorMode={colorMode}
              yearRange={(() => {
                const years = graphData.nodes.map(n => n.year).filter((y): y is number => y != null);
                return years.length ? [Math.min(...years), Math.max(...years)] as [number, number] : [2000, 2025] as [number, number];
              })()}
            />
          )}
        </div>
      ) : (
        <EmptyState
          icon={isDegraded ? "⚠️" : "🕸️"}
          title={t("title")}
          description={
            buildFailed
              ? lastBuild?.error || t("buildFailed")
              : builtButEmpty
                ? (lastBuild?.citations_unavailable ?? 0) > 0
                  ? t("emptyThrottled")
                  : t("emptyDegraded")
                : tEmpty("citationGraph.description")
          }
          prerequisites={
            isDegraded
              ? undefined
              : [
                  {
                    label: tEmpty("prerequisites.atLeastPapers", { count: 3 }),
                    completed: paperCount >= 3,
                    current: paperCount,
                    total: 3,
                  },
                ]
          }
          actionLabel={isDegraded ? t("retryBuild") : t("buildGraph")}
          onAction={() =>
            buildGraph.mutate(
              { projectId },
              {
                onSuccess: (data) => {
                  setBuildTaskId(data.task_id);
                  toast.info(t("building"));
                },
                onError: (err) => toast.error(err.message || t("buildFailed")),
              },
            )
          }
          actionDisabled={
            (!isDegraded && paperCount < 3) || buildGraph.isPending || isBuilding
          }
        />
      )}
    </div>
  );
}
