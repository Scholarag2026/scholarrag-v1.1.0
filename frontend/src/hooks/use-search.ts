// frontend/src/hooks/use-search.ts
import { useCallback, useEffect, useState } from "react";

import { apiFetch } from "@/lib/api";
import { useTaskPolling } from "@/hooks/use-task-polling";

export interface SearchPaper {
  doi: string | null;
  title: string;
  authors: { name: string }[];
  year: number | null;
  journal_name: string | null;
  journal_issn?: string | null;
  citation_count: number | null;
  abstract: string | null;
  source_api: string;
  external_id: string | null;
  full_text_url: string | null;
  wos_collection?: string | null;
}

interface SearchTaskStatus {
  status: string;
  progress: number;
  progress_message: string | null;
  result: {
    papers: SearchPaper[];
    total: number;
    sources: Record<string, number>;
  } | null;
  error: string | null;
}

export function useSearch(projectId: string) {
  const [taskId, setTaskId] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [results, setResults] = useState<SearchPaper[]>([]);
  const [sources, setSources] = useState<Record<string, number>>({});
  const [error, setError] = useState<string | null>(null);

  const {
    data: task,
    timedOut,
    isCancelling,
    maxDurationMinutes,
    cancel,
    resume,
  } = useTaskPolling<SearchTaskStatus>({
    taskId,
    kind: "basic_search",
    onCancelled: () => setTaskId(null),
  });

  useEffect(() => {
    if (!taskId || !task) return;
    if (task.status === "completed") {
      setResults(task.result?.papers ?? []);
      setSources(task.result?.sources ?? {});
      setError(null);
      setTaskId(null);
    } else if (task.status === "failed") {
      setError(task.error || "Search failed");
      setTaskId(null);
    } else if (task.status === "cancelled") {
      setTaskId(null);
    }
  }, [task, taskId]);

  const search = useCallback(
    async (params: {
      query: string;
      year_from?: number;
      year_to?: number;
      min_citations?: number;
    }) => {
      setResults([]);
      setSources({});
      setError(null);
      setIsStarting(true);
      try {
        const { task_id } = await apiFetch<{ task_id: string }>(
          `/projects/${projectId}/search`,
          { method: "POST", body: JSON.stringify(params) },
        );
        setTaskId(task_id);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Search request failed");
      } finally {
        setIsStarting(false);
      }
    },
    [projectId],
  );

  const clearResults = useCallback(() => {
    setTaskId(null);
    setResults([]);
    setSources({});
    setError(null);
  }, []);

  const isSearching = isStarting || (!!taskId && !timedOut);

  return {
    isSearching,
    progress: task?.progress ?? 0,
    progressMessage: task?.progress_message ?? null,
    results,
    sources,
    error,
    timedOut,
    isCancelling,
    maxDurationMinutes,
    cancel,
    resume,
    search,
    clearResults,
  };
}
