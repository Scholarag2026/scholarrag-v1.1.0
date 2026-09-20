"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { toast } from "sonner";

import { apiFetch, cancelTask } from "@/lib/api";

const MINUTE = 60_000;

/** Every background job type the UI polls via `GET /tasks/{id}`. */
export type PollJobKind =
  | "basic_search"
  | "smart_search"
  | "deep_search"
  | "graph_build"
  | "seed_expand"
  | "field_foundations"
  | "full_text"
  | "deep_analysis"
  | "verification"
  | "quantitative"
  | "qualitative"
  | "inter_coder"
  | "generate_section"
  | "refine"
  | "research_design"
  | "collection_plan"
  | "article_extract";

/**
 * Max wait per job type. Smart/deep search get 45 min (both are multi-round search jobs;
 * production smart searches have been observed at 31.9 min), graph build 30 min, everything
 * else 10 min. When the budget is spent we stop polling and tell the user, instead of spinning
 * forever against a job that may be dead.
 */
export const POLL_MAX_DURATION_MS: Record<PollJobKind, number> = {
  basic_search: 10 * MINUTE,
  smart_search: 45 * MINUTE,
  deep_search: 45 * MINUTE,
  graph_build: 30 * MINUTE,
  seed_expand: 10 * MINUTE,
  field_foundations: 10 * MINUTE,
  full_text: 10 * MINUTE,
  deep_analysis: 10 * MINUTE,
  verification: 10 * MINUTE,
  quantitative: 10 * MINUTE,
  qualitative: 10 * MINUTE,
  inter_coder: 10 * MINUTE,
  generate_section: 10 * MINUTE,
  refine: 10 * MINUTE,
  research_design: 10 * MINUTE,
  collection_plan: 10 * MINUTE,
  article_extract: 10 * MINUTE,
};

/** Backoff ladder: 2s for the first minute, 5s up to 5 min, 10s after that. */
export function pollIntervalMs(elapsedMs: number): number {
  if (elapsedMs >= 5 * MINUTE) return 10_000;
  if (elapsedMs >= MINUTE) return 5_000;
  return 2_000;
}

/** `cancelled` is a terminal state, alongside completed/failed. */
export function isTerminalStatus(status: string | undefined): boolean {
  return status === "completed" || status === "failed" || status === "cancelled";
}

export interface TaskStatusBase {
  status: string;
}

export interface UseTaskPollingOptions {
  /** Task id to poll, or null when no job is in flight. */
  taskId: string | null;
  /** Selects the max-duration budget. */
  kind: PollJobKind;
  /** Called once when the budget is exhausted (after the toast has been shown). */
  onTimeout?: () => void;
  /** Called once after a successful cancel. Callers should clear their taskId here. */
  onCancelled?: () => void;
}

export interface UseTaskPollingResult<T> {
  data: T | undefined;
  error: Error | null;
  isPolling: boolean;
  timedOut: boolean;
  isCancelling: boolean;
  maxDurationMinutes: number;
  cancel: () => void;
  resume: () => void;
}

/**
 * Single source of truth for job polling.
 *
 * - `retry: false` — a failed tick must not trigger v5's silent 3-retry burst. The interval
 *   itself keeps running, so a transient blip self-heals on the next tick.
 * - backoff 2s -> 5s -> 10s via `pollIntervalMs`.
 * - a per-kind max duration; on expiry polling stops, `timedOut` flips and a toast with a
 *   "Keep watching" action appears.
 * - `cancel()` calls DELETE /api/v1/tasks/{id} and then stops polling.
 */
export function useTaskPolling<T extends TaskStatusBase>({
  taskId,
  kind,
  onTimeout,
  onCancelled,
}: UseTaskPollingOptions): UseTaskPollingResult<T> {
  const t = useTranslations("polling");
  const maxDurationMs = POLL_MAX_DURATION_MS[kind];
  const maxDurationMinutes = Math.round(maxDurationMs / MINUTE);

  const startedAtRef = useRef<number>(Date.now());
  const [stopped, setStopped] = useState(false);
  const [timedOut, setTimedOut] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);

  // Keep the latest callbacks without making the effects depend on their identity.
  const onTimeoutRef = useRef(onTimeout);
  const onCancelledRef = useRef(onCancelled);
  useEffect(() => {
    onTimeoutRef.current = onTimeout;
    onCancelledRef.current = onCancelled;
  }, [onTimeout, onCancelled]);

  const query = useQuery<T>({
    queryKey: ["tasks", taskId],
    queryFn: () => apiFetch<T>(`/tasks/${taskId}`),
    enabled: !!taskId && !stopped,
    retry: false,
    refetchInterval: (q) => {
      if (isTerminalStatus(q.state.data?.status)) return false;
      const elapsed = Date.now() - startedAtRef.current;
      if (elapsed >= maxDurationMs) return false;
      return pollIntervalMs(elapsed);
    },
  });

  const status = query.data?.status;

  // A new task resets the clock and clears any previous stop.
  useEffect(() => {
    startedAtRef.current = Date.now();
    setStopped(false);
    setTimedOut(false);
  }, [taskId]);

  // The max-duration alarm. Re-armed by `resume()` (which flips `stopped` back to false).
  useEffect(() => {
    if (!taskId || stopped || isTerminalStatus(status)) return;
    const remaining = Math.max(0, startedAtRef.current + maxDurationMs - Date.now());
    const timer = setTimeout(() => {
      setTimedOut(true);
      setStopped(true);
    }, remaining);
    return () => clearTimeout(timer);
  }, [taskId, stopped, status, maxDurationMs]);

  // Surface the timeout once per expiry.
  useEffect(() => {
    if (!timedOut) return;
    toast.error(t("timedOut", { minutes: maxDurationMinutes }), {
      duration: 10_000,
      action: {
        label: t("retry"),
        onClick: () => {
          startedAtRef.current = Date.now();
          setTimedOut(false);
          setStopped(false);
        },
      },
    });
    onTimeoutRef.current?.();
  }, [timedOut, maxDurationMinutes, t]);

  const resume = useCallback(() => {
    startedAtRef.current = Date.now();
    setTimedOut(false);
    setStopped(false);
  }, []);

  const cancel = useCallback(() => {
    if (!taskId) return;
    setIsCancelling(true);
    cancelTask(taskId)
      .then(() => {
        setStopped(true);
        setTimedOut(false);
        toast.success(t("cancelled"));
        onCancelledRef.current?.();
      })
      .catch((err: unknown) => {
        toast.error(err instanceof Error ? err.message : t("cancelFailed"));
      })
      .finally(() => setIsCancelling(false));
  }, [taskId, t]);

  return {
    data: query.data,
    error: query.error,
    isPolling: !!taskId && !stopped && !isTerminalStatus(status),
    timedOut,
    isCancelling,
    maxDurationMinutes,
    cancel,
    resume,
  };
}
