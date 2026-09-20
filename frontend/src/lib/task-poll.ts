import { apiFetch } from "@/lib/api";
import { pollIntervalMs } from "@/hooks/use-task-polling";

export interface TaskSnapshot<T> {
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  progress_message: string | null;
  result: T | null;
  error: string | null;
}

export interface PollOptions {
  /** Give up after this many ms (default 180000 = 3 min). */
  timeoutMs?: number;
}

/**
 * Poll `GET /tasks/{taskId}` until the job reaches a terminal state.
 * Resolves with the job result, rejects on failure, cancellation or timeout.
 *
 * This is a one-shot, promise-based helper for call sites that cannot use the
 * `useTaskPolling` hook — a mutation's `mutationFn` and a plain async event
 * handler are not React render contexts, so hooks are unavailable there. It reuses
 * `pollIntervalMs`'s backoff ladder from `use-task-polling` instead of a fixed
 * interval so the two polling paths share one timing policy.
 */
export async function pollTaskResult<T>(
  taskId: string,
  opts: PollOptions = {},
): Promise<T> {
  const timeoutMs = opts.timeoutMs ?? 180_000;
  const startedAt = Date.now();

  for (;;) {
    const task = await apiFetch<TaskSnapshot<T>>(`/tasks/${taskId}`);

    if (task.status === "completed") {
      if (task.result === null) {
        throw new Error("Task completed without a result");
      }
      return task.result;
    }
    if (task.status === "failed") {
      throw new Error(task.error || "Task failed");
    }
    if (task.status === "cancelled") {
      throw new Error("Task was cancelled");
    }
    const elapsed = Date.now() - startedAt;
    if (elapsed > timeoutMs) {
      throw new Error("Task timed out");
    }
    await new Promise((resolve) => setTimeout(resolve, pollIntervalMs(elapsed)));
  }
}
