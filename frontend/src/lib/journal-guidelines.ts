import { apiFetch } from "@/lib/api";
import { pollTaskResult } from "@/lib/task-poll";

export type JournalGuidelinesResult = Record<string, unknown>;

/**
 * Kick off the journal-guidelines extraction job and wait for its result.
 * The endpoint returns 202 + task_id; the extraction itself can take ~60s.
 */
export async function fetchJournalGuidelines(
  projectId: string,
  guidelinesText?: string,
): Promise<JournalGuidelinesResult> {
  const { task_id } = await apiFetch<{ task_id: string }>(
    `/projects/${projectId}/fetch-journal-guidelines`,
    {
      method: "POST",
      body: JSON.stringify({ guidelines_text: guidelinesText || undefined }),
    },
  );
  return pollTaskResult<JournalGuidelinesResult>(task_id);
}
