/**
 * Bounds for the AI Write dialog's target-words field. The backend
 * (`app/services/writing.py`) accepts any positive integer; these bounds are a
 * frontend-only sanity range so the field cannot be left empty or set to something
 * absurd, matching the dialog's own default.
 */

export const DEFAULT_TARGET_WORDS = 400;
export const MIN_TARGET_WORDS = 100;
export const MAX_TARGET_WORDS = 3000;

/** Clamp a target-words input to the dialog's supported range, defaulting an
 * unparsable or missing value to `DEFAULT_TARGET_WORDS`. */
export function clampTargetWords(value: number | null | undefined): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return DEFAULT_TARGET_WORDS;
  return Math.min(MAX_TARGET_WORDS, Math.max(MIN_TARGET_WORDS, Math.round(value)));
}

/**
 * `generate_section` saves the finalized section itself, and finalize can legitimately
 * empty it (every cited sentence unverified, or every uncited sentence tagged
 * "finding"). A drafts-tab completion effect gated only on `result.content` being
 * truthy would miss this: the empty string is falsy, so a completed job with nothing
 * left in it would run neither branch, leaving the draft query never invalidated, no
 * message shown, and the editor still rendering text the backend already deleted.
 *
 * This pure helper decides what the effect should do, given only the polled job's own
 * `status` and `result`: `null` when there is nothing to act on yet (still running, or
 * completed with no result), `"generated"` for an ordinary non-empty result, and
 * `"emptyRemoved"` when the job completed but finalize left nothing -- the caller
 * invalidates the draft query and tells the user the section was removed either way.
 */
export interface GenerateCompletionInput {
  content: string;
  section_type: string;
  papers_used: number;
}

export type GenerateCompletionAction =
  | { kind: "generated"; sectionType: string; papersUsed: number }
  | { kind: "emptyRemoved"; sectionType: string };

export function describeGenerateCompletion(
  status: string | null | undefined,
  result: GenerateCompletionInput | null | undefined,
): GenerateCompletionAction | null {
  if (status !== "completed" || !result) return null;
  if (result.content === "") {
    return { kind: "emptyRemoved", sectionType: result.section_type };
  }
  return { kind: "generated", sectionType: result.section_type, papersUsed: result.papers_used };
}
