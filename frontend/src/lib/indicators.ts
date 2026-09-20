/**
 * Turn paper-record *indicators* (WoS indexing, recency, citation count) into chip
 * descriptors. Indicators are contextual signals, not verification checks: they never
 * affect `overall_status`, so the UI renders them as neutral chips.
 *
 * The backend may attach structured `details`; when it does not, the human-readable
 * message is parsed best-effort and the raw message is the final fallback.
 */

export interface IndicatorLike {
  check_type: string;
  status: string;
  message: string;
  details?: Record<string, unknown> | null;
}

export type IndicatorChip =
  | { kind: "wos"; collection: string | null }
  | { kind: "notWos" }
  | { kind: "year"; year: number }
  | { kind: "citations"; count: number }
  | { kind: "text"; text: string };

export interface IndicatorContext {
  /** WoS collection already known for the row (e.g. from the paper record). */
  wosCollection?: string | null;
}

function numberFrom(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && /^\d+$/.test(value)) return Number(value);
  return null;
}

function stringFrom(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

export function describeIndicator(
  indicator: IndicatorLike,
  ctx: IndicatorContext = {},
): IndicatorChip | null {
  if (indicator.status === "skipped") return null;
  const details = indicator.details ?? {};
  const message = indicator.message ?? "";

  switch (indicator.check_type) {
    case "wos_indexed": {
      const indexed =
        typeof details.indexed === "boolean" ? details.indexed : !/\bnot\b/i.test(message);
      if (!indexed) return { kind: "notWos" };
      const collection =
        stringFrom(details.collection) ??
        stringFrom(details.wos_collection) ??
        stringFrom(ctx.wosCollection);
      return { kind: "wos", collection };
    }
    case "recency": {
      const year =
        numberFrom(details.year) ?? numberFrom(message.match(/\b(1[89]|20)\d{2}\b/)?.[0]);
      return year === null ? { kind: "text", text: message } : { kind: "year", year };
    }
    case "citation_count": {
      const count =
        numberFrom(details.count) ??
        numberFrom(details.citation_count) ??
        numberFrom(message.match(/(\d+)\s+citations?/i)?.[1]);
      return count === null ? { kind: "text", text: message } : { kind: "citations", count };
    }
    default:
      return { kind: "text", text: message };
  }
}

export function describeIndicators(
  indicators: IndicatorLike[] | null | undefined,
  ctx: IndicatorContext = {},
): IndicatorChip[] {
  if (!indicators) return [];
  const chips: IndicatorChip[] = [];
  for (const indicator of indicators) {
    const chip = describeIndicator(indicator, ctx);
    if (chip) chips.push(chip);
  }
  return chips;
}
