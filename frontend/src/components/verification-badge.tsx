"use client";

import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  HelpCircle,
  Loader2,
} from "lucide-react";
import { useTranslations } from "next-intl";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { describeIndicator, type IndicatorChip, type IndicatorLike } from "@/lib/indicators";

interface VerificationBadgeProps {
  /** Record status derived from doi_exists + metadata_match only. */
  status: string;
  /** Record checks (doi_exists, metadata_match) shown in the tooltip. */
  checks?: { check_type: string; status: string; message: string }[];
}

/**
 * Level-1 ("record") verification badge. Labels name what was checked, not a value
 * judgement: the DOI resolves and the metadata agrees (record verified), the metadata
 * disagrees (metadata mismatch), the DOI does not resolve (DOI not found), or there is
 * no DOI to check. Indicators (WoS, recency, citations) are rendered separately by
 * `IndicatorChips` and never change this status.
 */
const statusStyleConfig = {
  pass: {
    icon: CheckCircle2,
    color: "text-[var(--ds-success)]",
    bg: "bg-[rgba(5,150,105,0.08)]",
    labelKey: "recordVerified" as const,
  },
  fail: {
    icon: XCircle,
    color: "text-[var(--ds-error)]",
    bg: "bg-red-400/15",
    labelKey: "doiNotFound" as const,
  },
  warning: {
    icon: AlertTriangle,
    color: "text-yellow-400",
    bg: "bg-yellow-400/15",
    labelKey: "metadataMismatch" as const,
  },
  no_doi: {
    icon: HelpCircle,
    color: "text-[var(--ds-text-muted)]",
    bg: "bg-[var(--ds-text-muted)]/15",
    labelKey: "noDoi" as const,
  },
  pending: {
    icon: Loader2,
    color: "text-[var(--ds-text-secondary)]",
    bg: "bg-[var(--ds-text-secondary)]/15",
    labelKey: "pending" as const,
  },
} as const;

export function VerificationBadge({ status, checks }: VerificationBadgeProps) {
  const t = useTranslations("verification");
  const config =
    statusStyleConfig[status as keyof typeof statusStyleConfig] || statusStyleConfig.pending;
  const Icon = config.icon;

  const badgeContent = (
    <>
      <Icon
        className={`size-3 ${status === "pending" ? "animate-spin" : ""}`}
      />
      {t(config.labelKey)}
    </>
  );

  if (!checks || checks.length === 0) {
    return (
      <span
        className={`inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-xs font-medium ${config.bg} ${config.color}`}
      >
        {badgeContent}
      </span>
    );
  }

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger
          render={
            <span
              className={`inline-flex cursor-default items-center gap-1 rounded-md px-1.5 py-0.5 text-xs font-medium ${config.bg} ${config.color}`}
            />
          }
        >
          {badgeContent}
        </TooltipTrigger>
        <TooltipContent
          side="bottom"
          className="max-w-xs border-[var(--ds-border)] bg-[var(--ds-bg-card)] text-[var(--ds-text-heading)]"
        >
          <div className="space-y-1">
            {checks.map((check, i) => (
              <div key={i} className="flex items-start gap-1.5 text-xs">
                <span
                  className={
                    check.status === "pass"
                      ? "text-[var(--ds-success)]"
                      : check.status === "fail"
                        ? "text-[var(--ds-error)]"
                        : check.status === "warning"
                          ? "text-yellow-400"
                          : "text-[var(--ds-text-muted)]"
                  }
                >
                  {check.status === "pass"
                    ? "\u2713"
                    : check.status === "fail"
                      ? "\u2717"
                      : check.status === "warning"
                        ? "!"
                        : "\u2013"}
                </span>
                <span className="text-[var(--ds-text-body)]">{check.message}</span>
              </div>
            ))}
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

interface IndicatorChipsProps {
  indicators?: IndicatorLike[] | null;
  /** WoS collection already known for the row, used when the indicator carries none. */
  wosCollection?: string | null;
}

/**
 * Neutral chips for the contextual indicators returned next to a record verification:
 * WoS indexing, publication year, citation count. Skipped indicators are hidden.
 */
export function IndicatorChips({ indicators, wosCollection }: IndicatorChipsProps) {
  const t = useTranslations("verification");
  if (!indicators || indicators.length === 0) return null;

  const label = (chip: IndicatorChip): string => {
    switch (chip.kind) {
      case "wos":
        return chip.collection
          ? t("indicatorWos", { collection: chip.collection })
          : t("indicatorWosIndexed");
      case "notWos":
        return t("indicatorNotWos");
      case "year":
        return String(chip.year);
      case "citations":
        return t("indicatorCitations", { count: chip.count });
      case "text":
        return chip.text;
    }
  };

  const rendered = indicators
    .map((indicator, i) => ({ indicator, chip: describeIndicator(indicator, { wosCollection }), i }))
    .filter((entry): entry is { indicator: IndicatorLike; chip: IndicatorChip; i: number } =>
      entry.chip !== null,
    );
  if (rendered.length === 0) return null;

  return (
    <span
      className="inline-flex flex-wrap items-center gap-1"
      role="group"
      aria-label={t("indicatorsLabel")}
      data-testid="indicator-chips"
    >
      {rendered.map(({ indicator, chip, i }) => (
        <span
          key={`${indicator.check_type}-${i}`}
          title={indicator.message}
          className="inline-flex items-center rounded-md border border-[var(--ds-border)] bg-[var(--ds-bg-page)] px-1.5 py-0.5 text-xs text-[var(--ds-text-secondary)]"
        >
          {label(chip)}
        </span>
      ))}
    </span>
  );
}
