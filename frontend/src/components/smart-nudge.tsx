"use client";

import { useCallback, useMemo, useState } from "react";
import { useTranslations } from "next-intl";

interface SmartNudgeProps {
  projectId: string;
  paperCount: number;
  fullTextCount: number;
  analyzedCount: number;
  hasGapReport: boolean;
  draftCount: number;
  hasComplianceCheck: boolean;
  onNavigate: (view: string) => void;
}

interface NudgeDefinition {
  id: string;
  icon: string;
  titleKey: string;
  titleParams?: Record<string, number>;
  descriptionKey: string;
  actionKey: string;
  navigateTo: string;
}

function getDismissKey(projectId: string, nudgeId: string): string {
  return `nudge-dismissed-${projectId}-${nudgeId}`;
}

function isNudgeDismissed(projectId: string, nudgeId: string): boolean {
  if (typeof window === "undefined") return false;
  return localStorage.getItem(getDismissKey(projectId, nudgeId)) === "true";
}

export function SmartNudge({
  projectId,
  paperCount,
  fullTextCount,
  analyzedCount,
  hasGapReport,
  draftCount,
  hasComplianceCheck,
  onNavigate,
}: SmartNudgeProps) {
  const t = useTranslations("nudges");

  // Track dismissed nudges in state so the component re-renders on dismiss
  const [dismissed, setDismissed] = useState<Set<string>>(() => {
    const set = new Set<string>();
    if (typeof window === "undefined") return set;
    const ids = [
      "welcome",
      "full-texts",
      "analyze",
      "gap-analysis",
      "write",
      "compliance",
    ];
    for (const id of ids) {
      if (isNudgeDismissed(projectId, id)) {
        set.add(id);
      }
    }
    return set;
  });

  // Build the ordered list of applicable nudges based on project state
  const applicableNudges = useMemo<NudgeDefinition[]>(() => {
    const nudges: NudgeDefinition[] = [];

    if (paperCount === 0) {
      nudges.push({
        id: "welcome",
        icon: "\u{1F680}",
        titleKey: "welcome.title",
        descriptionKey: "welcome.description",
        actionKey: "welcome.action",
        navigateTo: "papers",
      });
    }

    if (paperCount > 0 && fullTextCount === 0) {
      nudges.push({
        id: "full-texts",
        icon: "\u{1F4D1}",
        titleKey: "fullTexts.title",
        titleParams: { count: paperCount },
        descriptionKey: "fullTexts.description",
        actionKey: "fullTexts.action",
        navigateTo: "papers",
      });
    }

    if (paperCount > 0 && analyzedCount === 0) {
      nudges.push({
        id: "analyze",
        icon: "\u{1F52C}",
        titleKey: "analyze.title",
        titleParams: { count: paperCount },
        descriptionKey: "analyze.description",
        actionKey: "analyze.action",
        navigateTo: "papers",
      });
    }

    if (analyzedCount >= 5 && !hasGapReport) {
      nudges.push({
        id: "gap-analysis",
        icon: "\u{1F52C}",
        titleKey: "gapAnalysis.title",
        titleParams: { count: analyzedCount },
        descriptionKey: "gapAnalysis.description",
        actionKey: "gapAnalysis.action",
        navigateTo: "gap-analysis",
      });
    }

    if (hasGapReport && draftCount === 0) {
      nudges.push({
        id: "write",
        icon: "\u270D\uFE0F",
        titleKey: "write.title",
        descriptionKey: "write.description",
        actionKey: "write.action",
        navigateTo: "drafts",
      });
    }

    if (draftCount > 0 && !hasComplianceCheck) {
      nudges.push({
        id: "compliance",
        icon: "\u2705",
        titleKey: "compliance.title",
        descriptionKey: "compliance.description",
        actionKey: "compliance.action",
        navigateTo: "drafts",
      });
    }

    return nudges;
  }, [
    paperCount,
    fullTextCount,
    analyzedCount,
    hasGapReport,
    draftCount,
    hasComplianceCheck,
  ]);

  // Find the first non-dismissed nudge
  const activeNudge = useMemo(() => {
    return applicableNudges.find((n) => !dismissed.has(n.id)) ?? null;
  }, [applicableNudges, dismissed]);

  const handleDismiss = useCallback(
    (nudgeId: string) => {
      localStorage.setItem(getDismissKey(projectId, nudgeId), "true");
      setDismissed((prev) => {
        const next = new Set(prev);
        next.add(nudgeId);
        return next;
      });
    },
    [projectId],
  );

  if (!activeNudge) return null;

  return (
    <div className="relative flex items-start gap-3 rounded-lg border border-[var(--ds-primary)]/15 bg-[var(--ds-primary-light)] px-4 py-3">
      {/* Icon */}
      <span className="shrink-0 text-[20px] leading-none" role="img">
        {activeNudge.icon}
      </span>

      {/* Text content */}
      <div className="min-w-0 flex-1">
        <p className="text-[13px] font-bold text-[var(--ds-text-heading)]">
          {t(activeNudge.titleKey, activeNudge.titleParams)}
        </p>
        <p className="mt-0.5 text-[12px] text-[var(--ds-text-muted)]">
          {t(activeNudge.descriptionKey)}
        </p>
      </div>

      {/* Action button */}
      <button
        onClick={() => onNavigate(activeNudge.navigateTo)}
        className="shrink-0 self-center rounded-md bg-[var(--ds-primary)] px-3 py-1.5 text-[12px] font-medium text-white transition-colors hover:bg-[var(--ds-primary-hover)]"
      >
        {t(activeNudge.actionKey)}
      </button>

      {/* Dismiss button */}
      <button
        onClick={() => handleDismiss(activeNudge.id)}
        className="absolute right-2 top-2 flex h-5 w-5 items-center justify-center rounded text-[var(--ds-text-muted)] transition-colors hover:text-[var(--ds-text-heading)]"
        aria-label={t("dismiss")}
      >
        <svg
          width="12"
          height="12"
          viewBox="0 0 12 12"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M2.5 2.5L9.5 9.5M9.5 2.5L2.5 9.5"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </svg>
      </button>
    </div>
  );
}
