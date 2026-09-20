"use client";

import { useMemo } from "react";
import { useTranslations } from "next-intl";
import { useProject } from "@/hooks/use-projects";
import { useProjectPapers } from "@/hooks/use-papers";
import { useProjectDrafts } from "@/hooks/use-drafts";
import { usePaperAnalyses, useGapReport } from "@/hooks/use-analysis";
import { SmartNudge } from "@/components/smart-nudge";

interface ProjectDashboardProps {
  projectId: string;
  onNavigate: (view: string) => void;
}

const phaseConfigs = [
  { number: 1, labelKey: "findLabel", descKey: "findDesc", navigateTo: "papers" },
  { number: 2, labelKey: "exploreLabel", descKey: "exploreDesc", navigateTo: "paper-chat" },
  { number: 3, labelKey: "createLabel", descKey: "createDesc", navigateTo: "drafts" },
];

export function ProjectDashboard({
  projectId,
  onNavigate,
}: ProjectDashboardProps) {
  const tDash = useTranslations("dashboard");
  const tCommon = useTranslations("common");
  const { data: project } = useProject(projectId);
  const { data: papersData } = useProjectPapers(projectId);
  const { data: draftsData } = useProjectDrafts(projectId);
  const { data: analysesData } = usePaperAnalyses(projectId);
  const { data: gapReport } = useGapReport(projectId);

  const papers = papersData?.papers ?? [];
  const drafts = draftsData?.drafts ?? [];
  const analyses = analysesData?.analyses ?? [];

  const fullTextCount = useMemo(
    () =>
      papers.filter(
        (pp) => pp.paper.metadata?.fulltext_status === "acquired",
      ).length,
    [papers],
  );

  const analyzedCount = analyses.length;
  const pendingAnalysisCount = papers.length - analyzedCount;
  const hasGapReport = !!gapReport?.summary;

  // Count draft sections (keys in content object)
  const sectionCount = useMemo(
    () =>
      drafts.reduce((sum, d) => {
        const content = d.content;
        if (content && typeof content === "object") {
          return sum + Object.keys(content).length;
        }
        return sum;
      }, 0),
    [drafts],
  );

  const stats = [
    {
      label: tDash("stats.papers"),
      value: papers.length,
      sub:
        papers.length > 0
          ? tDash("sub.withFullText", { count: fullTextCount })
          : tDash("sub.noneYet"),
    },
    {
      label: tDash("stats.analyzed"),
      value: analyzedCount,
      sub:
        pendingAnalysisCount > 0
          ? tDash("sub.pending", { count: pendingAnalysisCount })
          : analyzedCount > 0
            ? tDash("sub.allDone")
            : tDash("sub.noneYet"),
    },
    {
      label: tDash("stats.gapReport"),
      value: hasGapReport ? tDash("sub.yes") : "\u2014",
      sub: hasGapReport
        ? tDash("sub.gapsFound", { count: gapReport?.gaps?.length ?? 0 })
        : tDash("sub.notGenerated"),
    },
    {
      label: tDash("stats.drafts"),
      value: drafts.length,
      sub:
        drafts.length > 0
          ? tDash("sub.sections", { count: sectionCount })
          : tDash("sub.noneYet"),
    },
  ];

  const metaParts: string[] = [];
  if (project?.citation_style) metaParts.push(project.citation_style);
  if (project?.target_journal) metaParts.push(project.target_journal);
  if (project?.status) {
    metaParts.push(
      project.status.charAt(0).toUpperCase() + project.status.slice(1),
    );
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6 px-4 py-6">
      {/* ── Header ─────────────────────────────────────── */}
      <div>
        <h1
          className="text-2xl font-bold text-[var(--ds-text-heading)]"
          style={{ fontFamily: "var(--font-heading)" }}
        >
          {project?.title ?? tCommon("loading")}
        </h1>

        {metaParts.length > 0 && (
          <p className="mt-1 text-[13px] text-[var(--ds-text-muted)]">
            {metaParts.join(" \u00b7 ")}
          </p>
        )}

        {project?.refined_topic && (
          <div className="mt-3 rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-card)] px-4 py-2.5">
            <p className="text-[12px] font-medium uppercase tracking-wide text-[var(--ds-text-muted)]">
              {tDash("refinedTopic")}
            </p>
            <p className="mt-0.5 text-[13px] leading-relaxed text-[var(--ds-text-body)]">
              {project.refined_topic}
            </p>
          </div>
        )}
      </div>

      {/* ── Stats Row ──────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {stats.map((stat) => (
          <div
            key={stat.label}
            className="rounded-xl border border-[var(--ds-border)] bg-[var(--ds-bg-card)] px-4 py-3 text-center"
          >
            <p className="text-[12px] font-medium text-[var(--ds-text-muted)]">
              {stat.label}
            </p>
            <p
              className="mt-1 text-xl font-bold text-[var(--ds-text-heading)]"
              style={{ fontFamily: "var(--font-heading)" }}
            >
              {stat.value}
            </p>
            <p className="mt-0.5 text-[11px] text-[var(--ds-text-muted)]">
              {stat.sub}
            </p>
          </div>
        ))}
      </div>

      {/* ── Smart Nudge ────────────────────────────────── */}
      <SmartNudge
        projectId={projectId}
        paperCount={papers.length}
        fullTextCount={fullTextCount}
        analyzedCount={analyzedCount}
        hasGapReport={hasGapReport}
        draftCount={drafts.length}
        hasComplianceCheck={false}
        onNavigate={onNavigate}
      />

      {/* ── Phase Cards ────────────────────────────────── */}
      <div>
        <h2
          className="mb-3 text-sm font-bold text-[var(--ds-text-heading)]"
          style={{ fontFamily: "var(--font-heading)" }}
        >
          {tDash("researchPhases")}
        </h2>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {phaseConfigs.map((phase) => (
            <button
              key={phase.number}
              onClick={() => onNavigate(phase.navigateTo)}
              className="flex items-start gap-3 rounded-xl border border-[var(--ds-border)] bg-[var(--ds-bg-card)] px-4 py-3.5 text-left transition-colors hover:border-[var(--ds-primary)]/40"
            >
              {/* Number circle */}
              <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-[var(--ds-primary)] text-[12px] font-bold text-white">
                {phase.number}
              </span>

              <div className="min-w-0">
                <p className="text-[13px] font-bold text-[var(--ds-text-heading)]">
                  {tDash(`phases.${phase.labelKey}`)}
                </p>
                <p className="mt-0.5 text-[12px] leading-relaxed text-[var(--ds-text-muted)]">
                  {tDash(`phases.${phase.descKey}`)}
                </p>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
