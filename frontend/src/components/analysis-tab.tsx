"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState } from "@/components/empty-state";
import { PaperAnalysisDetail } from "@/components/paper-analysis-detail";
import {
  useAnalyzeGaps,
  useGapReport,
  usePaperAnalyses,
} from "@/hooks/use-analysis";
import {
  AlertTriangle,
  BrainCircuit,
  BookOpen,
  Lightbulb,
  Loader2,
  SearchX,
} from "lucide-react";
import { useTranslations } from "next-intl";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface AnalysisTabProps {
  projectId: string;
}

export function AnalysisTab({ projectId }: AnalysisTabProps) {
  const t = useTranslations("analysis");
  const tEmpty = useTranslations("emptyStates");
  const { data: analysesData, isLoading: analysesLoading } =
    usePaperAnalyses(projectId);
  const { data: gapReport, isLoading: reportLoading } =
    useGapReport(projectId);
  const analyzeGaps = useAnalyzeGaps();

  const analyses = analysesData?.analyses ?? [];
  const total = analysesData?.total ?? 0;
  const hasEnoughPapers = total >= 5;

  const handleRunGapAnalysis = () => {
    analyzeGaps.mutate({ projectId });
  };

  if (analysesLoading || reportLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-6 w-6 animate-spin text-[var(--ds-text-secondary)]" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-[var(--ds-text-heading)]">
            {t("title")}
          </h2>
          <p className="text-sm text-[var(--ds-text-secondary)]">
            {t("analyzedCount", { count: total })}
          </p>
        </div>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger
              render={
                <Button
                  onClick={handleRunGapAnalysis}
                  disabled={!hasEnoughPapers || analyzeGaps.isPending}
                  className="gap-2"
                />
              }
            >
              {analyzeGaps.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <BrainCircuit className="h-4 w-4" />
              )}
              {t("runGapAnalysis")}
            </TooltipTrigger>
            {!hasEnoughPapers && (
              <TooltipContent>
                <p>{t("minPapersRequired", { count: 5 })}</p>
              </TooltipContent>
            )}
          </Tooltip>
        </TooltipProvider>
      </div>

      {/* Gap Report */}
      {gapReport && (
        <div className="space-y-4">
          {/* Summary */}
          <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
            <CardHeader>
              <CardTitle className="text-sm text-[var(--ds-text-body)]">
                {t("summary")}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-[var(--ds-text-body)] whitespace-pre-line">
                {gapReport.summary}
              </p>
            </CardContent>
          </Card>

          {/* Gaps */}
          {gapReport.gaps.length > 0 && (
            <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-sm text-[var(--ds-text-body)]">
                  <SearchX className="h-4 w-4" />
                  {t("researchGaps")} ({gapReport.gaps.length})
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {gapReport.gaps.map((gap, i) => (
                  <div key={i} className="rounded-lg border border-[var(--ds-border)] p-3">
                    <div className="flex items-center gap-2 mb-1">
                      <Badge
                        variant="outline"
                        className={
                          gap.severity === "high"
                            ? "border-[var(--ds-error)]/50 text-[var(--ds-error)]"
                            : gap.severity === "medium"
                              ? "border-[var(--ds-warning)]/50 text-[var(--ds-warning)]"
                              : "border-[var(--ds-border)] text-[var(--ds-text-secondary)]"
                        }
                      >
                        {t(`severity.${gap.severity}`)}
                      </Badge>
                      <Badge variant="outline" className="text-xs">
                        {t(`gapType.${gap.gap_type}`)}
                      </Badge>
                    </div>
                    <p className="text-sm text-[var(--ds-text-body)] mt-1">{gap.description}</p>
                    {gap.evidence.length > 0 && (
                      <ul className="mt-2 list-disc list-inside text-xs text-[var(--ds-text-secondary)]">
                        {gap.evidence.map((e, j) => (
                          <li key={j}>{e}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                ))}
              </CardContent>
            </Card>
          )}

          {/* Controversies */}
          {gapReport.controversies.length > 0 && (
            <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-sm text-[var(--ds-text-body)]">
                  <AlertTriangle className="h-4 w-4" />
                  {t("controversies")} ({gapReport.controversies.length})
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {gapReport.controversies.map((c, i) => (
                  <div key={i} className="rounded-lg border border-[var(--ds-border)] p-3">
                    <p className="text-sm font-medium text-[var(--ds-text-body)] mb-2">
                      {c.topic}
                    </p>
                    {c.positions.map((pos, j) => (
                      <div key={j} className="ml-3 mb-1">
                        <p className="text-sm text-[var(--ds-text-body)]">
                          <span className="text-[var(--ds-text-secondary)]">{t("view")}:</span>{" "}
                          {pos.view}
                        </p>
                        <p className="text-xs text-[var(--ds-text-heading)]0">
                          {pos.supporters.join(", ")}
                        </p>
                      </div>
                    ))}
                  </div>
                ))}
              </CardContent>
            </Card>
          )}

          {/* Suggested Questions */}
          {gapReport.suggested_questions.length > 0 && (
            <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-sm text-[var(--ds-text-body)]">
                  <Lightbulb className="h-4 w-4" />
                  {t("suggestedQuestions")} ({gapReport.suggested_questions.length})
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {gapReport.suggested_questions.map((sq, i) => (
                  <div key={i} className="rounded-lg border border-[var(--ds-border)] p-3">
                    <p className="text-sm font-medium text-[var(--ds-text-body)]">
                      {i + 1}. {sq.question}
                    </p>
                    <p className="text-xs text-[var(--ds-text-secondary)] mt-1">{sq.rationale}</p>
                    <p className="text-xs text-[var(--ds-text-heading)]0 mt-1">
                      <span className="text-[var(--ds-text-secondary)]">
                        {t("suggestedMethod")}:
                      </span>{" "}
                      {sq.methodology_hint}
                    </p>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}

          {/* Theoretical Landscape */}
          {gapReport.theoretical_landscape.length > 0 && (
            <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-sm text-[var(--ds-text-body)]">
                  <BookOpen className="h-4 w-4" />
                  {t("theoreticalLandscape")}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {gapReport.theoretical_landscape.map((theory, i) => (
                    <Badge key={i} variant="secondary">
                      {theory}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* No report yet */}
      {!gapReport && total > 0 && (
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardContent className="py-8 text-center">
            <BrainCircuit className="mx-auto h-8 w-8 text-[var(--ds-text-heading)]0 mb-2" />
            <p className="text-sm text-[var(--ds-text-secondary)]">{t("noReport")}</p>
          </CardContent>
        </Card>
      )}

      {/* Paper Analyses List */}
      {analyses.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-[var(--ds-text-body)]">
            {t("paperAnalyses")} ({total})
          </h3>
          {analyses.map((a) => (
            <PaperAnalysisDetail key={a.id} analysis={a} />
          ))}
        </div>
      )}

      {/* Empty state */}
      {total === 0 && (
        <EmptyState
          icon="🔬"
          title={t("title")}
          description={tEmpty("gapAnalysis.description")}
          prerequisites={[
            {
              label: tEmpty("prerequisites.atLeastAnalyzed", { count: 5 }),
              completed: total >= 5,
              current: total,
              total: 5,
            },
          ]}
          actionLabel={t("runGapAnalysis")}
          onAction={handleRunGapAnalysis}
          actionDisabled={!hasEnoughPapers || analyzeGaps.isPending}
        />
      )}
    </div>
  );
}
