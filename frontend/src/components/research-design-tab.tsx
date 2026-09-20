"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { useQueryClient } from "@tanstack/react-query";
import {
  Compass,
  Lightbulb,
  Loader2,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { EmptyState } from "@/components/empty-state";
import { Input } from "@/components/ui/input";
import { CollectionPlanView } from "@/components/collection-plan-view";
import { ResearchDesignView } from "@/components/research-design-view";
import { useGapReport } from "@/hooks/use-analysis";
import { useProjectPapers } from "@/hooks/use-papers";
import {
  useCollectionPlan,
  useGenerateCollectionPlan,
  useGenerateDesign,
  useResearchDesign,
} from "@/hooks/use-research-design";
import { useTaskPolling } from "@/hooks/use-task-polling";

interface ResearchDesignTabProps {
  projectId: string;
}

export function ResearchDesignTab({ projectId }: ResearchDesignTabProps) {
  const t = useTranslations("design");
  const tEmpty = useTranslations("emptyStates");
  const queryClient = useQueryClient();

  const [researchQuestion, setResearchQuestion] = useState("");
  const [designTaskId, setDesignTaskId] = useState<string | null>(null);
  const [planTaskId, setPlanTaskId] = useState<string | null>(null);

  const { data: design, isLoading: designLoading } = useResearchDesign(projectId);
  const { data: plan, isLoading: planLoading } = useCollectionPlan(projectId);
  const { data: gapReport } = useGapReport(projectId);
  const { data: papersData } = useProjectPapers(projectId);
  const paperCount = papersData?.total ?? 0;
  const generateDesign = useGenerateDesign();
  const generatePlan = useGenerateCollectionPlan();

  // Poll design task status — bounded, backed off.
  const { data: designTaskStatus } = useTaskPolling<{
    status: string;
    progress: number;
    progress_message: string | null;
  }>({ taskId: designTaskId, kind: "research_design" });

  // Poll collection plan task status — bounded, backed off.
  const { data: planTaskStatus } = useTaskPolling<{
    status: string;
    progress: number;
    progress_message: string | null;
  }>({ taskId: planTaskId, kind: "collection_plan" });

  // When design task completes, refresh design data
  if (designTaskStatus?.status === "completed" && designTaskId) {
    queryClient.invalidateQueries({
      queryKey: ["projects", projectId, "research-design"],
    });
    setDesignTaskId(null);
  }

  // When plan task completes, refresh plan data
  if (planTaskStatus?.status === "completed" && planTaskId) {
    queryClient.invalidateQueries({
      queryKey: ["projects", projectId, "data-collection-plan"],
    });
    setPlanTaskId(null);
  }

  const isGeneratingDesign =
    !!designTaskId &&
    designTaskStatus?.status !== "completed" &&
    designTaskStatus?.status !== "failed";

  const isGeneratingPlan =
    !!planTaskId &&
    planTaskStatus?.status !== "completed" &&
    planTaskStatus?.status !== "failed";

  function handleGenerateDesign() {
    if (!researchQuestion.trim()) return;
    generateDesign.mutate(
      { projectId, researchQuestion: researchQuestion.trim() },
      { onSuccess: (data) => setDesignTaskId(data.task_id) },
    );
  }

  function handleGeneratePlan() {
    generatePlan.mutate(
      { projectId },
      { onSuccess: (data) => setPlanTaskId(data.task_id) },
    );
  }

  function handleSuggestedClick(question: string) {
    setResearchQuestion(question);
  }

  if (designLoading || planLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-6 w-6 animate-spin text-[var(--ds-text-secondary)]" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-lg font-semibold text-[var(--ds-text-heading)]">{t("title")}</h2>
      </div>

      {/* Research Question Input */}
      <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
        <CardContent className="pt-6 space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-[var(--ds-text-heading)]">
              {t("researchQuestion")}
            </label>
            <div className="flex gap-2">
              <Input
                value={researchQuestion}
                onChange={(e) => setResearchQuestion(e.target.value)}
                placeholder={t("researchQuestionPlaceholder")}
                className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-heading)]0"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleGenerateDesign();
                }}
              />
              <Button
                onClick={handleGenerateDesign}
                disabled={
                  !researchQuestion.trim() ||
                  generateDesign.isPending ||
                  isGeneratingDesign
                }
                className="shrink-0 gap-2"
              >
                {generateDesign.isPending || isGeneratingDesign ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Compass className="h-4 w-4" />
                )}
                {isGeneratingDesign
                  ? designTaskStatus?.progress_message || t("generating")
                  : t("generateDesign")}
              </Button>
            </div>
          </div>

          {/* Suggested questions from gap report */}
          {gapReport?.suggested_questions &&
            gapReport.suggested_questions.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center gap-1.5">
                  <Lightbulb className="h-3.5 w-3.5 text-amber-400" />
                  <span className="text-xs font-medium text-[var(--ds-text-secondary)]">
                    {t("suggestedQuestions")}
                  </span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {gapReport.suggested_questions.map((sq, i) => (
                    <button
                      key={i}
                      onClick={() => handleSuggestedClick(sq.question)}
                      className="rounded-full border border-[var(--ds-border)] bg-[var(--ds-bg-page)] px-3 py-1 text-xs text-[var(--ds-text-heading)] transition-colors hover:border-[var(--ds-primary)]/50 hover:text-[var(--ds-primary)]"
                    >
                      {sq.question}
                    </button>
                  ))}
                </div>
              </div>
            )}
        </CardContent>
      </Card>

      {/* Design Task Failed */}
      {designTaskStatus?.status === "failed" && (
        <Card className="border-[var(--ds-error)]/30 bg-[var(--ds-bg-card)]">
          <CardContent className="py-4 text-center">
            <p className="text-sm text-[var(--ds-error)]">
              {t("designFailed")}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Research Design Display */}
      {design ? (
        <ResearchDesignView design={design} />
      ) : (
        !isGeneratingDesign && (
          <EmptyState
            icon="🧭"
            title={t("title")}
            description={tEmpty("researchDesign.description")}
            prerequisites={[
              {
                label: tEmpty("prerequisites.atLeastPapers", { count: 5 }),
                completed: paperCount >= 5,
                current: paperCount,
                total: 5,
              },
              {
                label: tEmpty("prerequisites.gapReportOptional"),
                completed: !!gapReport,
                optional: true,
              },
            ]}
            actionLabel={t("generateDesign")}
            onAction={handleGenerateDesign}
            actionDisabled={!researchQuestion.trim() || generateDesign.isPending || isGeneratingDesign}
          />
        )
      )}

      {/* Generate Collection Plan Button */}
      {design && (
        <div className="flex justify-center">
          <Button
            onClick={handleGeneratePlan}
            disabled={generatePlan.isPending || isGeneratingPlan}
            variant="outline"
            className="gap-2 border-[var(--ds-border)] text-[var(--ds-text-heading)] hover:border-[var(--ds-primary)]/50 hover:text-[var(--ds-primary)]"
          >
            {generatePlan.isPending || isGeneratingPlan ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : null}
            {isGeneratingPlan
              ? planTaskStatus?.progress_message || t("generatingPlan")
              : t("generateCollectionPlan")}
          </Button>
        </div>
      )}

      {/* Plan Task Failed */}
      {planTaskStatus?.status === "failed" && (
        <Card className="border-[var(--ds-error)]/30 bg-[var(--ds-bg-card)]">
          <CardContent className="py-4 text-center">
            <p className="text-sm text-[var(--ds-error)]">
              {t("planFailed")}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Collection Plan Display */}
      {plan ? (
        <CollectionPlanView plan={plan} />
      ) : (
        design &&
        !isGeneratingPlan && (
          <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
            <CardContent className="py-8 text-center">
              <p className="text-sm text-[var(--ds-text-secondary)]">{t("noPlan")}</p>
            </CardContent>
          </Card>
        )
      )}
    </div>
  );
}
