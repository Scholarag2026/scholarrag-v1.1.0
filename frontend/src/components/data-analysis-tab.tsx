"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { useQueryClient } from "@tanstack/react-query";
import {
  BarChart3,
  BookOpen,
  ChevronDown,
  ChevronRight,
  Database,
  Loader2,
  Trash2,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { EmptyState } from "@/components/empty-state";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { DataPreviewTable } from "@/components/data-preview-table";
import { DataSummaryPanel } from "@/components/data-summary-panel";
import { DatasetUpload } from "@/components/dataset-upload";
import { InterCoderReportView } from "@/components/inter-coder-report";
import { QualitativeCodingView } from "@/components/qualitative-coding-view";
import { QuantitativeView } from "@/components/quantitative-view";
import {
  useDatasets,
  useDatasetDetail,
  useDeleteDataset,
} from "@/hooks/use-datasets";
import {
  useQuantitativeResults,
  useTriggerQuantitative,
} from "@/hooks/use-quantitative";
import {
  useQualitativeResults,
  useTriggerQualitative,
  useInterCoderReport,
  useTriggerInterCoder,
} from "@/hooks/use-qualitative";
import { useTaskPolling } from "@/hooks/use-task-polling";

interface DataAnalysisTabProps {
  projectId: string;
}

type AnalysisMode = "quantitative" | "qualitative";

export function DataAnalysisTab({ projectId }: DataAnalysisTabProps) {
  const t = useTranslations("dataAnalysis");
  const tEmpty = useTranslations("emptyStates");
  const queryClient = useQueryClient();

  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>("quantitative");
  const [quantTaskId, setQuantTaskId] = useState<string | null>(null);
  const [qualTaskId, setQualTaskId] = useState<string | null>(null);
  const [interCoderTaskId, setInterCoderTaskId] = useState<string | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);

  // Data fetching
  const { data: datasets, isLoading: datasetsLoading } = useDatasets(projectId);
  const activeDatasetId = selectedDatasetId || (datasets && datasets.length > 0 ? datasets[0].id : null);
  const { data: datasetDetail } = useDatasetDetail(activeDatasetId);
  const { data: quantResults, isLoading: quantLoading } = useQuantitativeResults(activeDatasetId);
  const { data: qualResults, isLoading: qualLoading } = useQualitativeResults(activeDatasetId);
  const { data: interCoderReport, isLoading: interCoderLoading } = useInterCoderReport(activeDatasetId);

  const triggerQuant = useTriggerQuantitative();
  const triggerQual = useTriggerQualitative();
  const triggerInterCoder = useTriggerInterCoder();
  const deleteDataset = useDeleteDataset();

  // Poll quantitative task status — bounded, backed off.
  const { data: quantTaskStatus } = useTaskPolling<{
    status: string;
    progress: number;
    progress_message: string | null;
  }>({ taskId: quantTaskId, kind: "quantitative" });

  // Poll qualitative task status — bounded, backed off.
  const { data: qualTaskStatus } = useTaskPolling<{
    status: string;
    progress: number;
    progress_message: string | null;
  }>({ taskId: qualTaskId, kind: "qualitative" });

  // Poll inter-coder task status — bounded, backed off.
  const { data: interCoderTaskStatus } = useTaskPolling<{
    status: string;
    progress: number;
    progress_message: string | null;
  }>({ taskId: interCoderTaskId, kind: "inter_coder" });

  // When quantitative task completes
  if (quantTaskStatus?.status === "completed" && quantTaskId) {
    queryClient.invalidateQueries({
      queryKey: ["datasets", activeDatasetId, "analysis", "quantitative"],
    });
    setQuantTaskId(null);
  }

  // When qualitative task completes
  if (qualTaskStatus?.status === "completed" && qualTaskId) {
    queryClient.invalidateQueries({
      queryKey: ["datasets", activeDatasetId, "analysis", "qualitative"],
    });
    setQualTaskId(null);
  }

  // When inter-coder task completes
  if (interCoderTaskStatus?.status === "completed" && interCoderTaskId) {
    queryClient.invalidateQueries({
      queryKey: ["datasets", activeDatasetId, "inter-coder-reliability"],
    });
    setInterCoderTaskId(null);
  }

  const isRunningQuant =
    !!quantTaskId &&
    quantTaskStatus?.status !== "completed" &&
    quantTaskStatus?.status !== "failed";

  const isRunningQual =
    !!qualTaskId &&
    qualTaskStatus?.status !== "completed" &&
    qualTaskStatus?.status !== "failed";

  const isRunningInterCoder =
    !!interCoderTaskId &&
    interCoderTaskStatus?.status !== "completed" &&
    interCoderTaskStatus?.status !== "failed";

  function handleRunAnalysis() {
    if (!activeDatasetId) return;
    if (analysisMode === "quantitative") {
      triggerQuant.mutate(
        { datasetId: activeDatasetId },
        { onSuccess: (data) => setQuantTaskId(data.task_id) },
      );
    } else {
      triggerQual.mutate(
        { datasetId: activeDatasetId },
        { onSuccess: (data) => setQualTaskId(data.task_id) },
      );
    }
  }

  function handleRunInterCoder() {
    if (!activeDatasetId) return;
    triggerInterCoder.mutate(
      { datasetId: activeDatasetId },
      { onSuccess: (data) => setInterCoderTaskId(data.task_id) },
    );
  }

  function handleDeleteDataset() {
    if (!activeDatasetId) return;
    deleteDataset.mutate(
      { datasetId: activeDatasetId, projectId },
      {
        onSuccess: () => {
          setSelectedDatasetId(null);
        },
      },
    );
  }

  if (datasetsLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-6 w-6 animate-spin text-[var(--ds-text-secondary)]" />
      </div>
    );
  }

  const hasDatasets = datasets && datasets.length > 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-lg font-semibold text-[var(--ds-text-heading)]">{t("title")}</h2>
      </div>

      {/* Upload Section - hidden when dataset exists */}
      {!hasDatasets && (
        <>
          <EmptyState
            icon="📊"
            title={t("title")}
            description={tEmpty("dataAnalysis.description")}
            prerequisites={[
              { label: tEmpty("prerequisites.uploadDataset"), completed: false },
            ]}
            actionLabel={t("upload.uploadButton")}
            onAction={() => {}}
            actionDisabled={true}
          />
          <DatasetUpload
            projectId={projectId}
            onUploadSuccess={(id) => setSelectedDatasetId(id)}
          />
        </>
      )}

      {/* Dataset Selector + Actions */}
      {hasDatasets && (
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardContent className="py-3">
            <div className="flex items-center gap-3">
              <Database className="h-4 w-4 text-[var(--ds-primary)]" />
              {datasets.length > 1 ? (
                <select
                  value={activeDatasetId || ""}
                  onChange={(e) => setSelectedDatasetId(e.target.value)}
                  className="flex h-8 flex-1 rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-subtle)] px-3 text-sm text-[var(--ds-text-body)] outline-none focus:border-[var(--ds-primary)] focus:ring-1 focus:ring-[var(--ds-primary)]"
                >
                  {datasets.map((ds) => (
                    <option key={ds.id} value={ds.id}>
                      {ds.filename} ({ds.row_count} {t("summary.rows")}, {ds.column_count} {t("summary.columns")})
                    </option>
                  ))}
                </select>
              ) : (
                <span className="text-sm text-[var(--ds-text-body)] flex-1">
                  {datasets[0].filename}
                  <Badge
                    variant="outline"
                    className="ml-2 border-[var(--ds-border)] text-[var(--ds-text-secondary)] text-[10px]"
                  >
                    {datasets[0].row_count} {t("summary.rows")}
                  </Badge>
                </span>
              )}
              <Button
                variant="ghost"
                size="sm"
                className="h-7 text-[var(--ds-text-secondary)] hover:text-[var(--ds-error)]"
                onClick={handleDeleteDataset}
                disabled={deleteDataset.isPending}
              >
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Data Summary + Preview (Collapsible) */}
      {datasetDetail && (
        <>
          <DataSummaryPanel dataset={datasetDetail} />

          <Collapsible open={previewOpen} onOpenChange={setPreviewOpen}>
            <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
              <CardHeader className="cursor-pointer pb-3 px-4 pt-4">
                <CollapsibleTrigger className="w-full">
                  <div className="flex items-center gap-2">
                    <Database className="h-4 w-4 text-[var(--ds-primary)]" />
                    <span className="text-sm font-semibold text-[var(--ds-text-body)]">
                      {t("preview.title")}
                    </span>
                    {previewOpen ? (
                      <ChevronDown className="ml-auto h-4 w-4 text-[var(--ds-text-secondary)]" />
                    ) : (
                      <ChevronRight className="ml-auto h-4 w-4 text-[var(--ds-text-secondary)]" />
                    )}
                  </div>
                </CollapsibleTrigger>
              </CardHeader>
              <CollapsibleContent>
                <CardContent className="pt-0">
                  <DataPreviewTable dataset={datasetDetail} />
                </CardContent>
              </CollapsibleContent>
            </Card>
          </Collapsible>
        </>
      )}

      {/* Analysis Mode Toggle + Run */}
      {hasDatasets && (
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardContent className="py-4 space-y-4">
            {/* Mode Toggle */}
            <div className="flex gap-1 rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-subtle)] p-1">
              <button
                onClick={() => setAnalysisMode("quantitative")}
                className={`flex items-center gap-1.5 rounded-md px-4 py-2 text-sm transition-colors flex-1 justify-center ${
                  analysisMode === "quantitative"
                    ? "bg-[var(--ds-primary-light)] text-[var(--ds-primary)]"
                    : "text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-body)]"
                }`}
              >
                <BarChart3 className="h-4 w-4" />
                {t("mode.quantitative")}
              </button>
              <button
                onClick={() => setAnalysisMode("qualitative")}
                className={`flex items-center gap-1.5 rounded-md px-4 py-2 text-sm transition-colors flex-1 justify-center ${
                  analysisMode === "qualitative"
                    ? "bg-[var(--ds-primary-light)] text-[var(--ds-primary)]"
                    : "text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-body)]"
                }`}
              >
                <BookOpen className="h-4 w-4" />
                {t("mode.qualitative")}
              </button>
            </div>

            {/* Run Analysis Button */}
            <div className="flex justify-center">
              <Button
                onClick={handleRunAnalysis}
                disabled={
                  !activeDatasetId ||
                  (analysisMode === "quantitative"
                    ? triggerQuant.isPending || isRunningQuant
                    : triggerQual.isPending || isRunningQual)
                }
                className="gap-2"
              >
                {(analysisMode === "quantitative" &&
                  (triggerQuant.isPending || isRunningQuant)) ||
                (analysisMode === "qualitative" &&
                  (triggerQual.isPending || isRunningQual)) ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <BarChart3 className="h-4 w-4" />
                )}
                {analysisMode === "quantitative"
                  ? isRunningQuant
                    ? quantTaskStatus?.progress_message || t("common.running")
                    : t("runQuantitative")
                  : isRunningQual
                    ? qualTaskStatus?.progress_message || t("common.running")
                    : t("runQualitative")}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Task Failed Messages */}
      {quantTaskStatus?.status === "failed" && (
        <Card className="border-[var(--ds-error)]/30 bg-[var(--ds-bg-card)]">
          <CardContent className="py-4 text-center">
            <p className="text-sm text-[var(--ds-error)]">{t("common.analysisFailed")}</p>
          </CardContent>
        </Card>
      )}
      {qualTaskStatus?.status === "failed" && (
        <Card className="border-[var(--ds-error)]/30 bg-[var(--ds-bg-card)]">
          <CardContent className="py-4 text-center">
            <p className="text-sm text-[var(--ds-error)]">{t("common.analysisFailed")}</p>
          </CardContent>
        </Card>
      )}

      {/* Quantitative Results */}
      {analysisMode === "quantitative" && (
        <>
          {quantLoading && (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-[var(--ds-text-secondary)]" />
            </div>
          )}
          {quantResults ? (
            <QuantitativeView results={quantResults} />
          ) : (
            !quantLoading &&
            !isRunningQuant &&
            hasDatasets && (
              <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
                <CardContent className="py-8 text-center">
                  <BarChart3 className="mx-auto h-8 w-8 text-[var(--ds-text-muted)] mb-2" />
                  <p className="text-sm text-[var(--ds-text-secondary)]">
                    {t("quantitative.noResults")}
                  </p>
                </CardContent>
              </Card>
            )
          )}
        </>
      )}

      {/* Qualitative Results */}
      {analysisMode === "qualitative" && (
        <>
          {qualLoading && (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-[var(--ds-text-secondary)]" />
            </div>
          )}
          {qualResults ? (
            <>
              <QualitativeCodingView results={qualResults} />
              <InterCoderReportView
                report={interCoderReport}
                isLoading={interCoderLoading || isRunningInterCoder}
                onRunReport={handleRunInterCoder}
                isTriggering={triggerInterCoder.isPending}
              />
            </>
          ) : (
            !qualLoading &&
            !isRunningQual &&
            hasDatasets && (
              <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
                <CardContent className="py-8 text-center">
                  <BookOpen className="mx-auto h-8 w-8 text-[var(--ds-text-muted)] mb-2" />
                  <p className="text-sm text-[var(--ds-text-secondary)]">
                    {t("qualitative.noResults")}
                  </p>
                </CardContent>
              </Card>
            )
          )}
        </>
      )}
    </div>
  );
}
