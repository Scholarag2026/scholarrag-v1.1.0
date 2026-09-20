// frontend/src/components/papers-tab.tsx
"use client";

import { useState, useEffect, useCallback } from "react";
import { Search, Loader2, Download, FileText, X, ShieldCheck, BrainCircuit, Upload } from "lucide-react";
import { toast } from "sonner";
import { useTranslations } from "next-intl";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { PaperCard } from "@/components/paper-card";
import { QualityBadge } from "@/components/quality-badge";
import { IndicatorChips, VerificationBadge } from "@/components/verification-badge";
import { useQueryClient } from "@tanstack/react-query";
import { fetchWithAuth } from "@/lib/api";
import { useProjectPapers, useAddPaper, useRemovePaper, useAcquireFullTexts, useDeepAnalyze } from "@/hooks/use-papers";
import { Badge } from "@/components/ui/badge";
import { useSearch, SearchPaper } from "@/hooks/use-search";
import { useTaskPolling } from "@/hooks/use-task-polling";
import {
  useVerifyReferences,
  useTaskResult,
  type PaperVerificationResult,
} from "@/hooks/use-verification";
import { useAnalyzeQuality, usePaperAnalyses } from "@/hooks/use-analysis";
import { ArticleUploadModal } from "@/components/article-upload-modal";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

interface PapersTabProps {
  projectId: string;
}

export function PapersTab({ projectId }: PapersTabProps) {
  const t = useTranslations("papers");
  const tPolling = useTranslations("polling");
  const { data: libraryData, isLoading: libraryLoading } = useProjectPapers(projectId);
  const addPaper = useAddPaper(projectId);
  const removePaper = useRemovePaper(projectId);
  const {
    isSearching,
    progress,
    progressMessage,
    results,
    sources,
    error: searchError,
    timedOut: searchTimedOut,
    isCancelling: isCancellingSearch,
    maxDurationMinutes: searchMaxMinutes,
    cancel: cancelSearch,
    resume: resumeSearchPolling,
    search,
    clearResults,
  } = useSearch(projectId);

  const [query, setQuery] = useState("");
  const [yearFrom, setYearFrom] = useState("");
  const [yearTo, setYearTo] = useState("");
  const [minCitations, setMinCitations] = useState("");
  const [showFilters, setShowFilters] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);

  // Acquire full texts state
  const [selectedPaperIds, setSelectedPaperIds] = useState<Set<string>>(new Set());
  const [acquireTaskId, setAcquireTaskId] = useState<string | null>(null);
  const acquireFullTexts = useAcquireFullTexts(projectId);
  const queryClient = useQueryClient();

  // Deep analyze state
  const [deepAnalyzeTaskId, setDeepAnalyzeTaskId] = useState<string | null>(null);
  const deepAnalyze = useDeepAnalyze(projectId);

  // Deep analyze task polling
  const { data: deepAnalyzeTaskData } = useTaskPolling<{
    status: string;
    progress: number;
    progress_message: string | null;
    result: unknown | null;
    error: string | null;
  }>({ taskId: deepAnalyzeTaskId, kind: "deep_analysis" });

  const isDeepAnalyzing = !!deepAnalyzeTaskId && deepAnalyzeTaskData?.status !== "completed" && deepAnalyzeTaskData?.status !== "failed";

  // Verification state
  const [verifyTaskId, setVerifyTaskId] = useState<string | null>(null);
  const [verificationResults, setVerificationResults] = useState<
    Map<string, PaperVerificationResult>
  >(new Map());

  const analyzeQuality = useAnalyzeQuality();
  const { data: analysesData } = usePaperAnalyses(projectId);
  const qualityMap = new Map(
    (analysesData?.analyses ?? []).map((a) => [a.paper_id, a.quality_score])
  );

  const verifyMutation = useVerifyReferences(projectId);
  const { data: taskData } = useTaskResult(verifyTaskId);

  const isVerifying =
    !!verifyTaskId &&
    taskData?.status !== "completed" &&
    taskData?.status !== "failed";

  // Acquire full texts task polling
  const { data: acquireTaskData } = useTaskPolling<{
    status: string;
    progress: number;
    progress_message: string | null;
    result: { acquired: number; abstract_only: number } | null;
    error: string | null;
  }>({ taskId: acquireTaskId, kind: "full_text" });

  const isAcquiring = !!acquireTaskId && acquireTaskData?.status !== "completed" && acquireTaskData?.status !== "failed";

  // Handle verification completion
  useEffect(() => {
    if (!taskData || !verifyTaskId) return;

    if (taskData.status === "completed" && taskData.result) {
      const resultMap = new Map<string, PaperVerificationResult>();
      for (const r of taskData.result.results) {
        resultMap.set(r.paper_id, r);
      }
      setVerificationResults(resultMap);
      setVerifyTaskId(null);
      toast.success(
        t("verifyComplete", {
          passed: taskData.result.passed,
          failed: taskData.result.failed,
          warnings: taskData.result.warnings,
        }),
      );
    } else if (taskData.status === "failed") {
      setVerifyTaskId(null);
      toast.error(taskData.error || t("verifyFailed"));
    }
  }, [taskData, verifyTaskId, t]);

  // Handle acquisition completion
  useEffect(() => {
    if (!acquireTaskData || !acquireTaskId) return;
    if (acquireTaskData.status === "completed") {
      setAcquireTaskId(null);
      setSelectedPaperIds(new Set());
      toast.success(t("acquireComplete"));
      // Refresh papers to get updated metadata
      queryClient.invalidateQueries({ queryKey: ["projects", projectId, "papers"] });
    } else if (acquireTaskData.status === "failed") {
      setAcquireTaskId(null);
      toast.error(acquireTaskData.error || t("acquireFailed"));
    }
  }, [acquireTaskData, acquireTaskId, t, queryClient, projectId]);

  // Handle deep analyze completion
  useEffect(() => {
    if (!deepAnalyzeTaskData || !deepAnalyzeTaskId) return;
    if (deepAnalyzeTaskData.status === "completed") {
      setDeepAnalyzeTaskId(null);
      toast.success(t("deepAnalyzeComplete"));
      queryClient.invalidateQueries({ queryKey: ["projects", projectId, "papers"] });
    } else if (deepAnalyzeTaskData.status === "failed") {
      setDeepAnalyzeTaskId(null);
      toast.error(deepAnalyzeTaskData.error || t("deepAnalyzeFailed"));
    }
  }, [deepAnalyzeTaskData, deepAnalyzeTaskId, t, queryClient, projectId]);

  const handleVerifyAll = useCallback(() => {
    verifyMutation.mutate(
      { paper_ids: [] },
      {
        onSuccess: (data) => {
          setVerifyTaskId(data.task_id);
          toast.info(t("verifyStarted"));
        },
        onError: (err) => {
          toast.error(err.message || t("verifyFailed"));
        },
      },
    );
  }, [verifyMutation, t]);

  // Track which search results have been added (by DOI or title)
  const libraryDois = new Set(
    libraryData?.papers.map((pp) => pp.paper.doi?.toLowerCase()).filter(Boolean),
  );

  function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    search({
      query: query.trim(),
      year_from: yearFrom ? parseInt(yearFrom) : undefined,
      year_to: yearTo ? parseInt(yearTo) : undefined,
      min_citations: minCitations ? parseInt(minCitations) : undefined,
    });
  }

  function handleAddPaper(paper: SearchPaper) {
    addPaper.mutate({
      doi: paper.doi,
      title: paper.title,
      authors: paper.authors,
      year: paper.year,
      journal_name: paper.journal_name,
      journal_issn: paper.journal_issn,
      citation_count: paper.citation_count,
      abstract: paper.abstract,
      source_api: paper.source_api,
      external_id: paper.external_id,
      full_text_url: paper.full_text_url,
      wos_collection: paper.wos_collection,
    });
  }

  async function handleExport() {
    try {
      const res = await fetchWithAuth(
        `${API_URL}/projects/${projectId}/papers/export?format=bibtex`,
      );
      if (!res.ok) throw new Error("Export failed");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "references.bib";
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      // Error handled silently
    }
  }

  function toggleSelectAll() {
    if (!libraryData) return;
    if (selectedPaperIds.size === libraryData.papers.length) {
      setSelectedPaperIds(new Set());
    } else {
      setSelectedPaperIds(new Set(libraryData.papers.map(pp => pp.paper.id)));
    }
  }

  function handleAcquireFullTexts() {
    acquireFullTexts.mutate(
      { paper_ids: Array.from(selectedPaperIds) },
      {
        onSuccess: (data) => {
          setAcquireTaskId(data.task_id);
          toast.info(t("acquiring"));
        },
        onError: (err) => toast.error(err.message),
      }
    );
  }

  function handleDeepAnalyze() {
    deepAnalyze.mutate(undefined, {
      onSuccess: (data) => {
        setDeepAnalyzeTaskId(data.task_id);
        toast.info(t("deepAnalyzing"));
      },
      onError: (err) => toast.error(err.message),
    });
  }

  return (
    <div className="space-y-6">
      {/* Search Section */}
      <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
        <CardHeader className="pb-3">
          <CardTitle
            className="flex items-center gap-2 text-base text-[var(--ds-text-heading)]"
            style={{ fontFamily: "var(--font-heading)" }}
          >
            <Search className="size-4 text-[var(--ds-primary)]" />
            {t("searchTitle")}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSearch} className="space-y-3">
            <div className="flex gap-2">
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={t("searchPlaceholder")}
                className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)]"
                style={{ fontFamily: "var(--font-body)" }}
              />
              <Button
                type="submit"
                disabled={isSearching || !query.trim()}
                className="bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
              >
                {isSearching ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <Search className="size-4" />
                )}
              </Button>
              <Button
                type="button"
                variant="ghost"
                className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
                onClick={() => setShowFilters(!showFilters)}
              >
                {t("filters")}
              </Button>
            </div>

            {showFilters && (
              <div className="flex flex-wrap gap-3">
                <div className="space-y-1">
                  <Label className="text-xs text-[var(--ds-text-secondary)]">{t("yearFrom")}</Label>
                  <Input
                    type="number"
                    value={yearFrom}
                    onChange={(e) => setYearFrom(e.target.value)}
                    placeholder="2015"
                    className="h-8 w-24 border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-sm text-[var(--ds-text-heading)]"
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-xs text-[var(--ds-text-secondary)]">{t("yearTo")}</Label>
                  <Input
                    type="number"
                    value={yearTo}
                    onChange={(e) => setYearTo(e.target.value)}
                    placeholder="2026"
                    className="h-8 w-24 border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-sm text-[var(--ds-text-heading)]"
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-xs text-[var(--ds-text-secondary)]">{t("minCitations")}</Label>
                  <Input
                    type="number"
                    value={minCitations}
                    onChange={(e) => setMinCitations(e.target.value)}
                    placeholder="10"
                    className="h-8 w-24 border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-sm text-[var(--ds-text-heading)]"
                  />
                </div>
              </div>
            )}
          </form>

          {/* Search Progress */}
          {isSearching && (
            <div className="mt-4 space-y-2">
              <div className="flex items-center gap-2">
                <Loader2 className="size-4 animate-spin text-[var(--ds-primary)]" />
                <span
                  className="text-sm text-[var(--ds-text-secondary)]"
                  style={{ fontFamily: "var(--font-body)" }}
                >
                  {progressMessage || t("searching")}
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  className="ml-auto shrink-0 text-[var(--ds-text-muted)] hover:text-[var(--ds-error)]"
                  onClick={cancelSearch}
                  disabled={isCancellingSearch}
                >
                  {isCancellingSearch ? (
                    <Loader2 className="size-3.5 animate-spin" />
                  ) : (
                    <X className="size-3.5" />
                  )}
                  {isCancellingSearch ? tPolling("cancelling") : tPolling("cancelJob")}
                </Button>
              </div>
              <div className="h-1 overflow-hidden rounded-full bg-[var(--ds-border)]">
                <div
                  className="h-full rounded-full bg-[var(--ds-primary)] transition-all duration-300"
                  style={{ width: `${Math.max(progress * 100, 5)}%` }}
                />
              </div>
            </div>
          )}

          {/* Polling gave up */}
          {searchTimedOut && (
            <div className="mt-4 flex flex-wrap items-center gap-3 rounded-lg border border-[var(--ds-warning)]/40 bg-[var(--ds-bg-card)] px-4 py-3">
              <span className="text-sm text-[var(--ds-text-secondary)]">
                {tPolling("stoppedBanner", { minutes: searchMaxMinutes })}
              </span>
              <Button variant="outline" size="sm" onClick={resumeSearchPolling}>
                {tPolling("retry")}
              </Button>
              <Button
                variant="destructive"
                size="sm"
                onClick={cancelSearch}
                disabled={isCancellingSearch}
              >
                {isCancellingSearch ? tPolling("cancelling") : tPolling("cancelJob")}
              </Button>
            </div>
          )}

          {/* Search Error */}
          {searchError && (
            <div className="mt-4 rounded-lg border border-[var(--ds-error)]/30 bg-[rgba(239,68,68,0.08)] px-4 py-3">
              <p className="text-sm text-[var(--ds-error)]">{searchError}</p>
            </div>
          )}

          {/* Search Results */}
          {results.length > 0 && (
            <div className="mt-4">
              <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <h4
                    className="text-sm font-medium text-[var(--ds-text-heading)]"
                    style={{ fontFamily: "var(--font-heading)" }}
                  >
                    {t("searchResults", { count: results.length })}
                  </h4>
                  {Object.entries(sources).map(([source, count]) =>
                    count > 0 ? (
                      <span
                        key={source}
                        className="rounded bg-[var(--ds-border)] px-1.5 py-0.5 text-xs text-[var(--ds-text-secondary)]"
                      >
                        {source === "openalex" ? "OpenAlex" : source}: {count}
                      </span>
                    ) : null,
                  )}
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
                  onClick={clearResults}
                >
                  <X className="mr-1 size-3.5" />
                  {t("clear")}
                </Button>
              </div>
              <div className="space-y-2">
                {results.map((paper, i) => (
                  <PaperCard
                    key={`${paper.doi || paper.title}-${i}`}
                    title={paper.title}
                    authors={paper.authors}
                    year={paper.year}
                    journalName={paper.journal_name}
                    citationCount={paper.citation_count}
                    doi={paper.doi}
                    abstract={paper.abstract}
                    sourceApi={paper.source_api}
                    fullTextUrl={paper.full_text_url}
                    onAdd={() => handleAddPaper(paper)}
                    isAdding={addPaper.isPending}
                    isInLibrary={
                      !!(paper.doi && libraryDois.has(paper.doi.toLowerCase()))
                    }
                    verificationBadge={
                      paper.wos_collection ? (
                        <Badge variant="outline" className="border-[var(--ds-success)]/40 bg-[var(--ds-primary)]/10 text-[var(--ds-success)]">
                          {paper.wos_collection}
                        </Badge>
                      ) : undefined
                    }
                  />
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Upload Section */}
      <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
        <CardHeader className="pb-3">
          <CardTitle
            className="flex items-center gap-2 text-base text-[var(--ds-text-heading)]"
            style={{ fontFamily: "var(--font-heading)" }}
          >
            <Upload className="size-4 text-[var(--ds-primary)]" />
            {t("uploadTitle")}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Button
            variant="outline"
            className="w-full border-dashed border-[var(--ds-border)] text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)] hover:border-[var(--ds-primary)]/50"
            onClick={() => setUploadOpen(true)}
          >
            <Upload className="mr-2 size-4" />
            {t("uploadDragDrop")}
          </Button>
          <p
            className="mt-2 text-center text-xs"
            style={{ color: "var(--ds-text-muted)" }}
          >
            {t("uploadAcceptedFormats")}
          </p>
        </CardContent>
      </Card>

      {/* Paper Library */}
      <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle
              className="flex items-center gap-2 text-base text-[var(--ds-text-heading)]"
              style={{ fontFamily: "var(--font-heading)" }}
            >
              <FileText className="size-4 text-[var(--ds-success)]" />
              {t("library")}
              {libraryData && (
                <span className="rounded-full bg-[var(--ds-border)] px-2 py-0.5 text-xs font-normal text-[var(--ds-text-secondary)]">
                  {libraryData.total}
                </span>
              )}
            </CardTitle>
            {libraryData && libraryData.total > 0 && (
              <div className="flex items-center gap-1">
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
                  onClick={toggleSelectAll}
                >
                  {selectedPaperIds.size > 0 && selectedPaperIds.size === libraryData?.papers.length
                    ? t("deselectAll")
                    : t("selectAll")}
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
                  onClick={handleAcquireFullTexts}
                  disabled={selectedPaperIds.size === 0 || acquireFullTexts.isPending || isAcquiring}
                >
                  {isAcquiring ? (
                    <Loader2 className="mr-1 size-3.5 animate-spin" />
                  ) : (
                    <Download className="mr-1 size-3.5" />
                  )}
                  {t("acquireFullTexts")}
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
                  onClick={() => analyzeQuality.mutate({ projectId })}
                  disabled={analyzeQuality.isPending}
                >
                  {analyzeQuality.isPending ? (
                    <Loader2 className="mr-1 size-3.5 animate-spin" />
                  ) : (
                    <BrainCircuit className="mr-1 size-3.5" />
                  )}
                  {t("analyzeAll")}
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
                  onClick={handleDeepAnalyze}
                  disabled={deepAnalyze.isPending || isDeepAnalyzing}
                >
                  {isDeepAnalyzing ? (
                    <Loader2 className="mr-1 size-3.5 animate-spin" />
                  ) : (
                    <BrainCircuit className="mr-1 size-3.5" />
                  )}
                  {t("deepAnalyze")}
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
                  onClick={handleVerifyAll}
                  disabled={isVerifying || verifyMutation.isPending}
                >
                  {isVerifying ? (
                    <Loader2 className="mr-1 size-3.5 animate-spin" />
                  ) : (
                    <ShieldCheck className="mr-1 size-3.5" />
                  )}
                  {isVerifying ? t("verifying") : t("verifyAll")}
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
                  onClick={handleExport}
                >
                  <Download className="mr-1 size-3.5" />
                  {t("exportBibtex")}
                </Button>
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {/* Verification Progress */}
          {isVerifying && taskData && (
            <div className="mb-4 space-y-2 rounded-lg border border-[var(--ds-primary)]/30 bg-[var(--ds-primary-light)] p-3">
              <div className="flex items-center gap-2">
                <Loader2 className="size-4 animate-spin text-[var(--ds-primary)]" />
                <span
                  className="text-sm text-[var(--ds-text-secondary)]"
                  style={{ fontFamily: "var(--font-body)" }}
                >
                  {taskData.progress_message || t("verifyingProgress")}
                </span>
              </div>
              <div className="h-1 overflow-hidden rounded-full bg-[var(--ds-border)]">
                <div
                  className="h-full rounded-full bg-[var(--ds-primary)] transition-all duration-300"
                  style={{ width: `${Math.max(taskData.progress * 100, 5)}%` }}
                />
              </div>
            </div>
          )}

          {/* Acquisition Progress */}
          {isAcquiring && acquireTaskData && (
            <div className="mb-4 space-y-2 rounded-lg border border-[var(--ds-success)]/30 bg-[rgba(5,150,105,0.08)] p-3">
              <div className="flex items-center gap-2">
                <Loader2 className="size-4 animate-spin text-[var(--ds-success)]" />
                <span className="text-sm text-[var(--ds-text-secondary)]" style={{ fontFamily: "var(--font-body)" }}>
                  {acquireTaskData.progress_message || t("acquiring")}
                </span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-[var(--ds-border)]">
                <div
                  className="h-full rounded-full bg-[var(--ds-primary)] transition-all duration-300"
                  style={{ width: `${Math.max(acquireTaskData.progress * 100, 5)}%` }}
                />
              </div>
            </div>
          )}

          {/* Deep Analyze Progress */}
          {isDeepAnalyzing && deepAnalyzeTaskData && (
            <div className="mb-4 space-y-2 rounded-lg border border-[var(--ds-accent)]/30 bg-[rgba(59,130,246,0.05)] p-3">
              <div className="flex items-center gap-2">
                <Loader2 className="size-4 animate-spin text-[var(--ds-accent)]" />
                <span className="text-sm text-[var(--ds-text-secondary)]">
                  {deepAnalyzeTaskData.progress_message || t("deepAnalyzing")}
                </span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-[var(--ds-border)]">
                <div className="h-full rounded-full bg-[var(--ds-accent)] transition-all duration-300"
                  style={{ width: `${Math.max(deepAnalyzeTaskData.progress * 100, 5)}%` }}
                />
              </div>
            </div>
          )}

          {libraryLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-24 bg-[var(--ds-bg-page)]" />
              ))}
            </div>
          ) : libraryData && libraryData.papers.length > 0 ? (
            <div className="space-y-2">
              {libraryData.papers.map((pp) => {
                const vr = verificationResults.get(pp.paper.id);
                return (
                  <div key={pp.id} className="flex items-start gap-3">
                    <input
                      type="checkbox"
                      checked={selectedPaperIds.has(pp.paper.id)}
                      onChange={() => {
                        setSelectedPaperIds(prev => {
                          const next = new Set(prev);
                          if (next.has(pp.paper.id)) next.delete(pp.paper.id);
                          else next.add(pp.paper.id);
                          return next;
                        });
                      }}
                      className="mt-4 size-4 shrink-0 rounded border-[var(--ds-border)] accent-[var(--ds-primary)]"
                    />
                    <div className="flex-1">
                      <PaperCard
                        title={pp.paper.title}
                        authors={pp.paper.authors}
                        year={pp.paper.year}
                        journalName={pp.paper.journal_name}
                        citationCount={pp.paper.citation_count}
                        doi={pp.paper.doi}
                        abstract={pp.paper.abstract}
                        sourceApi={pp.paper.source_api}
                        fullTextUrl={pp.paper.full_text_url}
                        onRemove={() => removePaper.mutate(pp.paper.id)}
                        isRemoving={removePaper.isPending}
                        verificationBadge={
                          <>
                            {/* Once a verification result carries a non-skipped WoS indicator, WoS is shown as a neutral chip instead */}
                            {pp.paper.wos_collection &&
                              !vr?.indicators?.some(
                                (i) => i.check_type === "wos_indexed" && i.status !== "skipped",
                              ) && (
                              <Badge variant="outline" className="border-[var(--ds-success)]/40 bg-[var(--ds-primary)]/10 text-[var(--ds-success)]">
                                {pp.paper.wos_collection}
                              </Badge>
                            )}
                            {qualityMap.has(pp.paper.id) && (
                              <QualityBadge score={qualityMap.get(pp.paper.id)!} />
                            )}
                            {isVerifying ? (
                              <VerificationBadge status="pending" />
                            ) : vr ? (
                              <>
                                <VerificationBadge
                                  status={vr.overall_status}
                                  checks={vr.checks}
                                />
                                <IndicatorChips
                                  indicators={vr.indicators}
                                  wosCollection={pp.paper.wos_collection}
                                />
                              </>
                            ) : null}
                            {pp.paper.metadata?.fulltext_status === "acquired" && (
                              <Badge variant="outline" className="border-[var(--ds-success)]/40 bg-[var(--ds-primary)]/10 text-[var(--ds-success)]">
                                {t("fullText")}
                              </Badge>
                            )}
                            {pp.paper.metadata?.fulltext_status === "abstract_only" && (
                              <Badge variant="outline" className="border-[var(--ds-warning)]/40 bg-[rgba(234,179,8,0.08)] text-[var(--ds-warning)]">
                                {t("abstractOnly")}
                              </Badge>
                            )}
                            {pp.paper.metadata?.deep_analysis && (
                              <Badge variant="outline" className="border-[var(--ds-accent)]/40 bg-[var(--ds-accent)]/10 text-[var(--ds-accent)]">
                                {t("analyzed")}
                              </Badge>
                            )}
                          </>
                        }
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="py-8 text-center">
              <div className="mx-auto mb-3 flex size-10 items-center justify-center rounded-full bg-[var(--ds-border)]">
                <FileText className="size-5 text-[var(--ds-text-muted)]" />
              </div>
              <p
                className="text-sm text-[var(--ds-text-secondary)]"
                style={{ fontFamily: "var(--font-body)" }}
              >
                {t("noLibraryPapers")}
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      <ArticleUploadModal
        projectId={projectId}
        open={uploadOpen}
        onClose={() => setUploadOpen(false)}
      />
    </div>
  );
}
