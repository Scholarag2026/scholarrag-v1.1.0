"use client";

import { useState, useEffect } from "react";
import { useTranslations } from "next-intl";
import { useQueryClient } from "@tanstack/react-query";
import {
  Search,
  Loader2,
  Filter,
  Plus,
  X,
  Zap,
} from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { PaperCard } from "@/components/paper-card";
import {
  useTriggerDeepSearch,
  type CoverageMetrics,
  type RoundMetrics,
} from "@/hooks/use-deep-search";
import { useProjectPapers, useAddPaper } from "@/hooks/use-papers";
import { useTaskPolling } from "@/hooks/use-task-polling";

interface DeepSearchPanelProps {
  projectId: string;
}

interface DeepSearchResult {
  papers: Array<{
    doi: string | null;
    title: string;
    authors: { name: string }[];
    year: number | null;
    journal_name: string | null;
    journal_issn?: string | null;
    citation_count: number | null;
    abstract: string | null;
    source_api: string;
    external_id: string | null;
    full_text_url: string | null;
    wos_collection?: string | null;
  }>;
  coverage: CoverageMetrics;
}

export function DeepSearchPanel({ projectId }: DeepSearchPanelProps) {
  const t = useTranslations("search");
  const tPolling = useTranslations("polling");
  const queryClient = useQueryClient();

  const [query, setQuery] = useState("");
  const [yearFrom, setYearFrom] = useState("");
  const [yearTo, setYearTo] = useState("");
  const [minCitations, setMinCitations] = useState("");
  const [maxRounds, setMaxRounds] = useState("3");
  const [showFilters, setShowFilters] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [results, setResults] = useState<DeepSearchResult | null>(null);
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(
    new Set(),
  );

  const triggerDeepSearch = useTriggerDeepSearch();
  const { data: libraryData } = useProjectPapers(projectId);
  const addPaper = useAddPaper(projectId);

  const libraryDois = new Set(
    libraryData?.papers
      .map((pp) => pp.paper.doi?.toLowerCase())
      .filter(Boolean),
  );

  // Poll task status — bounded, backed off, cancellable.
  const {
    data: taskStatus,
    timedOut: pollTimedOut,
    isCancelling,
    maxDurationMinutes,
    cancel: cancelSearch,
    resume: resumePolling,
  } = useTaskPolling<{
    status: string;
    progress: number;
    progress_message: string | null;
    result: DeepSearchResult | null;
  }>({
    taskId,
    kind: "deep_search",
    onCancelled: () => setTaskId(null),
  });

  // Handle task completion
  useEffect(() => {
    if (!taskStatus || !taskId) return;
    if (taskStatus.status === "completed") {
      if (taskStatus.result) {
        setResults(taskStatus.result);
      }
      queryClient.invalidateQueries({
        queryKey: ["projects", projectId, "papers"],
      });
      setTaskId(null);
    } else if (taskStatus.status === "failed") {
      toast.error(t("deepSearch.failed"));
      setTaskId(null);
    } else if (taskStatus.status === "cancelled") {
      setTaskId(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskStatus?.status, taskId]);

  const isSearching =
    !!taskId &&
    !pollTimedOut &&
    taskStatus?.status !== "completed" &&
    taskStatus?.status !== "failed" &&
    taskStatus?.status !== "cancelled";

  function handleRunSearch() {
    if (!query.trim()) return;
    setResults(null);
    setSelectedIndices(new Set());
    triggerDeepSearch.mutate(
      {
        projectId,
        query: query.trim(),
        year_from: yearFrom ? parseInt(yearFrom) : null,
        year_to: yearTo ? parseInt(yearTo) : null,
        min_citations: minCitations ? parseInt(minCitations) : null,
        max_rounds: maxRounds ? parseInt(maxRounds) : 3,
      },
      {
        onSuccess: (data) => setTaskId(data.task_id),
        onError: (err) => toast.error(err.message),
      },
    );
  }

  function toggleSelect(index: number) {
    setSelectedIndices((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  }

  function toggleSelectAll() {
    if (!results) return;
    if (selectedIndices.size === results.papers.length) {
      setSelectedIndices(new Set());
    } else {
      setSelectedIndices(new Set(results.papers.map((_, i) => i)));
    }
  }

  function handleAddSelected() {
    if (!results) return;
    for (const idx of selectedIndices) {
      const paper = results.papers[idx];
      if (paper.doi && libraryDois.has(paper.doi.toLowerCase())) continue;
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
    setSelectedIndices(new Set());
    toast.success(t("deepSearch.addedToLibrary"));
  }

  // Extract round info from task progress
  const rounds: RoundMetrics[] = results?.coverage?.rounds ?? [];

  return (
    <div className="space-y-4">
      {/* Search Input */}
      <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
        <CardHeader className="pb-3">
          <CardTitle
            className="flex items-center gap-2 text-base text-[var(--ds-text-heading)]"
            style={{ fontFamily: "var(--font-heading)" }}
          >
            <Zap className="size-4 text-[var(--ds-primary)]" />
            {t("deepSearch.title")}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex gap-2">
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={t("deepSearch.placeholder")}
              className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)]"
              style={{ fontFamily: "var(--font-body)" }}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleRunSearch();
              }}
            />
            <Button
              onClick={handleRunSearch}
              disabled={
                !query.trim() ||
                triggerDeepSearch.isPending ||
                isSearching
              }
              className="shrink-0 gap-2 bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
            >
              {triggerDeepSearch.isPending || isSearching ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Search className="size-4" />
              )}
              {t("deepSearch.run")}
            </Button>
            <Button
              type="button"
              variant="ghost"
              className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
              onClick={() => setShowFilters(!showFilters)}
            >
              <Filter className="size-4" />
            </Button>
          </div>

          {showFilters && (
            <div className="flex flex-wrap gap-3">
              <div className="space-y-1">
                <Label className="text-xs text-[var(--ds-text-secondary)]">
                  {t("deepSearch.yearFrom")}
                </Label>
                <Input
                  type="number"
                  value={yearFrom}
                  onChange={(e) => setYearFrom(e.target.value)}
                  placeholder="2015"
                  className="h-8 w-24 border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-sm text-[var(--ds-text-heading)]"
                />
              </div>
              <div className="space-y-1">
                <Label className="text-xs text-[var(--ds-text-secondary)]">
                  {t("deepSearch.yearTo")}
                </Label>
                <Input
                  type="number"
                  value={yearTo}
                  onChange={(e) => setYearTo(e.target.value)}
                  placeholder="2026"
                  className="h-8 w-24 border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-sm text-[var(--ds-text-heading)]"
                />
              </div>
              <div className="space-y-1">
                <Label className="text-xs text-[var(--ds-text-secondary)]">
                  {t("deepSearch.minCitations")}
                </Label>
                <Input
                  type="number"
                  value={minCitations}
                  onChange={(e) => setMinCitations(e.target.value)}
                  placeholder="10"
                  className="h-8 w-24 border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-sm text-[var(--ds-text-heading)]"
                />
              </div>
              <div className="space-y-1">
                <Label className="text-xs text-[var(--ds-text-secondary)]">
                  {t("deepSearch.maxRounds")}
                </Label>
                <Input
                  type="number"
                  value={maxRounds}
                  onChange={(e) => setMaxRounds(e.target.value)}
                  placeholder="3"
                  className="h-8 w-24 border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-sm text-[var(--ds-text-heading)]"
                />
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Progress */}
      {isSearching && taskStatus && (
        <Card className="border-[var(--ds-primary)]/30 bg-[var(--ds-bg-card)]">
          <CardContent className="py-4 space-y-3">
            <div className="flex items-center gap-2">
              <Loader2 className="size-4 animate-spin text-[var(--ds-primary)]" />
              <span
                className="text-sm text-[var(--ds-text-secondary)]"
                style={{ fontFamily: "var(--font-body)" }}
              >
                {taskStatus.progress_message || t("deepSearch.searching")}
              </span>
              <Button
                variant="ghost"
                size="sm"
                className="ml-auto shrink-0 text-[var(--ds-text-muted)] hover:text-[var(--ds-error)]"
                onClick={cancelSearch}
                disabled={isCancelling}
              >
                {isCancelling ? (
                  <Loader2 className="size-3.5 animate-spin" />
                ) : (
                  <X className="size-3.5" />
                )}
                {isCancelling ? tPolling("cancelling") : tPolling("cancelJob")}
              </Button>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-[var(--ds-border)]">
              <div
                className="h-full rounded-full bg-[var(--ds-primary)] transition-all duration-300"
                style={{
                  width: `${Math.max(taskStatus.progress * 100, 5)}%`,
                }}
              />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Polling gave up — offer to keep watching or cancel the job */}
      {pollTimedOut && taskId && (
        <Card className="border-[var(--ds-warning)]/40 bg-[var(--ds-bg-card)]">
          <CardContent className="flex flex-wrap items-center gap-3 py-4">
            <span
              className="text-sm text-[var(--ds-text-secondary)]"
              style={{ fontFamily: "var(--font-body)" }}
            >
              {tPolling("stoppedBanner", { minutes: maxDurationMinutes })}
            </span>
            <Button variant="outline" size="sm" onClick={resumePolling}>
              {tPolling("retry")}
            </Button>
            <Button
              variant="destructive"
              size="sm"
              onClick={cancelSearch}
              disabled={isCancelling}
            >
              {isCancelling ? tPolling("cancelling") : tPolling("cancelJob")}
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {results && (
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle
                className="flex items-center gap-2 text-base text-[var(--ds-text-heading)]"
                style={{ fontFamily: "var(--font-heading)" }}
              >
                {t("deepSearch.results")}
                <span className="rounded-full bg-[var(--ds-border)] px-2 py-0.5 text-xs font-normal text-[var(--ds-text-secondary)]">
                  {results.papers.length}
                </span>
              </CardTitle>
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
                  onClick={toggleSelectAll}
                >
                  {selectedIndices.size === results.papers.length
                    ? t("deepSearch.deselectAll")
                    : t("deepSearch.selectAll")}
                </Button>
                {selectedIndices.size > 0 && (
                  <Button
                    size="sm"
                    className="gap-1 bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
                    onClick={handleAddSelected}
                    disabled={addPaper.isPending}
                  >
                    <Plus className="size-3.5" />
                    {t("deepSearch.addSelected", {
                      count: selectedIndices.size,
                    })}
                  </Button>
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
                  onClick={() => setResults(null)}
                >
                  <X className="size-3.5" />
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Round summaries */}
            {rounds.length > 0 && (
              <div className="space-y-2">
                <h4
                  className="text-xs font-medium text-[var(--ds-text-secondary)]"
                  style={{ fontFamily: "var(--font-heading)" }}
                >
                  {t("deepSearch.roundSummary")}
                </h4>
                <div className="flex flex-wrap gap-2">
                  {rounds.map((round) => (
                    <Badge
                      key={round.round}
                      variant="outline"
                      className={`border-[var(--ds-border)] text-[var(--ds-text-secondary)] ${
                        round.stopped
                          ? "border-[var(--ds-warning)]/30 text-[var(--ds-warning)]"
                          : ""
                      }`}
                    >
                      R{round.round}: {round.strategy} (+
                      {round.new_papers})
                    </Badge>
                  ))}
                </div>
                <Separator className="bg-[var(--ds-border)]" />
              </div>
            )}

            {/* Paper list */}
            <div className="space-y-2">
              {results.papers.map((paper, i) => {
                const isInLibrary = !!(
                  paper.doi && libraryDois.has(paper.doi.toLowerCase())
                );
                return (
                  <div key={`${paper.doi || paper.title}-${i}`} className="flex items-start gap-3">
                    <input
                      type="checkbox"
                      checked={selectedIndices.has(i)}
                      onChange={() => toggleSelect(i)}
                      className="mt-4 size-4 rounded border-[var(--ds-border)] accent-[var(--ds-primary)]"
                    />
                    <div className="flex-1">
                      <PaperCard
                        title={paper.title}
                        authors={paper.authors}
                        year={paper.year}
                        journalName={paper.journal_name}
                        citationCount={paper.citation_count}
                        doi={paper.doi}
                        abstract={paper.abstract}
                        sourceApi={paper.source_api}
                        fullTextUrl={paper.full_text_url}
                        isInLibrary={isInLibrary}
                        onAdd={() =>
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
                          })
                        }
                        isAdding={addPaper.isPending}
                        verificationBadge={
                          paper.wos_collection ? (
                            <Badge variant="outline" className="border-[var(--ds-success)]/40 bg-[var(--ds-primary)]/10 text-[var(--ds-success)]">
                              {paper.wos_collection}
                            </Badge>
                          ) : undefined
                        }
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
