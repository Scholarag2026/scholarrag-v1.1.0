"use client";

import { useState, useEffect } from "react";
import { useTranslations } from "next-intl";
import { useQueryClient } from "@tanstack/react-query";
import {
  Loader2,
  Sprout,
  Plus,
  X,
  Network,
} from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { PaperCard } from "@/components/paper-card";
import { useTriggerSeedExpand } from "@/hooks/use-deep-search";
import { useProjectPapers, useAddPaper } from "@/hooks/use-papers";
import { useTaskPolling } from "@/hooks/use-task-polling";

interface SeedPapersPanelProps {
  projectId: string;
}

interface SeedExpandResult {
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
  seeds_resolved: number;
  total_expanded: number;
}

export function SeedPapersPanel({ projectId }: SeedPapersPanelProps) {
  const t = useTranslations("search");
  const queryClient = useQueryClient();

  const [doiInput, setDoiInput] = useState("");
  const [titleInput, setTitleInput] = useState("");
  const [dois, setDois] = useState<string[]>([]);
  const [titles, setTitles] = useState<string[]>([]);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [results, setResults] = useState<SeedExpandResult | null>(null);
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(
    new Set(),
  );

  const triggerSeedExpand = useTriggerSeedExpand();
  const { data: libraryData } = useProjectPapers(projectId);
  const addPaper = useAddPaper(projectId);

  const libraryDois = new Set(
    libraryData?.papers
      .map((pp) => pp.paper.doi?.toLowerCase())
      .filter(Boolean),
  );

  // Poll task status — bounded, backed off.
  const { data: taskStatus } = useTaskPolling<{
    status: string;
    progress: number;
    progress_message: string | null;
    result: SeedExpandResult | null;
  }>({ taskId, kind: "seed_expand" });

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
      toast.error(t("seedPapers.failed"));
      setTaskId(null);
    } else if (taskStatus.status === "cancelled") {
      setTaskId(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskStatus?.status, taskId]);

  const isExpanding =
    !!taskId &&
    taskStatus?.status !== "completed" &&
    taskStatus?.status !== "failed" &&
    taskStatus?.status !== "cancelled";

  function addDoi() {
    const trimmed = doiInput.trim();
    if (trimmed && !dois.includes(trimmed)) {
      setDois((prev) => [...prev, trimmed]);
      setDoiInput("");
    }
  }

  function addTitle() {
    const trimmed = titleInput.trim();
    if (trimmed && !titles.includes(trimmed)) {
      setTitles((prev) => [...prev, trimmed]);
      setTitleInput("");
    }
  }

  function removeDoi(index: number) {
    setDois((prev) => prev.filter((_, i) => i !== index));
  }

  function removeTitle(index: number) {
    setTitles((prev) => prev.filter((_, i) => i !== index));
  }

  function handleExpand() {
    if (dois.length === 0 && titles.length === 0) return;
    setResults(null);
    setSelectedIndices(new Set());
    triggerSeedExpand.mutate(
      {
        projectId,
        dois: dois.length > 0 ? dois : undefined,
        titles: titles.length > 0 ? titles : undefined,
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
    toast.success(t("seedPapers.addedToLibrary"));
  }

  return (
    <div className="space-y-4">
      {/* Seed Input */}
      <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
        <CardHeader className="pb-3">
          <CardTitle
            className="flex items-center gap-2 text-base text-[var(--ds-text-heading)]"
            style={{ fontFamily: "var(--font-heading)" }}
          >
            <Sprout className="size-4 text-[var(--ds-success)]" />
            {t("seedPapers.title")}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* DOI Input */}
          <div className="space-y-2">
            <Label className="text-xs text-[var(--ds-text-secondary)]">
              {t("seedPapers.doiLabel")}
            </Label>
            <div className="flex gap-2">
              <Input
                value={doiInput}
                onChange={(e) => setDoiInput(e.target.value)}
                placeholder="10.1234/example.doi"
                className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)]"
                style={{ fontFamily: "var(--font-body)" }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    addDoi();
                  }
                }}
              />
              <Button
                variant="ghost"
                className="shrink-0 text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
                onClick={addDoi}
                disabled={!doiInput.trim()}
              >
                <Plus className="size-4" />
              </Button>
            </div>
            {dois.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {dois.map((doi, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center gap-1 rounded-full border border-[var(--ds-border)] bg-[var(--ds-bg-page)] px-2.5 py-1 text-xs text-[var(--ds-text-secondary)]"
                  >
                    {doi}
                    <button
                      onClick={() => removeDoi(i)}
                      className="ml-0.5 text-[var(--ds-text-muted)] hover:text-[var(--ds-error)]"
                    >
                      <X className="size-3" />
                    </button>
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Title Input */}
          <div className="space-y-2">
            <Label className="text-xs text-[var(--ds-text-secondary)]">
              {t("seedPapers.titleLabel")}
            </Label>
            <div className="flex gap-2">
              <Input
                value={titleInput}
                onChange={(e) => setTitleInput(e.target.value)}
                placeholder={t("seedPapers.titlePlaceholder")}
                className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)]"
                style={{ fontFamily: "var(--font-body)" }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    addTitle();
                  }
                }}
              />
              <Button
                variant="ghost"
                className="shrink-0 text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
                onClick={addTitle}
                disabled={!titleInput.trim()}
              >
                <Plus className="size-4" />
              </Button>
            </div>
            {titles.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {titles.map((title, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center gap-1 rounded-full border border-[var(--ds-border)] bg-[var(--ds-bg-page)] px-2.5 py-1 text-xs text-[var(--ds-text-secondary)]"
                  >
                    {title.length > 50 ? `${title.slice(0, 50)}...` : title}
                    <button
                      onClick={() => removeTitle(i)}
                      className="ml-0.5 text-[var(--ds-text-muted)] hover:text-[var(--ds-error)]"
                    >
                      <X className="size-3" />
                    </button>
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Expand Button */}
          <Button
            onClick={handleExpand}
            disabled={
              (dois.length === 0 && titles.length === 0) ||
              triggerSeedExpand.isPending ||
              isExpanding
            }
            className="gap-2 bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
          >
            {triggerSeedExpand.isPending || isExpanding ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <Network className="size-4" />
            )}
            {t("seedPapers.expand")}
          </Button>
        </CardContent>
      </Card>

      {/* Progress */}
      {isExpanding && taskStatus && (
        <Card className="border-[var(--ds-primary)]/30 bg-[var(--ds-bg-card)]">
          <CardContent className="py-4 space-y-3">
            <div className="flex items-center gap-2">
              <Loader2 className="size-4 animate-spin text-[var(--ds-primary)]" />
              <span
                className="text-sm text-[var(--ds-text-secondary)]"
                style={{ fontFamily: "var(--font-body)" }}
              >
                {taskStatus.progress_message || t("seedPapers.expanding")}
              </span>
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

      {/* Results */}
      {results && (
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle
                className="flex items-center gap-2 text-base text-[var(--ds-text-heading)]"
                style={{ fontFamily: "var(--font-heading)" }}
              >
                {t("seedPapers.results")}
                <span className="rounded-full bg-[var(--ds-border)] px-2 py-0.5 text-xs font-normal text-[var(--ds-text-secondary)]">
                  {results.papers.length}
                </span>
                <span className="text-xs font-normal text-[var(--ds-text-muted)]">
                  ({t("seedPapers.seedsResolved", { count: results.seeds_resolved })})
                </span>
              </CardTitle>
              <div className="flex items-center gap-2">
                {selectedIndices.size > 0 && (
                  <Button
                    size="sm"
                    className="gap-1 bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
                    onClick={handleAddSelected}
                    disabled={addPaper.isPending}
                  >
                    <Plus className="size-3.5" />
                    {t("seedPapers.addSelected", {
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
          <CardContent>
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
