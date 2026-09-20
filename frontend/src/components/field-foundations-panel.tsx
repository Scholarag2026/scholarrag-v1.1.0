"use client";

import { useState, useEffect } from "react";
import { useTranslations } from "next-intl";
import { useQueryClient } from "@tanstack/react-query";
import {
  Loader2,
  BookOpen,
  Plus,
  X,
  CheckCircle2,
  AlertCircle,
  Landmark,
} from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import {
  useTriggerFieldFoundations,
  type FoundationalWork,
} from "@/hooks/use-deep-search";
import { useAddPaper } from "@/hooks/use-papers";
import { useTaskPolling } from "@/hooks/use-task-polling";

interface FieldFoundationsPanelProps {
  projectId: string;
}

interface FieldFoundationsResult {
  works: FoundationalWork[];
  topic: string;
}

export function FieldFoundationsPanel({
  projectId,
}: FieldFoundationsPanelProps) {
  const t = useTranslations("search");
  const queryClient = useQueryClient();

  const [topic, setTopic] = useState("");
  const [rqInput, setRqInput] = useState("");
  const [researchQuestions, setResearchQuestions] = useState<string[]>([]);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [results, setResults] = useState<FieldFoundationsResult | null>(null);

  const triggerFieldFoundations = useTriggerFieldFoundations();
  const addPaper = useAddPaper(projectId);

  // Poll task status — bounded, backed off.
  const { data: taskStatus } = useTaskPolling<{
    status: string;
    progress: number;
    progress_message: string | null;
    result: FieldFoundationsResult | null;
  }>({ taskId, kind: "field_foundations" });

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
      toast.error(t("fieldFoundations.failed"));
      setTaskId(null);
    } else if (taskStatus.status === "cancelled") {
      setTaskId(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskStatus?.status, taskId]);

  const isGenerating =
    !!taskId &&
    taskStatus?.status !== "completed" &&
    taskStatus?.status !== "failed" &&
    taskStatus?.status !== "cancelled";

  function addResearchQuestion() {
    const trimmed = rqInput.trim();
    if (trimmed && !researchQuestions.includes(trimmed)) {
      setResearchQuestions((prev) => [...prev, trimmed]);
      setRqInput("");
    }
  }

  function removeResearchQuestion(index: number) {
    setResearchQuestions((prev) => prev.filter((_, i) => i !== index));
  }

  function handleSuggest() {
    if (!topic.trim()) return;
    setResults(null);
    triggerFieldFoundations.mutate(
      {
        projectId,
        topic: topic.trim(),
        research_questions:
          researchQuestions.length > 0 ? researchQuestions : undefined,
      },
      {
        onSuccess: (data) => setTaskId(data.task_id),
        onError: (err) => toast.error(err.message),
      },
    );
  }

  function handleAddWork(work: FoundationalWork) {
    if (!work.matched_paper) return;
    const mp = work.matched_paper as Record<string, unknown>;
    addPaper.mutate(
      {
        doi: (mp.doi as string) || null,
        title: (mp.title as string) || work.suggested_title,
        authors: (mp.authors as { name: string }[]) || work.suggested_authors.map((a) => ({ name: a })),
        year: (mp.year as number) || work.suggested_year,
        journal_name: (mp.journal_name as string) || null,
        citation_count: (mp.citation_count as number) || null,
        abstract: (mp.abstract as string) || null,
        source_api: (mp.source_api as string) || "field_foundations",
        external_id: (mp.external_id as string) || null,
        full_text_url: (mp.full_text_url as string) || null,
        journal_issn: (mp.journal_issn as string) || null,
        wos_collection: (mp.wos_collection as string) || null,
      },
      {
        onSuccess: () => toast.success(t("fieldFoundations.added")),
      },
    );
  }

  return (
    <div className="space-y-4">
      {/* Input */}
      <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
        <CardHeader className="pb-3">
          <CardTitle
            className="flex items-center gap-2 text-base text-[var(--ds-text-heading)]"
            style={{ fontFamily: "var(--font-heading)" }}
          >
            <Landmark className="size-4 text-[var(--ds-primary)]" />
            {t("fieldFoundations.title")}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Topic */}
          <div className="space-y-2">
            <Label className="text-xs text-[var(--ds-text-secondary)]">
              {t("fieldFoundations.topicLabel")}
            </Label>
            <Input
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder={t("fieldFoundations.topicPlaceholder")}
              className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)]"
              style={{ fontFamily: "var(--font-body)" }}
            />
          </div>

          {/* Research Questions */}
          <div className="space-y-2">
            <Label className="text-xs text-[var(--ds-text-secondary)]">
              {t("fieldFoundations.questionsLabel")}
            </Label>
            <div className="flex gap-2">
              <Input
                value={rqInput}
                onChange={(e) => setRqInput(e.target.value)}
                placeholder={t("fieldFoundations.questionsPlaceholder")}
                className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)]"
                style={{ fontFamily: "var(--font-body)" }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    addResearchQuestion();
                  }
                }}
              />
              <Button
                variant="ghost"
                className="shrink-0 text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
                onClick={addResearchQuestion}
                disabled={!rqInput.trim()}
              >
                <Plus className="size-4" />
              </Button>
            </div>
            {researchQuestions.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {researchQuestions.map((rq, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center gap-1 rounded-full border border-[var(--ds-border)] bg-[var(--ds-bg-page)] px-2.5 py-1 text-xs text-[var(--ds-text-secondary)]"
                  >
                    {rq.length > 60 ? `${rq.slice(0, 60)}...` : rq}
                    <button
                      onClick={() => removeResearchQuestion(i)}
                      className="ml-0.5 text-[var(--ds-text-muted)] hover:text-[var(--ds-error)]"
                    >
                      <X className="size-3" />
                    </button>
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Suggest Button */}
          <Button
            onClick={handleSuggest}
            disabled={
              !topic.trim() ||
              triggerFieldFoundations.isPending ||
              isGenerating
            }
            className="gap-2 bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
          >
            {triggerFieldFoundations.isPending || isGenerating ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <BookOpen className="size-4" />
            )}
            {t("fieldFoundations.suggest")}
          </Button>
        </CardContent>
      </Card>

      {/* Progress */}
      {isGenerating && taskStatus && (
        <Card className="border-[var(--ds-primary)]/30 bg-[var(--ds-bg-card)]">
          <CardContent className="py-4 space-y-3">
            <div className="flex items-center gap-2">
              <Loader2 className="size-4 animate-spin text-[var(--ds-primary)]" />
              <span
                className="text-sm text-[var(--ds-text-secondary)]"
                style={{ fontFamily: "var(--font-body)" }}
              >
                {taskStatus.progress_message ||
                  t("fieldFoundations.generating")}
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
      {results && results.works.length > 0 && (
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardHeader className="pb-3">
            <CardTitle
              className="flex items-center gap-2 text-base text-[var(--ds-text-heading)]"
              style={{ fontFamily: "var(--font-heading)" }}
            >
              {t("fieldFoundations.results")}
              <span className="rounded-full bg-[var(--ds-border)] px-2 py-0.5 text-xs font-normal text-[var(--ds-text-secondary)]">
                {results.works.length}
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {results.works.map((work, i) => (
                <div
                  key={i}
                  className="rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-page)] p-4"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <h4
                          className="text-sm font-medium text-[var(--ds-text-heading)]"
                          style={{ fontFamily: "var(--font-heading)" }}
                        >
                          {work.suggested_title}
                        </h4>
                        <Badge
                          variant="outline"
                          className={
                            work.verified
                              ? "border-[var(--ds-success)]/50 text-[var(--ds-success)]"
                              : "border-[var(--ds-warning)]/50 text-[var(--ds-warning)]"
                          }
                        >
                          {work.verified ? (
                            <CheckCircle2 className="mr-1 size-3" />
                          ) : (
                            <AlertCircle className="mr-1 size-3" />
                          )}
                          {work.verified
                            ? t("fieldFoundations.verified")
                            : t("fieldFoundations.unverified")}
                        </Badge>
                      </div>
                      <p
                        className="mt-1 text-xs text-[var(--ds-text-secondary)]"
                        style={{ fontFamily: "var(--font-body)" }}
                      >
                        {work.suggested_authors.join(", ")} ({work.suggested_year})
                      </p>
                      <p
                        className="mt-2 text-xs text-[var(--ds-text-secondary)]"
                        style={{ fontFamily: "var(--font-body)" }}
                      >
                        {work.why_essential}
                      </p>
                    </div>
                    <div className="shrink-0">
                      {work.verified && work.matched_paper && (
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-8 text-[var(--ds-success)] hover:bg-[rgba(5,150,105,0.08)] hover:text-[var(--ds-success)]"
                          onClick={() => handleAddWork(work)}
                          disabled={addPaper.isPending}
                        >
                          <Plus className="mr-1 size-3.5" />
                          {t("fieldFoundations.addToLibrary")}
                        </Button>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Empty results */}
      {results && results.works.length === 0 && (
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardContent className="py-8 text-center">
            <BookOpen className="mx-auto size-8 text-[var(--ds-text-muted)] mb-2" />
            <p
              className="text-sm text-[var(--ds-text-secondary)]"
              style={{ fontFamily: "var(--font-body)" }}
            >
              {t("fieldFoundations.noResults")}
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
