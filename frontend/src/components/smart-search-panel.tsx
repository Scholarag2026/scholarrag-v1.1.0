"use client";

import { useState, useEffect } from "react";
import { useTranslations } from "next-intl";
import { useQueryClient } from "@tanstack/react-query";
import {
  Search,
  Loader2,
  Plus,
  X,
  BrainCircuit,
  MessageCircleQuestion,
  CheckCircle2,
  ClipboardList,
  Download,
  Info,
} from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { PaperCard } from "@/components/paper-card";
import {
  useTriggerSmartSearch,
  useRefineScope,
  resetCriteriaFlags,
  toggleCriterionFlag,
  criteriaStagesFromFlags,
  type ScopeRefineResponse,
  type SmartSearchResult,
  type WosFilter,
} from "@/hooks/use-smart-search";
import { useProjectPapers, useAddPaper } from "@/hooks/use-papers";
import { useTaskPolling } from "@/hooks/use-task-polling";
import { apiFetch, fetchWithAuth, ApiError } from "@/lib/api";
import { downloadBlob, screeningRecordFilename } from "@/lib/download";
import {
  flowRows,
  roundLogRows,
  secondPassStageSummary,
  stopRuleSettings,
  summarizeProvenance,
} from "@/lib/provenance";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

const WOS_FILTER_OPTIONS: { value: WosFilter; labelKey: "venueFilterAuto" | "venueFilterOn" | "venueFilterOff" }[] = [
  { value: "auto", labelKey: "venueFilterAuto" },
  { value: "on", labelKey: "venueFilterOn" },
  { value: "off", labelKey: "venueFilterOff" },
];

interface SmartSearchPanelProps {
  projectId: string;
}

export function SmartSearchPanel({ projectId }: SmartSearchPanelProps) {
  const t = useTranslations("smartSearch");
  const tPolling = useTranslations("polling");
  const queryClient = useQueryClient();

  const [query, setQuery] = useState("");
  const [originalQuery, setOriginalQuery] = useState("");
  const [taskId, setTaskId] = useState<string | null>(null);
  const [results, setResults] = useState<SmartSearchResult | null>(null);
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(
    new Set(),
  );
  // Which results list is shown: the included papers, or the NEEDS_REVIEW ones flagged
  // for a human to read. Bulk add/select stays bound to results.papers regardless of
  // this filter -- see toggleSelectAll/handleAddSelected.
  const [resultsFilter, setResultsFilter] = useState<"included" | "needs_review">("included");
  // Venue filter (Stage 1): "auto" applies the WoS journal list only when one is imported.
  const [wosFilter, setWosFilter] = useState<WosFilter>("auto");
  // Task id of the last completed search — the screening-record export is keyed by it.
  const [completedTaskId, setCompletedTaskId] = useState<string | null>(null);
  const [isDownloadingRecord, setIsDownloadingRecord] = useState(false);

  // Scope refinement state
  const [scopePhase, setScopePhase] = useState<"input" | "refining" | "ready" | "searching">("input");
  const [previousAnswers, setPreviousAnswers] = useState<{ question: string; answer: string }[]>([]);
  const [currentQuestion, setCurrentQuestion] = useState<ScopeRefineResponse | null>(null);
  const [selectedOptions, setSelectedOptions] = useState<Set<string>>(new Set());
  const [customAnswer, setCustomAnswer] = useState("");
  const [inclusionCriteria, setInclusionCriteria] = useState<string[]>([]);
  const [exclusionCriteria, setExclusionCriteria] = useState<string[]>([]);
  // The per-criterion "needs the full paper" checkbox. backend/app/api/smart_search.py
  // forwards inclusion_criteria_stages/exclusion_criteria_stages to the pipeline.
  // Parallel to inclusionCriteria/exclusionCriteria; reset to all-false wherever that
  // array itself is replaced, via resetCriteriaFlags (four sites).
  const [inclusionFullText, setInclusionFullText] = useState<boolean[]>([]);
  const [exclusionFullText, setExclusionFullText] = useState<boolean[]>([]);
  const [refinedTopic, setRefinedTopic] = useState<string | null>(null);
  // Save refined topic to project when scope refinement completes
  function saveRefinedTopicToProject(topic: string, inclusion: string[], exclusion: string[]) {
    apiFetch(`/projects/${projectId}/refined-topic`, {
      method: "POST",
      body: JSON.stringify({
        refined_topic: topic,
        inclusion_criteria: inclusion,
        exclusion_criteria: exclusion,
      }),
    }).then(() => {
      // Invalidate project cache so the header updates immediately
      // Only the project row changed (refined_topic + criteria). exact:true keeps this from
      // prefix-matching the 500-paper payload, the citation graph, drafts and analyses.
      queryClient.invalidateQueries({ queryKey: ["projects", projectId], exact: true });
    }).catch(() => {});
  }

  const triggerSmartSearch = useTriggerSmartSearch();
  const refineScope = useRefineScope();
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
    result: SmartSearchResult | null;
  }>({
    taskId,
    kind: "smart_search",
    onCancelled: () => {
      setTaskId(null);
      setScopePhase("input");
    },
  });

  // Handle task completion
  useEffect(() => {
    if (!taskStatus || !taskId) return;
    if (taskStatus.status === "completed") {
      if (taskStatus.result) {
        setResults(taskStatus.result);
        setCompletedTaskId(taskId);
      }
      queryClient.invalidateQueries({
        queryKey: ["projects", projectId, "papers"],
      });
      setTaskId(null);
      setScopePhase("input");
    } else if (taskStatus.status === "failed") {
      toast.error(t("failed"));
      setTaskId(null);
      setScopePhase("input");
    } else if (taskStatus.status === "cancelled") {
      setTaskId(null);
      setScopePhase("input");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskStatus?.status, taskId]);

  const isSearching =
    !!taskId &&
    !pollTimedOut &&
    taskStatus?.status !== "completed" &&
    taskStatus?.status !== "failed" &&
    taskStatus?.status !== "cancelled";

  // Step 1: User submits query → AI checks if scope is specific enough
  function handleStartRefining() {
    if (!query.trim()) return;
    setOriginalQuery(query.trim());
    setScopePhase("refining");
    setResults(null);
    setPreviousAnswers([]);
    setCurrentQuestion(null);

    refineScope.mutate(
      { query: query.trim(), previous_answers: [] },
      {
        onSuccess: (data) => {
          if (data.is_specific_enough) {
            // Scope is already specific — go straight to search
            setRefinedTopic(data.refined_topic);
            if (data.refined_topic) setQuery(data.refined_topic);
            setInclusionCriteria(data.inclusion_criteria);
            setExclusionCriteria(data.exclusion_criteria);
            setInclusionFullText(resetCriteriaFlags(data.inclusion_criteria.length));
            setExclusionFullText(resetCriteriaFlags(data.exclusion_criteria.length));
            setScopePhase("ready");
            if (data.refined_topic) saveRefinedTopicToProject(data.refined_topic, data.inclusion_criteria, data.exclusion_criteria);
          } else {
            setCurrentQuestion(data);
          }
        },
        onError: (err) => {
          toast.error(err.message);
          setScopePhase("input");
        },
      },
    );
  }

  // Toggle an option in multi-select
  function toggleOption(option: string) {
    setSelectedOptions((prev) => {
      const next = new Set(prev);
      if (next.has(option)) {
        next.delete(option);
      } else {
        next.add(option);
      }
      return next;
    });
  }

  // Step 2: User confirms selected options → AI checks again
  function handleConfirmAnswer() {
    if (selectedOptions.size === 0) return;
    const answer = Array.from(selectedOptions).join("; ");
    submitAnswer(answer);
  }

  function submitAnswer(answer: string) {
    const newAnswers = [...previousAnswers, { question: currentQuestion?.clarifying_question || "", answer }];
    setPreviousAnswers(newAnswers);
    setCurrentQuestion(null);
    setSelectedOptions(new Set());
    setCustomAnswer("");

    refineScope.mutate(
      { query: query.trim(), previous_answers: newAnswers },
      {
        onSuccess: (data) => {
          if (data.is_specific_enough) {
            setRefinedTopic(data.refined_topic);
            if (data.refined_topic) setQuery(data.refined_topic);
            setInclusionCriteria(data.inclusion_criteria);
            setExclusionCriteria(data.exclusion_criteria);
            setInclusionFullText(resetCriteriaFlags(data.inclusion_criteria.length));
            setExclusionFullText(resetCriteriaFlags(data.exclusion_criteria.length));
            setScopePhase("ready");
            if (data.refined_topic) saveRefinedTopicToProject(data.refined_topic, data.inclusion_criteria, data.exclusion_criteria);
          } else {
            setCurrentQuestion(data);
          }
        },
        onError: (err) => {
          toast.error(err.message);
          setScopePhase("input");
        },
      },
    );
  }

  // 409 means the venue filter was forced "on" without an imported WoS journal list (C3).
  function handleTriggerError(err: Error) {
    if (err instanceof ApiError && err.status === 409) {
      toast.error(t("wosRequired"));
    } else {
      toast.error(err.message);
    }
    setScopePhase("input");
  }

  async function handleDownloadScreeningRecord() {
    if (!completedTaskId) return;
    setIsDownloadingRecord(true);
    try {
      const res = await fetchWithAuth(
        `${API_URL}/tasks/${completedTaskId}/screening-record?format=csv`,
      );
      if (!res.ok) throw new Error(res.statusText);
      downloadBlob(await res.blob(), screeningRecordFilename(completedTaskId));
    } catch {
      toast.error(t("downloadFailed"));
    } finally {
      setIsDownloadingRecord(false);
    }
  }

  // Step 3: Scope is ready → start the search with criteria
  function handleRunSearch() {
    setScopePhase("searching");
    setSelectedIndices(new Set());
    setResultsFilter("included");
    triggerSmartSearch.mutate(
      {
        projectId,
        query: refinedTopic || query.trim(),
        inclusion_criteria: inclusionCriteria,
        exclusion_criteria: exclusionCriteria,
        inclusion_criteria_stages: criteriaStagesFromFlags(inclusionFullText),
        exclusion_criteria_stages: criteriaStagesFromFlags(exclusionFullText),
        wos_filter: wosFilter,
      },
      {
        onSuccess: (data) => setTaskId(data.task_id),
        onError: handleTriggerError,
      },
    );
  }

  // Skip refinement and search directly
  function handleSkipRefinement() {
    setScopePhase("searching");
    setSelectedIndices(new Set());
    setResultsFilter("included");
    setResults(null);
    triggerSmartSearch.mutate(
      { projectId, query: query.trim(), wos_filter: wosFilter },
      {
        onSuccess: (data) => setTaskId(data.task_id),
        onError: handleTriggerError,
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
    toast.success(t("addedToLibrary"));
  }

  // Go back to a specific question (undo answers from that point forward)
  function handleGoBack(toIndex: number) {
    const keptAnswers = previousAnswers.slice(0, toIndex);
    setPreviousAnswers(keptAnswers);
    setCurrentQuestion(null);
    setSelectedOptions(new Set());
    setCustomAnswer("");

    // Re-run scope refinement with the kept answers to get the right question
    refineScope.mutate(
      { query: originalQuery || query.trim(), previous_answers: keptAnswers },
      {
        onSuccess: (data) => {
          if (data.is_specific_enough) {
            setRefinedTopic(data.refined_topic);
            if (data.refined_topic) setQuery(data.refined_topic);
            setInclusionCriteria(data.inclusion_criteria);
            setExclusionCriteria(data.exclusion_criteria);
            setInclusionFullText(resetCriteriaFlags(data.inclusion_criteria.length));
            setExclusionFullText(resetCriteriaFlags(data.exclusion_criteria.length));
            setScopePhase("ready");
            if (data.refined_topic) saveRefinedTopicToProject(data.refined_topic, data.inclusion_criteria, data.exclusion_criteria);
          } else {
            setCurrentQuestion(data);
          }
        },
        onError: (err) => {
          toast.error(err.message);
          setScopePhase("input");
        },
      },
    );
  }

  function handleReset() {
    setScopePhase("input");
    setPreviousAnswers([]);
    setCurrentQuestion(null);
    setRefinedTopic(null);
    setInclusionCriteria([]);
    setExclusionCriteria([]);
    setInclusionFullText([]);
    setExclusionFullText([]);
  }

  // Screening record (v1.1.0): PRISMA-style flow counts and the screener provenance.
  const flow = flowRows(results?.flow);
  const screener = summarizeProvenance(results?.provenance?.screener);
  const rounds = results?.flow?.rounds ?? results?.rounds ?? null;
  const stopReason = results?.flow?.stop_reason ?? results?.stop_reason ?? null;
  // The per-round query log and the two stop-rule settings a run applied
  // (`provenance.rounds`/`provenance.settings`).
  const roundLog = roundLogRows(results?.provenance?.rounds);
  const stopSettings = stopRuleSettings(results?.provenance?.settings);
  // The second pass's own stage block, rendered as its own line, separate from the
  // per-round table above/below -- that table's own included_new/needs_review_new stay
  // the pass-1-and-guard-only provisional counts, never the stage's own (job-level, not
  // per-round) demotions.
  const secondPassStage = secondPassStageSummary(results?.provenance?.screener_second_pass);
  // Known stop reasons get a human label; anything else is shown verbatim so nothing is hidden.
  const stopReasonLabel =
    stopReason && t.has(`stopReasons.${stopReason}`)
      ? t(`stopReasons.${stopReason}`)
      : stopReason;
  // Same pattern for a flagged record's guard reason.
  function needsReviewReasonLabel(reason: string | null | undefined) {
    if (!reason) return null;
    return t.has(`needsReviewReasons.${reason}`) ? t(`needsReviewReasons.${reason}`) : reason;
  }
  // The reserved "TOPIC" criterion id (no numbered criterion covers an off-topic
  // mismatch) gets the same human label treatment as needsReviewReasonLabel above, instead of
  // showing the raw code on a demoted (NEEDS_REVIEW) record.
  function screeningCriterionLabel(criterion: string | null | undefined) {
    if (!criterion) return criterion;
    return criterion === "TOPIC" ? t("criterionTopic") : criterion;
  }
  // "I3" -> the inclusion criterion's own text, for the to-confirm note's hover title.
  // Falls back to the bare id when the criteria list has since changed shape (e.g. a
  // re-run with fewer criteria than the ids a stored result still names).
  function fullTextCriterionText(id: string): string {
    const match = /^I(\d+)$/.exec(id);
    if (!match) return id;
    const text = inclusionCriteria[Number(match[1]) - 1];
    return text ? `${id}: ${text}` : id;
  }
  const showScreeningRecord =
    !!results && (flow.length > 0 || !!screener || !!completedTaskId);

  return (
    <div className="space-y-4">
      {/* Search Input */}
      <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
        <CardHeader className="pb-3">
          <CardTitle
            className="flex items-center gap-2 text-base text-[var(--ds-text-heading)]"
            style={{ fontFamily: "var(--font-heading)" }}
          >
            <BrainCircuit className="size-4 text-[var(--ds-primary)]" />
            {t("title")}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex gap-2">
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={t("placeholder")}
              className="flex min-h-[38px] w-full resize-none rounded-md border border-[var(--ds-border)] bg-[var(--ds-bg-page)] px-3 py-2 text-sm text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--ds-primary)]"
              style={{ fontFamily: "var(--font-body)" }}
              disabled={scopePhase !== "input"}
              rows={query.length > 100 ? 3 : query.length > 50 ? 2 : 1}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && scopePhase === "input") {
                  e.preventDefault();
                  handleStartRefining();
                }
              }}
            />
            {scopePhase === "input" && (
              <Button
                onClick={handleStartRefining}
                disabled={!query.trim() || refineScope.isPending}
                className="shrink-0 gap-2 bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
              >
                {refineScope.isPending ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <Search className="size-4" />
                )}
                {t("run")}
              </Button>
            )}
            {(scopePhase === "refining" || scopePhase === "ready") && (
              <Button
                variant="ghost"
                onClick={handleReset}
                className="shrink-0 text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
              >
                <X className="size-4" />
              </Button>
            )}
          </div>

          {/* Venue filter (Stage 1, Web of Science journal list) */}
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1" data-testid="venue-filter">
            <span
              id="venue-filter-label"
              className="text-xs font-medium text-[var(--ds-text-secondary)]"
              style={{ fontFamily: "var(--font-body)" }}
            >
              {t("venueFilter")}
            </span>
            <div
              role="radiogroup"
              aria-labelledby="venue-filter-label"
              className="inline-flex overflow-hidden rounded-md border border-[var(--ds-border)]"
            >
              {WOS_FILTER_OPTIONS.map((option, index) => {
                const active = wosFilter === option.value;
                return (
                  <button
                    key={option.value}
                    type="button"
                    role="radio"
                    aria-checked={active}
                    // Roving tabindex: only the checked option is in the tab order; arrow keys move
                    // (and select) within the group, as the WAI-ARIA radiogroup pattern requires.
                    tabIndex={active ? 0 : -1}
                    onClick={() => setWosFilter(option.value)}
                    onKeyDown={(e) => {
                      const last = WOS_FILTER_OPTIONS.length - 1;
                      let next: number | null = null;
                      if (e.key === "ArrowRight" || e.key === "ArrowDown") next = index === last ? 0 : index + 1;
                      else if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = index === 0 ? last : index - 1;
                      else if (e.key === "Home") next = 0;
                      else if (e.key === "End") next = last;
                      if (next === null) return;
                      e.preventDefault();
                      setWosFilter(WOS_FILTER_OPTIONS[next].value);
                      const sibling = e.currentTarget.parentElement?.children[next];
                      if (sibling instanceof HTMLElement) sibling.focus();
                    }}
                    disabled={isSearching}
                    className={`px-2.5 py-1 text-xs transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--ds-primary)] ${
                      active
                        ? "bg-[var(--ds-primary)] text-white"
                        : "bg-[var(--ds-bg-page)] text-[var(--ds-text-secondary)] hover:bg-[var(--ds-bg-hover)]"
                    }`}
                  >
                    {t(option.labelKey)}
                  </button>
                );
              })}
            </div>
            <span className="text-xs text-[var(--ds-text-muted)]">{t("venueFilterHelp")}</span>
          </div>
        </CardContent>
      </Card>

      {/* Scope Refinement — Clarifying Question */}
      {scopePhase === "refining" && currentQuestion && (
        <Card className="border-[var(--ds-primary)]/30 bg-[var(--ds-bg-card)]">
          <CardContent className="py-4 space-y-4">
            <div className="flex items-start gap-3">
              <MessageCircleQuestion className="size-5 text-[var(--ds-primary)] shrink-0 mt-0.5" />
              <div className="space-y-3 flex-1">
                <p className="text-sm text-[var(--ds-text-heading)]" style={{ fontFamily: "var(--font-body)" }}>
                  {currentQuestion.clarifying_question}
                </p>
                <p className="text-xs text-[var(--ds-text-muted)]">Select one or more options:</p>
                <div className="flex flex-wrap gap-2">
                  {currentQuestion.options.map((option) => (
                    <Button
                      key={option}
                      variant="outline"
                      size="sm"
                      onClick={() => toggleOption(option)}
                      disabled={refineScope.isPending}
                      className={
                        selectedOptions.has(option)
                          ? "border-[var(--ds-primary)] bg-[var(--ds-primary-light)] text-[var(--ds-text-heading)] hover:bg-[var(--ds-primary-light)] hover:text-[var(--ds-text-heading)]"
                          : "border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)] hover:bg-[var(--ds-bg-hover)] hover:text-[var(--ds-text-heading)]"
                      }
                    >
                      {selectedOptions.has(option) ? "✓ " : ""}{option}
                    </Button>
                  ))}
                </div>
                {/* Confirm + custom input */}
                <div className="flex gap-2 mt-1">
                  <Input
                    value={customAnswer}
                    onChange={(e) => setCustomAnswer(e.target.value)}
                    placeholder="Or type your own answer..."
                    className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)] text-sm h-8"
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && customAnswer.trim()) {
                        submitAnswer(customAnswer.trim());
                        setCustomAnswer("");
                      }
                    }}
                  />
                  <Button
                    size="sm"
                    onClick={() => {
                      if (customAnswer.trim()) {
                        submitAnswer(customAnswer.trim());
                        setCustomAnswer("");
                      } else {
                        handleConfirmAnswer();
                      }
                    }}
                    disabled={(selectedOptions.size === 0 && !customAnswer.trim()) || refineScope.isPending}
                    className="shrink-0 bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
                  >
                    {refineScope.isPending ? (
                      <Loader2 className="size-4 animate-spin" />
                    ) : (
                      "Confirm"
                    )}
                  </Button>
                </div>
                {previousAnswers.length > 0 && (
                  <div className="mt-3 space-y-1.5">
                    <p className="text-xs font-medium text-[var(--ds-text-muted)]">Previous answers (click to go back):</p>
                    {previousAnswers.map((qa, i) => (
                      <button
                        key={i}
                        onClick={() => handleGoBack(i)}
                        disabled={refineScope.isPending}
                        className="block w-full text-left text-xs rounded px-2 py-1 hover:bg-[var(--ds-bg-hover)] transition-colors group"
                      >
                        <span className="text-[var(--ds-text-muted)]">Q{i + 1}: {qa.question}</span>
                        <br />
                        <span className="text-[var(--ds-text-secondary)]">→ {qa.answer}</span>
                        <span className="ml-2 text-[var(--ds-primary)] opacity-0 group-hover:opacity-100 transition-opacity">✎ change</span>
                      </button>
                    ))}
                  </div>
                )}
                <div className="flex gap-2 mt-2">
                  {previousAnswers.length > 0 && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleGoBack(previousAnswers.length - 1)}
                      disabled={refineScope.isPending}
                      className="text-xs text-[var(--ds-text-muted)] hover:text-[var(--ds-text-secondary)]"
                    >
                      ← Go back
                    </Button>
                  )}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleSkipRefinement}
                    className="text-xs text-[var(--ds-text-muted)] hover:text-[var(--ds-text-secondary)]"
                  >
                    Skip refinement, search now
                  </Button>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Scope Refinement — Loading */}
      {scopePhase === "refining" && !currentQuestion && refineScope.isPending && (
        <Card className="border-[var(--ds-primary)]/30 bg-[var(--ds-bg-card)]">
          <CardContent className="py-4">
            <div className="flex items-center gap-2">
              <Loader2 className="size-4 animate-spin text-[var(--ds-primary)]" />
              <span className="text-sm text-[var(--ds-text-secondary)]">Analyzing your research scope...</span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Scope Ready — Show criteria and start search */}
      {scopePhase === "ready" && (
        <Card className="border-[var(--ds-success)]/30 bg-[var(--ds-bg-card)]">
          <CardContent className="py-4 space-y-3">
            <div className="flex items-start gap-3">
              <CheckCircle2 className="size-5 text-[var(--ds-success)] shrink-0 mt-0.5" />
              <div className="space-y-2 flex-1">
                <p className="text-sm font-medium text-[var(--ds-text-heading)]">{refinedTopic}</p>
                {(inclusionCriteria.length > 0 || exclusionCriteria.length > 0) && (
                  <p className="text-xs text-[var(--ds-text-muted)]">{t("fullTextOnlyHelp")}</p>
                )}
                {inclusionCriteria.length > 0 && (
                  <div>
                    <p className="text-xs font-medium text-[var(--ds-success)]">Include:</p>
                    {inclusionCriteria.map((c, i) => (
                      <div key={i} className="flex items-center gap-2">
                        <p className="flex-1 text-xs text-[var(--ds-text-secondary)]">• {c}</p>
                        <label className="flex shrink-0 items-center gap-1 text-[11px] text-[var(--ds-text-muted)]">
                          <input
                            type="checkbox"
                            checked={inclusionFullText[i] ?? false}
                            onChange={() =>
                              setInclusionFullText((prev) => toggleCriterionFlag(prev, i))
                            }
                            className="size-3.5 rounded border-[var(--ds-border)] accent-[var(--ds-primary)]"
                          />
                          {t("fullTextOnly")}
                        </label>
                      </div>
                    ))}
                  </div>
                )}
                {exclusionCriteria.length > 0 && (
                  <div>
                    <p className="text-xs font-medium text-[var(--ds-error)]">Exclude:</p>
                    {exclusionCriteria.map((c, i) => (
                      <div key={i} className="flex items-center gap-2">
                        <p className="flex-1 text-xs text-[var(--ds-text-secondary)]">• {c}</p>
                        <label className="flex shrink-0 items-center gap-1 text-[11px] text-[var(--ds-text-muted)]">
                          <input
                            type="checkbox"
                            checked={exclusionFullText[i] ?? false}
                            onChange={() =>
                              setExclusionFullText((prev) => toggleCriterionFlag(prev, i))
                            }
                            className="size-3.5 rounded border-[var(--ds-border)] accent-[var(--ds-primary)]"
                          />
                          {t("fullTextOnly")}
                        </label>
                      </div>
                    ))}
                  </div>
                )}
                {previousAnswers.length > 0 && (
                  <div className="mt-2 space-y-1">
                    <p className="text-xs font-medium text-[var(--ds-text-muted)]">Your answers (click to change):</p>
                    {previousAnswers.map((qa, i) => (
                      <button
                        key={i}
                        onClick={() => { setScopePhase("refining"); handleGoBack(i); }}
                        className="block w-full text-left text-xs rounded px-2 py-1 hover:bg-[var(--ds-bg-hover)] transition-colors group"
                      >
                        <span className="text-[var(--ds-text-muted)]">Q{i + 1}: {qa.question}</span>
                        <br />
                        <span className="text-[var(--ds-text-secondary)]">→ {qa.answer}</span>
                        <span className="ml-2 text-[var(--ds-primary)] opacity-0 group-hover:opacity-100 transition-opacity">✎ change</span>
                      </button>
                    ))}
                  </div>
                )}
                <div className="flex gap-2 mt-2">
                  <Button
                    onClick={handleRunSearch}
                    disabled={triggerSmartSearch.isPending}
                    className="gap-2 bg-[var(--ds-primary)] text-[var(--ds-text-heading)] hover:bg-[var(--ds-primary-hover)]"
                  >
                    {triggerSmartSearch.isPending ? (
                      <Loader2 className="size-4 animate-spin" />
                    ) : (
                      <Search className="size-4" />
                    )}
                    Start Smart Search
                  </Button>
                  {previousAnswers.length > 0 && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => { setScopePhase("refining"); handleGoBack(previousAnswers.length - 1); }}
                      className="text-[var(--ds-text-muted)] hover:text-[var(--ds-text-secondary)]"
                    >
                      ← Go back
                    </Button>
                  )}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

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
                {taskStatus.progress_message || t("searching")}
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
                {t("results")}
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
                    ? t("deselectAll")
                    : t("selectAll")}
                </Button>
                {selectedIndices.size > 0 && (
                  <Button
                    size="sm"
                    className="gap-1 bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
                    onClick={handleAddSelected}
                    disabled={addPaper.isPending}
                  >
                    <Plus className="size-3.5" />
                    {t("addSelected", {
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
          <CardContent className="space-y-2">
            {results.stage1_applied === false && (
              <div
                role="status"
                data-testid="venue-filter-notice"
                className="flex items-start gap-2 rounded-md border border-[var(--ds-border)] bg-[var(--ds-bg-page)] px-3 py-2 text-xs text-[var(--ds-text-secondary)]"
              >
                <Info className="mt-0.5 size-3.5 shrink-0 text-[var(--ds-primary)]" />
                <span>
                  {results.criteria?.wos_filter === "off"
                    ? t("venueFilterDisabled")
                    : t("venueFilterInactive")}
                </span>
              </div>
            )}
            {!!results.needs_review?.length && (
              <div className="flex flex-wrap gap-2" data-testid="results-filter">
                <button
                  type="button"
                  onClick={() => setResultsFilter("included")}
                  className={`rounded-full px-2.5 py-1 text-xs transition-colors ${
                    resultsFilter === "included"
                      ? "bg-[var(--ds-primary)] text-white"
                      : "bg-[var(--ds-bg-page)] text-[var(--ds-text-secondary)] hover:bg-[var(--ds-bg-hover)]"
                  }`}
                >
                  {t("filterIncluded", { count: results.papers.length })}
                </button>
                <button
                  type="button"
                  onClick={() => setResultsFilter("needs_review")}
                  className={`rounded-full px-2.5 py-1 text-xs transition-colors ${
                    resultsFilter === "needs_review"
                      ? "bg-[var(--ds-primary)] text-white"
                      : "bg-[var(--ds-bg-page)] text-[var(--ds-text-secondary)] hover:bg-[var(--ds-bg-hover)]"
                  }`}
                >
                  {t("filterNeedsReview", { count: results.needs_review.length })}
                </button>
              </div>
            )}
            {resultsFilter === "needs_review" && (
              <p className="text-xs text-[var(--ds-text-muted)]">{t("needsReviewHelp")}</p>
            )}
            {resultsFilter === "included" &&
              results.papers.some((p) => (p.screening_to_confirm?.length ?? 0) > 0) && (
                <p className="text-xs text-[var(--ds-text-muted)]">{t("toConfirmHelp")}</p>
              )}
            {resultsFilter === "included" &&
              results.papers.map((paper, i) => {
                const isInLibrary = !!(
                  paper.doi && libraryDois.has(paper.doi.toLowerCase())
                );
                const toConfirmIds = paper.screening_to_confirm ?? [];
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
                        footer={
                          paper.screening_reason ? (
                            <span>
                              <span className="font-medium">{t("screeningReason")}:</span>{" "}
                              {paper.screening_reason}
                            </span>
                          ) : undefined
                        }
                        verificationBadge={
                          paper.wos_collection || toConfirmIds.length > 0 ? (
                            <>
                              {paper.wos_collection && (
                                <Badge
                                  variant="outline"
                                  className="border-[var(--ds-success)]/40 bg-[var(--ds-primary)]/10 text-[var(--ds-success)]"
                                >
                                  {paper.wos_collection || "WoS"}
                                </Badge>
                              )}
                              {toConfirmIds.length > 0 && (
                                <span
                                  className="text-xs text-[var(--ds-text-muted)]"
                                  title={toConfirmIds.map(fullTextCriterionText).join("; ")}
                                >
                                  {t("toConfirm", { ids: toConfirmIds.join(", ") })}
                                </span>
                              )}
                            </>
                          ) : undefined
                        }
                      />
                    </div>
                  </div>
                );
              })}
            {resultsFilter === "needs_review" &&
              (results.needs_review ?? []).map((paper, i) => {
                const isInLibrary = !!(
                  paper.doi && libraryDois.has(paper.doi.toLowerCase())
                );
                return (
                  <div
                    key={`${paper.doi || paper.title}-needs-review-${i}`}
                    className="flex items-start gap-3"
                  >
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
                        footer={
                          paper.screening_criterion ||
                          paper.screening_quote ||
                          paper.screening_guard_reason ||
                          paper.screening_reason ? (
                            <span>
                              {paper.screening_guard_reason && (
                                <>
                                  <span className="font-medium">{t("needsReviewReason")}:</span>{" "}
                                  {needsReviewReasonLabel(paper.screening_guard_reason)}
                                  <br />
                                </>
                              )}
                              {paper.screening_criterion && (
                                <>
                                  <span className="font-medium">
                                    {t("screeningCriterion")}:
                                  </span>{" "}
                                  {screeningCriterionLabel(paper.screening_criterion)}{" "}
                                </>
                              )}
                              {paper.screening_quote && (
                                <>
                                  <span className="font-medium">{t("screeningQuote")}:</span>{" "}
                                  &ldquo;{paper.screening_quote}&rdquo;
                                </>
                              )}
                              {!paper.screening_criterion &&
                                !paper.screening_quote &&
                                paper.screening_reason && (
                                  <>
                                    <span className="font-medium">{t("screeningReason")}:</span>{" "}
                                    {paper.screening_reason}
                                  </>
                                )}
                            </span>
                          ) : undefined
                        }
                        verificationBadge={
                          paper.wos_collection ? (
                            <Badge
                              variant="outline"
                              className="border-[var(--ds-success)]/40 bg-[var(--ds-primary)]/10 text-[var(--ds-success)]"
                            >
                              {paper.wos_collection || "WoS"}
                            </Badge>
                          ) : undefined
                        }
                      />
                    </div>
                  </div>
                );
              })}
          </CardContent>
        </Card>
      )}

      {/* Screening record: flow counts, model provenance, CSV export */}
      {showScreeningRecord && (
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]" data-testid="screening-record">
          <CardHeader className="pb-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <CardTitle
                className="flex items-center gap-2 text-base text-[var(--ds-text-heading)]"
                style={{ fontFamily: "var(--font-heading)" }}
              >
                <ClipboardList className="size-4 text-[var(--ds-primary)]" />
                {t("screeningRecord")}
              </CardTitle>
              {completedTaskId && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleDownloadScreeningRecord}
                  disabled={isDownloadingRecord}
                  className="gap-1 border-[var(--ds-border)] text-[var(--ds-text-heading)]"
                >
                  {isDownloadingRecord ? (
                    <Loader2 className="size-3.5 animate-spin" />
                  ) : (
                    <Download className="size-3.5" />
                  )}
                  {t("downloadScreeningRecord")}
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <p
              className="text-xs text-[var(--ds-text-muted)]"
              style={{ fontFamily: "var(--font-body)" }}
            >
              {t("screeningRecordHelp")}
            </p>
            {flow.length > 0 && (
              <dl className="grid grid-cols-2 gap-2 sm:grid-cols-3" data-testid="screening-flow">
                {flow.map(({ key, value }) => (
                  <div
                    key={key}
                    className="flex items-baseline justify-between gap-2 rounded-md bg-[var(--ds-bg-page)] px-2.5 py-1.5"
                  >
                    <dt className="text-xs text-[var(--ds-text-secondary)]">{t(`flow.${key}`)}</dt>
                    <dd className="text-sm font-medium tabular-nums text-[var(--ds-text-heading)]">
                      {value}
                    </dd>
                  </div>
                ))}
              </dl>
            )}
            {(rounds !== null || stopReason) && (
              <p className="text-xs text-[var(--ds-text-muted)]">
                {rounds !== null && t("rounds", { count: rounds })}
                {rounds !== null && stopReason && " · "}
                {stopReasonLabel && t("stopReason", { reason: stopReasonLabel })}
              </p>
            )}
            {stopSettings && (
              <p className="text-xs text-[var(--ds-text-muted)]" data-testid="screening-stop-settings">
                {t("screeningStopSettings", {
                  minRounds: stopSettings.minRounds,
                  patience: stopSettings.dryRoundPatience,
                })}
              </p>
            )}
            {screener && (
              <p
                className="text-xs text-[var(--ds-text-muted)]"
                style={{ fontFamily: "var(--font-body)" }}
                data-testid="screening-model-line"
              >
                {t("modelLine", {
                  models: screener.models,
                  temperature: screener.temperature,
                  version: screener.promptVersion,
                })}
              </p>
            )}
            {/* The inclusion-only second pass's own stage, run once for the whole job,
                shown as its own line -- never folded into the per-round table below, whose
                own included/flagged counts stay the pass-1-and-guard-only provisional ones. */}
            {secondPassStage && (
              <p
                className="text-xs text-[var(--ds-text-muted)]"
                style={{ fontFamily: "var(--font-body)" }}
                data-testid="screening-second-pass-stage-line"
              >
                {t("secondPassStageLine", {
                  candidates: secondPassStage.candidates,
                  demoted: secondPassStage.demotions,
                  unavailable: secondPassStage.unavailable,
                  seconds: Math.round(secondPassStage.wallTimeSeconds),
                })}
              </p>
            )}
            {/* Round-by-round query log: a plain table, one heading row per round with its
                own screened/included/flagged totals, followed by one row per query that
                round issued. */}
            {roundLog.length > 0 && (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs" data-testid="screening-round-log">
                  <caption className="mb-1 text-left text-xs font-medium text-[var(--ds-text-heading)]" style={{ fontFamily: "var(--font-heading)" }}>
                    {t("screeningRoundsTitle")}
                  </caption>
                  <thead>
                    <tr className="text-[var(--ds-text-muted)]">
                      <th className="py-1 pr-2 font-medium">{t("screeningRoundQuery")}</th>
                      <th className="py-1 pr-2 font-medium">{t("screeningRoundReturned")}</th>
                      <th className="py-1 font-medium">{t("screeningRoundNewUnique")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {roundLog.map((row, i) =>
                      row.kind === "round" ? (
                        <tr key={`round-${row.round}-${i}`} className="bg-[var(--ds-bg-page)]">
                          <td
                            colSpan={3}
                            className="py-1.5 px-2 font-medium text-[var(--ds-text-heading)]"
                          >
                            {t("screeningRoundHeader", {
                              round: row.round,
                              screened: row.screened,
                              includedNew: row.includedNew,
                              needsReviewNew: row.needsReviewNew,
                            })}
                          </td>
                        </tr>
                      ) : (
                        <tr key={`query-${row.round}-${i}`}>
                          <td className="py-1 pr-2 text-[var(--ds-text-secondary)]">{row.query}</td>
                          <td className="py-1 pr-2 tabular-nums text-[var(--ds-text-secondary)]">
                            {row.returned}
                          </td>
                          <td className="py-1 tabular-nums text-[var(--ds-text-secondary)]">
                            {row.newUnique}
                          </td>
                        </tr>
                      ),
                    )}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
