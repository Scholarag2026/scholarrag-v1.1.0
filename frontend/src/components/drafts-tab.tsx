"use client";

import { useState, useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import { useTaskPolling } from "@/hooks/use-task-polling";
import {
  Plus,
  FileText,
  Trash2,
  Loader2,
  ArrowLeft,
  Download,
  Sparkles,
  ClipboardCheck,
  X,
} from "lucide-react";
import { toast } from "sonner";
import { useTranslations } from "next-intl";

import {
  useProjectDrafts,
  useDraft,
  useCreateDraft,
  useUpdateDraft,
  useDeleteDraft,
  useGenerateSection,
  useExportDraft,
  useCheckCompliance,
  type ComplianceReport,
  type GenerateSectionResult,
} from "@/hooks/use-drafts";
import { citationAuditCounts, summarizeProvenance } from "@/lib/provenance";
import {
  clampTargetWords,
  DEFAULT_TARGET_WORDS,
  MIN_TARGET_WORDS,
  MAX_TARGET_WORDS,
  describeGenerateCompletion,
} from "@/lib/writing";
import { useRefine } from "@/hooks/use-refine";
import { ClaimVerificationReport } from "@/components/claim-verification-report";
import { ComplianceReportView } from "@/components/compliance-report";
import { EmptyState } from "@/components/empty-state";
import { PaperEditor } from "@/components/paper-editor";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

const LANGUAGES = [
  { code: "en", label: "English" },
  { code: "zh", label: "\u4E2D\u6587" },
  { code: "ja", label: "\u65E5\u672C\u8A9E" },
  { code: "ko", label: "\uD55C\uAD6D\uC5B4" },
  { code: "de", label: "Deutsch" },
  { code: "fr", label: "Fran\u00E7ais" },
  { code: "es", label: "Espa\u00F1ol" },
  { code: "pt", label: "Portugu\u00EAs" },
];

interface DraftsTabProps {
  projectId: string;
  defaultSection?: string;
}

export function DraftsTab({ projectId, defaultSection }: DraftsTabProps) {
  const t = useTranslations("drafts");
  const tEmpty = useTranslations("emptyStates");
  const tCommon = useTranslations("common");
  const tSections = useTranslations("sections");
  const tCompliance = useTranslations("compliance");
  const tWriting = useTranslations("writing");
  const [activeDraftId, setActiveDraftId] = useState<string | null>(null);
  const [createOpen, setCreateOpen] = useState(false);
  const [newTitle, setNewTitle] = useState("");
  const [newPaperType, setNewPaperType] = useState<"literature_review" | "research_article">("literature_review");
  const [generateOpen, setGenerateOpen] = useState(false);
  const [generateContext, setGenerateContext] = useState("");
  const [sectionType, setSectionType] = useState(defaultSection || "literature_review");
  const [language, setLanguage] = useState("en");
  // The target body length sent as `target_words`, default 400, range 100-3000
  // (`@/lib/writing`).
  const [targetWords, setTargetWords] = useState(DEFAULT_TARGET_WORDS);
  const [complianceReport, setComplianceReport] = useState<ComplianceReport | null>(null);
  const [complianceOpen, setComplianceOpen] = useState(false);
  const [generateTaskId, setGenerateTaskId] = useState<string | null>(null);
  const [refineTaskId, setRefineTaskId] = useState<string | null>(null);
  const [refinedText, setRefinedText] = useState<string | null>(null);
  const [refineSectionKey, setRefineSectionKey] = useState<string | null>(null);
  const [nudgeDismissed, setNudgeDismissed] = useState(false);
  // Citation audit + model provenance of the last AI Write result (v1.1.0), keyed by draft.
  const [lastGenerateAudit, setLastGenerateAudit] = useState<{
    draftId: string | null;
    sectionType: string;
    audit: ReturnType<typeof citationAuditCounts>;
    provenance: ReturnType<typeof summarizeProvenance>;
  } | null>(null);
  const queryClient = useQueryClient();

  // Poll generate task status — bounded, backed off.
  const { data: generateTaskStatus } = useTaskPolling<{
    status: string;
    progress: number;
    progress_message: string | null;
    result: GenerateSectionResult | null;
  }>({ taskId: generateTaskId, kind: "generate_section" });

  const isGenerating = !!generateTaskId && generateTaskStatus?.status !== "completed" && generateTaskStatus?.status !== "failed";

  // Fetch project data for nudge banner
  const { data: project } = useQuery<{ refined_topic: string | null }>({
    queryKey: ["projects", projectId],
    queryFn: () => apiFetch(`/projects/${projectId}`),
    enabled: !!projectId,
  });

  // Poll refine task status — bounded, backed off.
  const { data: refineTaskStatus } = useTaskPolling<{
    status: string;
    progress: number;
    progress_message: string | null;
    result: { refined_text: string; section_key: string } | null;
  }>({ taskId: refineTaskId, kind: "refine" });

  const isRefining = !!refineTaskId && refineTaskStatus?.status !== "completed" && refineTaskStatus?.status !== "failed";

  // Handle generate completion. The write job itself verifies, finalizes and saves the
  // section directly to the draft (`app/services/writing.py`, `generate_section`) before
  // it reports "completed" -- `generateTaskStatus.result.content` already describes that
  // finalized, saved text. This effect does not rebuild a Tiptap document or PUT it; it
  // only refreshes the editor's own copy of the draft so it shows exactly what the
  // backend saved, with no marker of any kind.
  useEffect(() => {
    if (!generateTaskStatus || !generateTaskId) return;
    const action = describeGenerateCompletion(generateTaskStatus.status, generateTaskStatus.result);
    if (action?.kind === "generated") {
      // Surface the code-level citation audit and the model that actually wrote the text.
      const audit = citationAuditCounts(generateTaskStatus.result?.citation_audit);
      const provenance = summarizeProvenance(generateTaskStatus.result?.provenance);
      if (audit || provenance) {
        setLastGenerateAudit({
          draftId: activeDraftId,
          sectionType: action.sectionType,
          audit,
          provenance,
        });
      }
      if (activeDraftId) {
        queryClient.invalidateQueries({ queryKey: ["drafts", activeDraftId] });
      }
      toast.success(t("generateSuccess", { section: tSections(action.sectionType), count: action.papersUsed }));
      setGenerateTaskId(null);
    } else if (action?.kind === "emptyRemoved") {
      // finalize can legitimately empty the section (every cited sentence unverified,
      // every uncited sentence tagged "finding"). The job still completed and the
      // backend already saved the empty section, so the editor's own copy must be
      // refreshed the same way a normal generation refreshes it, and the user must be
      // told why the text is gone.
      if (activeDraftId) {
        queryClient.invalidateQueries({ queryKey: ["drafts", activeDraftId] });
      }
      toast.error(t("generateEmptyRemoved", { section: tSections(action.sectionType) }));
      setGenerateTaskId(null);
    } else if (generateTaskStatus.status === "failed") {
      toast.error(t("generateFailed"));
      setGenerateTaskId(null);
    }
  }, [generateTaskStatus?.status, generateTaskId]);

  // Handle refine completion — show accept/reject
  useEffect(() => {
    if (!refineTaskStatus || !refineTaskId) return;
    if (refineTaskStatus.status === "completed" && refineTaskStatus.result?.refined_text) {
      setRefinedText(refineTaskStatus.result.refined_text);
      setRefineSectionKey(refineTaskStatus.result.section_key || null);
      setRefineTaskId(null);
    } else if (refineTaskStatus.status === "failed") {
      toast.error(t("aiFailed"));
      setRefineTaskId(null);
    }
  }, [refineTaskStatus?.status, refineTaskId]);

  const { data: draftsData, isLoading } = useProjectDrafts(projectId);
  const { data: draftDetail } = useDraft(activeDraftId);
  const createDraft = useCreateDraft(projectId);
  const updateDraft = useUpdateDraft(projectId);
  const deleteDraft = useDeleteDraft(projectId);
  const generateSection = useGenerateSection();
  const exportDraft = useExportDraft();
  const checkCompliance = useCheckCompliance();
  const refine = useRefine();

  // Extract text for a given section from draft content
  function extractSectionText(sectionKey: string): string {
    const content = draftDetail?.draft?.content as Record<string, unknown> | null;
    const nodes = (content?.content as Record<string, unknown>[]) || [];
    let capturing = false;
    const paragraphs: string[] = [];

    for (const node of nodes) {
      if (node.type === "heading") {
        const headingText = ((node.content as Record<string, unknown>[]) || [])
          .map((c) => (c.text as string) || "")
          .join("")
          .toLowerCase()
          .replace(/\s+/g, "_");
        if (headingText === sectionKey || headingText === sectionKey.replace(/_/g, " ") || headingText.includes(sectionKey.replace(/_/g, " "))) {
          capturing = true;
          continue;
        } else if (capturing) {
          break; // Hit the next section heading
        }
      } else if (capturing && node.type === "paragraph") {
        const text = ((node.content as Record<string, unknown>[]) || [])
          .map((c) => (c.text as string) || "")
          .join("");
        if (text) paragraphs.push(text);
      }
    }
    return paragraphs.join("\n\n");
  }

  async function handleRefineSection() {
    if (!activeDraftId) return;
    const text = extractSectionText(sectionType);
    if (!text.trim()) {
      toast.error(t("refineNoContent"));
      return;
    }
    try {
      const result = await refine.mutateAsync({
        draftId: activeDraftId,
        text,
        scope: "section",
        sectionKey: sectionType,
      });
      setRefineTaskId(result.task_id);
      toast.info(t("refining"));
    } catch {
      toast.error(t("aiFailed"));
    }
  }

  function handleAcceptRefine() {
    if (!activeDraftId || !refinedText || !refineSectionKey) return;

    const existingContent = draftDetail?.draft?.content as Record<string, unknown> | null;
    const existingNodes = ((existingContent?.content as Record<string, unknown>[]) || []);

    // Replace the section's paragraph nodes with refined text
    const newNodes: Record<string, unknown>[] = [];
    let inTargetSection = false;
    let replacedContent = false;

    for (const node of existingNodes) {
      if (node.type === "heading") {
        const headingText = ((node.content as Record<string, unknown>[]) || [])
          .map((c) => (c.text as string) || "")
          .join("")
          .toLowerCase()
          .replace(/\s+/g, "_");
        const isTarget = headingText === refineSectionKey ||
          headingText === refineSectionKey.replace(/_/g, " ") ||
          headingText.includes(refineSectionKey.replace(/_/g, " "));

        if (isTarget) {
          inTargetSection = true;
          newNodes.push(node); // Keep the heading
          // Insert refined paragraphs
          refinedText.split("\n\n").filter(Boolean).forEach((para) => {
            newNodes.push({
              type: "paragraph",
              content: [{ type: "text", text: para }],
            });
          });
          replacedContent = true;
          continue;
        } else if (inTargetSection) {
          inTargetSection = false;
        }
      }

      if (!inTargetSection) {
        newNodes.push(node);
      }
      // Skip old paragraphs in target section (they're replaced)
    }

    if (replacedContent) {
      const mergedContent = { type: "doc", content: newNodes };
      apiFetch(`/drafts/${activeDraftId}`, {
        method: "PUT",
        body: JSON.stringify({ content: mergedContent }),
      }).then(() => {
        queryClient.invalidateQueries({ queryKey: ["drafts", activeDraftId] });
        toast.success(t("refineAccepted"));
      }).catch(() => {
        toast.error(t("refineApplyFailed"));
      });
    }

    setRefinedText(null);
    setRefineSectionKey(null);
  }

  function handleRejectRefine() {
    setRefinedText(null);
    setRefineSectionKey(null);
  }

  async function handleCreate() {
    if (!newTitle.trim()) return;
    try {
      const result = await createDraft.mutateAsync({ title: newTitle, paper_type: newPaperType });
      setNewTitle("");
      setCreateOpen(false);
      setActiveDraftId((result as { id: string }).id);
    } catch {
      toast.error(t("createFailed"));
    }
  }

  async function handleDelete(draftId: string) {
    try {
      await deleteDraft.mutateAsync(draftId);
      if (activeDraftId === draftId) setActiveDraftId(null);
      toast.success(t("deleted"));
    } catch {
      toast.error(t("deleteFailed"));
    }
  }

  function handleSave(content: Record<string, unknown>) {
    if (!activeDraftId) return;
    updateDraft.mutate({ draftId: activeDraftId, content });
  }

  async function handleGenerate() {
    if (!activeDraftId) return;
    try {
      const result = await generateSection.mutateAsync({
        draftId: activeDraftId,
        section_type: sectionType,
        context: generateContext || undefined,
        language,
        target_words: clampTargetWords(targetWords),
      });
      setGenerateTaskId(result.task_id);
      setGenerateOpen(false);
      setGenerateContext("");
      toast.info(t("aiStarted"));
    } catch {
      toast.error(t("aiFailed"));
    }
  }

  async function handleCheckCompliance() {
    if (!activeDraftId) return;
    try {
      const report = await checkCompliance.mutateAsync({ draftId: activeDraftId });
      setComplianceReport(report);
      setComplianceOpen(true);
    } catch {
      toast.error(tCompliance("checkCompliance"));
    }
  }

  function handleExport(format: string = "docx") {
    if (!activeDraftId) return;
    exportDraft.mutate({ draftId: activeDraftId, format });
  }

  // Editor view
  if (activeDraftId && draftDetail) {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setActiveDraftId(null)}
              className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
            >
              <ArrowLeft className="mr-1 size-4" />
              {tCommon("back")}
            </Button>
            <h2
              className="text-lg font-semibold text-[var(--ds-text-heading)]"
              style={{ fontFamily: "var(--font-heading)" }}
            >
              {draftDetail.draft.title}
            </h2>
            <span className="text-xs text-[var(--ds-text-muted)]">
              v{draftDetail.draft.current_version}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setGenerateOpen(true)}
              className="text-[var(--ds-primary)] hover:bg-[var(--ds-primary-light)] hover:text-[var(--ds-primary)]"
            >
              <Sparkles className="mr-1 size-4" />
              {t("aiWrite")}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleRefineSection}
              disabled={refine.isPending || isRefining}
              className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
            >
              {isRefining ? (
                <Loader2 className="mr-1 size-3.5 animate-spin" />
              ) : (
                <Sparkles className="mr-1 size-3.5" />
              )}
              {t("refine")}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleCheckCompliance}
              disabled={checkCompliance.isPending}
              className="text-[var(--ds-accent)] hover:bg-[var(--ds-accent)]/15 hover:text-[var(--ds-accent)]"
            >
              {checkCompliance.isPending ? (
                <Loader2 className="mr-1 size-4 animate-spin" />
              ) : (
                <ClipboardCheck className="mr-1 size-4" />
              )}
              {checkCompliance.isPending
                ? tCompliance("checking")
                : tCompliance("checkCompliance")}
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger
                render={<Button variant="ghost" size="sm" />}
                disabled={exportDraft.isPending}
                className="text-[var(--ds-success)] hover:bg-[var(--ds-success)]/15 hover:text-[var(--ds-success)]"
              >
                {exportDraft.isPending ? (
                  <Loader2 className="mr-1 size-4 animate-spin" />
                ) : (
                  <Download className="mr-1 size-4" />
                )}
                {t("export")}
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                <DropdownMenuItem onClick={() => handleExport("docx")}>
                  {t("exportDocx")}
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => handleExport("pdf")}>
                  {t("exportPdf")}
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => handleExport("latex")}>
                  {t("exportLatex")}
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>

        {/* Citation audit of the last AI Write (checked by code against the library) */}
        {lastGenerateAudit && lastGenerateAudit.draftId === activeDraftId && (
          <div
            className="mb-4 flex items-start gap-2 rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-page)] px-3 py-2"
            data-testid="citation-audit-line"
          >
            <ClipboardCheck className="mt-0.5 size-4 shrink-0 text-[var(--ds-primary)]" />
            <div className="min-w-0 flex-1 space-y-0.5">
              <p
                className="text-xs font-medium text-[var(--ds-text-heading)]"
                style={{ fontFamily: "var(--font-heading)" }}
              >
                {t("citationAuditTitle")} — {tSections(lastGenerateAudit.sectionType)}
              </p>
              {lastGenerateAudit.audit && (
                <p className="text-xs text-[var(--ds-text-secondary)]" style={{ fontFamily: "var(--font-body)" }}>
                  {t("citationAuditLine", {
                    matched: lastGenerateAudit.audit.matched,
                    unmatched: lastGenerateAudit.audit.unmatched,
                    flags: lastGenerateAudit.audit.flags,
                  })}
                </p>
              )}
              {lastGenerateAudit.provenance && (
                <p className="text-xs text-[var(--ds-text-muted)]" style={{ fontFamily: "var(--font-body)" }}>
                  {t("modelLine", {
                    models: lastGenerateAudit.provenance.models,
                    temperature: lastGenerateAudit.provenance.temperature,
                    version: lastGenerateAudit.provenance.promptVersion,
                  })}
                </p>
              )}
            </div>
            <button
              type="button"
              onClick={() => setLastGenerateAudit(null)}
              aria-label={t("nudgeDismiss")}
              className="rounded p-0.5 text-[var(--ds-text-muted)] hover:text-[var(--ds-text-heading)]"
            >
              <X className="size-3.5" />
            </button>
          </div>
        )}

        {/* Generation progress */}
        {isGenerating && generateTaskStatus && (
          <div className="mb-4 space-y-2 rounded-lg border border-[var(--ds-primary)]/30 bg-[var(--ds-primary-light)] p-3">
            <div className="flex items-center gap-2">
              <Loader2 className="size-4 animate-spin text-[var(--ds-primary)]" />
              <span className="text-sm text-[var(--ds-text-secondary)]">
                {generateTaskStatus.progress_message || t("generating")}
              </span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-[var(--ds-border)]">
              <div
                className="h-full rounded-full bg-[var(--ds-primary)] transition-all duration-300"
                style={{ width: `${Math.max(generateTaskStatus.progress * 100, 5)}%` }}
              />
            </div>
          </div>
        )}

        {/* Refine progress */}
        {isRefining && refineTaskStatus && (
          <div className="mb-4 space-y-2 rounded-lg border border-[var(--ds-accent)]/30 bg-[var(--ds-accent)]/5 p-3">
            <div className="flex items-center gap-2">
              <Loader2 className="size-4 animate-spin text-[var(--ds-accent)]" />
              <span className="text-sm text-[var(--ds-text-secondary)]">
                {refineTaskStatus.progress_message || t("refining")}
              </span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-[var(--ds-border)]">
              <div
                className="h-full rounded-full bg-[var(--ds-accent)] transition-all duration-300"
                style={{ width: `${Math.max(refineTaskStatus.progress * 100, 5)}%` }}
              />
            </div>
          </div>
        )}

        {/* Refine accept/reject banner */}
        {refinedText && (
          <div className="mb-4 space-y-3 rounded-lg border border-[var(--ds-accent)]/30 bg-[var(--ds-accent)]/5 p-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-[var(--ds-text-heading)]" style={{ fontFamily: "var(--font-heading)" }}>
                {t("refine")} — {refineSectionKey?.replace(/_/g, " ").replace(/\b\w/g, (c: string) => c.toUpperCase())}
              </span>
              <div className="flex gap-2">
                <Button
                  size="sm"
                  onClick={handleAcceptRefine}
                  className="bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
                >
                  {t("refineAccept")}
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={handleRejectRefine}
                  className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
                >
                  {t("refineReject")}
                </Button>
              </div>
            </div>
            <div className="max-h-48 overflow-y-auto rounded border border-[var(--ds-border)] bg-[var(--ds-bg-page)] p-3 text-sm text-[var(--ds-text-secondary)]" style={{ fontFamily: "var(--font-body)" }}>
              {refinedText.split("\n\n").map((para, i) => (
                <p key={i} className={i > 0 ? "mt-2" : ""}>{para}</p>
              ))}
            </div>
          </div>
        )}

        <PaperEditor
          content={draftDetail.draft.content as Record<string, unknown> | null}
          onSave={handleSave}
          saving={updateDraft.isPending}
          draftId={activeDraftId ?? undefined}
        />

        {/* Claim verification report (level 2): each cited claim checked against the cited full text */}
        {activeDraftId && (
          <div className="mt-4" data-testid="claim-verification-report">
            <ClaimVerificationReport draftId={activeDraftId} projectId={projectId} />
          </div>
        )}

        {/* AI Generate Dialog */}
        <Dialog open={generateOpen} onOpenChange={setGenerateOpen}>
          <DialogContent className="border-[var(--ds-border)] bg-[var(--ds-bg-card)] sm:max-w-md">
            <DialogHeader>
              <DialogTitle
                className="text-[var(--ds-text-heading)]"
                style={{ fontFamily: "var(--font-heading)" }}
              >
                {t("aiDialogTitle")}
              </DialogTitle>
              <DialogDescription
                className="text-[var(--ds-text-secondary)]"
                style={{ fontFamily: "var(--font-body)" }}
              >
                {t("aiDialogDescription")}
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-3 py-2">
              <div className="space-y-2">
                <Label className="text-[var(--ds-text-heading)]">{t("sectionType")}</Label>
                <select
                  value={sectionType}
                  onChange={(e) => setSectionType(e.target.value)}
                  className="w-full rounded-md border border-[var(--ds-border)] bg-[var(--ds-bg-page)] px-3 py-2 text-sm text-[var(--ds-text-heading)]"
                  style={{ fontFamily: "var(--font-body)" }}
                >
                  {(draftDetail?.draft?.paper_type === "research_article"
                    ? ["introduction", "literature_review", "methods", "results", "discussion", "implications", "conclusion", "abstract"] as const
                    : ["introduction", "literature_review", "discussion", "conclusion", "abstract"] as const
                  ).map(
                    (section) => (
                      <option key={section} value={section}>
                        {tSections(section)}
                      </option>
                    )
                  )}
                </select>
              </div>
              <div className="space-y-2">
                <Label className="text-[var(--ds-text-heading)]">{t("instructionsLabel")}</Label>
                <Input
                  value={generateContext}
                  onChange={(e) => setGenerateContext(e.target.value)}
                  placeholder={t("instructionsPlaceholder")}
                  className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)]"
                  style={{ fontFamily: "var(--font-body)" }}
                />
              </div>
              <div className="space-y-2">
                <Label className="text-[var(--ds-text-heading)]">{tWriting("language")}</Label>
                <select
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                  className="w-full rounded-md border border-[var(--ds-border)] bg-[var(--ds-bg-page)] px-3 py-2 text-sm text-[var(--ds-text-heading)]"
                  style={{ fontFamily: "var(--font-body)" }}
                >
                  {LANGUAGES.map((lang) => (
                    <option key={lang.code} value={lang.code}>
                      {lang.label}
                    </option>
                  ))}
                </select>
              </div>
              <div className="space-y-2">
                <Label className="text-[var(--ds-text-heading)]">{t("targetWordsLabel")}</Label>
                <Input
                  type="number"
                  min={MIN_TARGET_WORDS}
                  max={MAX_TARGET_WORDS}
                  value={targetWords}
                  onChange={(e) => setTargetWords(Number(e.target.value))}
                  onBlur={(e) => setTargetWords(clampTargetWords(Number(e.target.value)))}
                  className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)]"
                  style={{ fontFamily: "var(--font-body)" }}
                />
                <p className="text-xs text-[var(--ds-text-muted)]" style={{ fontFamily: "var(--font-body)" }}>
                  {t("targetWordsHelp", { min: MIN_TARGET_WORDS, max: MAX_TARGET_WORDS })}
                </p>
              </div>
            </div>
            <DialogFooter>
              <DialogClose
                render={<Button variant="ghost" className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]" />}
              >
                {tCommon("cancel")}
              </DialogClose>
              <Button
                onClick={handleGenerate}
                disabled={generateSection.isPending}
                className="bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
              >
                {generateSection.isPending && (
                  <Loader2 className="mr-1 size-4 animate-spin" />
                )}
                {t("generate")}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Compliance Report Dialog */}
        <Dialog open={complianceOpen} onOpenChange={setComplianceOpen}>
          <DialogContent className="border-[var(--ds-border)] bg-[var(--ds-bg-card)] sm:max-w-md">
            <DialogHeader>
              <DialogTitle
                className="text-[var(--ds-text-heading)]"
                style={{ fontFamily: "var(--font-heading)" }}
              >
                {tCompliance("title")}
              </DialogTitle>
              <DialogDescription className="sr-only">
                {tCompliance("checkCompliance")}
              </DialogDescription>
            </DialogHeader>
            {complianceReport && (
              <div className="py-2">
                <ComplianceReportView report={complianceReport} />
              </div>
            )}
            <DialogFooter>
              <DialogClose
                render={<Button variant="ghost" className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]" />}
              >
                {tCommon("cancel")}
              </DialogClose>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    );
  }

  // Draft list view
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2
          className="text-lg font-semibold text-[var(--ds-text-heading)]"
          style={{ fontFamily: "var(--font-heading)" }}
        >
          {t("title")}
        </h2>
        <Button
          size="sm"
          onClick={() => setCreateOpen(true)}
          className="bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
        >
          <Plus className="mr-1 size-4" />
          {t("newDraft")}
        </Button>
      </div>

      {/* Soft nudge banner */}
      {!nudgeDismissed && project && !project.refined_topic && (
        <div className="mb-4 flex items-center justify-between rounded-lg border border-[var(--ds-primary)]/30 bg-[var(--ds-primary-light)] px-4 py-3">
          <p className="text-sm text-[var(--ds-text-secondary)]" style={{ fontFamily: "var(--font-body)" }}>
            {t("nudgeRefineTopic")}
          </p>
          <div className="flex gap-2">
            <Button size="sm" variant="ghost" className="text-[var(--ds-primary)] hover:text-[var(--ds-primary-hover)]" onClick={() => { setNudgeDismissed(true); toast.info(t("nudgeRefineRedirect")); }}>
              {t("nudgeRefineAction")}
            </Button>
            <Button size="sm" variant="ghost" className="text-[var(--ds-text-secondary)]" onClick={() => setNudgeDismissed(true)}>
              {t("nudgeDismiss")}
            </Button>
          </div>
        </div>
      )}

      {isLoading ? (
        <div className="space-y-3">
          {[1, 2].map((i) => (
            <div
              key={i}
              className="h-20 animate-pulse rounded-lg bg-[var(--ds-bg-card)]"
            />
          ))}
        </div>
      ) : !draftsData?.drafts.length ? (
        <EmptyState
          icon="✍️"
          title={t("title")}
          description={tEmpty("drafts.description")}
          actionLabel={t("newDraft")}
          onAction={() => setCreateOpen(true)}
        />
      ) : (
        <div className="space-y-2">
          {draftsData.drafts.map((draft) => (
            <Card
              key={draft.id}
              className="cursor-pointer border-[var(--ds-border)] bg-[var(--ds-bg-card)] transition-colors hover:border-[var(--ds-primary)]/30"
              onClick={() => setActiveDraftId(draft.id)}
            >
              <CardContent className="flex items-center justify-between py-4">
                <div className="flex items-center gap-3">
                  <div className="flex size-9 items-center justify-center rounded-md bg-[var(--ds-primary-light)]">
                    <FileText className="size-4 text-[var(--ds-primary)]" />
                  </div>
                  <div>
                    <h3
                      className="text-sm font-medium text-[var(--ds-text-heading)]"
                      style={{ fontFamily: "var(--font-heading)" }}
                    >
                      {draft.title}
                    </h3>
                    <p
                      className="text-xs text-[var(--ds-text-muted)]"
                      style={{ fontFamily: "var(--font-body)" }}
                    >
                      v{draft.current_version} · {t(`status.${draft.status}`)} ·{" "}
                      {new Date(draft.updated_at).toLocaleDateString()}
                    </p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDelete(draft.id);
                  }}
                  className="text-[var(--ds-text-muted)] hover:text-[var(--ds-error)]"
                >
                  <Trash2 className="size-4" />
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Create Draft Dialog */}
      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent className="border-[var(--ds-border)] bg-[var(--ds-bg-card)] sm:max-w-sm">
          <DialogHeader>
            <DialogTitle
              className="text-[var(--ds-text-heading)]"
              style={{ fontFamily: "var(--font-heading)" }}
            >
              {t("createTitle")}
            </DialogTitle>
            <DialogDescription
              className="text-[var(--ds-text-secondary)]"
              style={{ fontFamily: "var(--font-body)" }}
            >
              {t("createDescription")}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3 py-2">
            <div className="space-y-2">
              <Label className="text-[var(--ds-text-heading)]">{tCommon("title")}</Label>
              <Input
                value={newTitle}
                onChange={(e) => setNewTitle(e.target.value)}
                placeholder={t("titlePlaceholder")}
                className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)]"
                style={{ fontFamily: "var(--font-body)" }}
                onKeyDown={(e) => e.key === "Enter" && handleCreate()}
              />
            </div>
            <div className="space-y-2">
              <Label className="text-[var(--ds-text-heading)]">{t("paperType")}</Label>
              <select
                value={newPaperType}
                onChange={(e) => setNewPaperType(e.target.value as "literature_review" | "research_article")}
                className="h-8 w-full rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-page)] px-2.5 py-1 text-sm text-[var(--ds-text-heading)] outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
                style={{ fontFamily: "var(--font-body)" }}
              >
                <option value="literature_review">{t("paperTypeLitReview")}</option>
                <option value="research_article">{t("paperTypeResearch")}</option>
              </select>
            </div>
          </div>
          <DialogFooter>
            <DialogClose
              render={<Button variant="ghost" className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]" />}
            >
              {tCommon("cancel")}
            </DialogClose>
            <Button
              onClick={handleCreate}
              disabled={!newTitle.trim() || createDraft.isPending}
              className="bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
            >
              {createDraft.isPending && (
                <Loader2 className="mr-1 size-4 animate-spin" />
              )}
              {tCommon("create")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
