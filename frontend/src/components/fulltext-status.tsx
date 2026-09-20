"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  Check,
  AlertTriangle,
  Upload,
  ClipboardPaste,
  FileText,
  Loader2,
  Download,
} from "lucide-react";
import { toast } from "sonner";
import { useTranslations } from "next-intl";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
  useTriggerAcquireFullTexts,
  useUploadFulltext,
  usePasteFulltext,
} from "@/hooks/use-fulltext";
import { useTaskResult } from "@/hooks/use-verification";

// ── Types ──────────────────────────────────────────────────────────────────

interface PaperWithFullText {
  id: string;
  title: string;
  metadata_?: { full_text_status?: string } | null;
}

interface FulltextStatusProps {
  projectId: string;
  papers: PaperWithFullText[];
}

// ── Helpers ────────────────────────────────────────────────────────────────

function statusBadge(
  status: string | undefined,
  t: (key: string) => string,
): { icon: typeof Check; color: string; bg: string; label: string } {
  switch (status) {
    case "available":
      return {
        icon: Check,
        color: "text-[var(--ds-success)]",
        bg: "bg-[rgba(5,150,105,0.08)]",
        label: t("fulltextAvailable"),
      };
    case "abstract_only":
    default:
      return {
        icon: AlertTriangle,
        color: "text-amber-500",
        bg: "bg-amber-500/15",
        label: t("fulltextAbstractOnly"),
      };
  }
}

// ── Component ──────────────────────────────────────────────────────────────

export function FulltextStatus({ projectId, papers }: FulltextStatusProps) {
  const t = useTranslations("search");

  // Mutations
  const acquireMutation = useTriggerAcquireFullTexts();
  const uploadMutation = useUploadFulltext();
  const pasteMutation = usePasteFulltext();

  // Task polling for acquire progress
  const [acquireTaskId, setAcquireTaskId] = useState<string | null>(null);
  const { data: taskData } = useTaskResult(acquireTaskId);

  const isAcquiring =
    !!acquireTaskId &&
    taskData?.status !== "completed" &&
    taskData?.status !== "failed";

  // Paste dialog state
  const [pasteOpen, setPasteOpen] = useState(false);
  const [pastePaperId, setPastePaperId] = useState<string | null>(null);
  const [pasteText, setPasteText] = useState("");

  // File input ref (one hidden input, reused per paper)
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadPaperId, setUploadPaperId] = useState<string | null>(null);

  // Handle acquire completion
  useEffect(() => {
    if (!taskData || !acquireTaskId) return;

    if (taskData.status === "completed") {
      setAcquireTaskId(null);
      toast.success(t("fulltextAcquireComplete"));
    } else if (taskData.status === "failed") {
      setAcquireTaskId(null);
      toast.error(taskData.error || t("fulltextAcquireFailed"));
    }
  }, [taskData, acquireTaskId, t]);

  // Batch acquire
  const handleAcquireAll = useCallback(() => {
    acquireMutation.mutate(
      { projectId },
      {
        onSuccess: (data) => {
          setAcquireTaskId(data.task_id);
          toast.info(t("fulltextAcquireStarted"));
        },
        onError: (err) => {
          toast.error(err.message || t("fulltextAcquireFailed"));
        },
      },
    );
  }, [acquireMutation, projectId, t]);

  // Upload handler
  function handleUploadClick(paperId: string) {
    setUploadPaperId(paperId);
    fileInputRef.current?.click();
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file || !uploadPaperId) return;

    uploadMutation.mutate(
      { paperId: uploadPaperId, projectId, file },
      {
        onSuccess: () => {
          toast.success(t("fulltextUploadSuccess"));
        },
        onError: (err) => {
          toast.error(err.message || t("fulltextUploadFailed"));
        },
      },
    );

    // Reset so the same file can be selected again
    e.target.value = "";
    setUploadPaperId(null);
  }

  // Paste handler
  function openPasteDialog(paperId: string) {
    setPastePaperId(paperId);
    setPasteText("");
    setPasteOpen(true);
  }

  function handlePasteSubmit() {
    if (!pastePaperId || !pasteText.trim()) return;

    pasteMutation.mutate(
      { paperId: pastePaperId, projectId, text: pasteText.trim() },
      {
        onSuccess: () => {
          toast.success(t("fulltextPasteSuccess"));
          setPasteOpen(false);
          setPasteText("");
          setPastePaperId(null);
        },
        onError: (err) => {
          toast.error(err.message || t("fulltextPasteFailed"));
        },
      },
    );
  }

  // Counts
  const availableCount = papers.filter(
    (p) => p.metadata_?.full_text_status === "available",
  ).length;
  const abstractOnlyCount = papers.length - availableCount;

  return (
    <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle
            className="flex items-center gap-2 text-base text-[var(--ds-text-heading)]"
            style={{ fontFamily: "var(--font-heading)" }}
          >
            <FileText className="size-4 text-[var(--ds-primary)]" />
            {t("fulltextTitle")}
            <span className="rounded-full bg-[var(--ds-border)] px-2 py-0.5 text-xs font-normal text-[var(--ds-text-secondary)]">
              {availableCount}/{papers.length}
            </span>
          </CardTitle>
          <Button
            variant="ghost"
            size="sm"
            className="text-[var(--ds-primary)] hover:bg-[var(--ds-primary-light)] hover:text-[var(--ds-primary)]"
            onClick={handleAcquireAll}
            disabled={isAcquiring || acquireMutation.isPending}
          >
            {isAcquiring ? (
              <Loader2 className="mr-1 size-3.5 animate-spin" />
            ) : (
              <Download className="mr-1 size-3.5" />
            )}
            {isAcquiring
              ? t("fulltextAcquiring")
              : t("fulltextAcquireAll")}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {/* Acquire progress */}
        {isAcquiring && taskData && (
          <div className="mb-4 space-y-2 rounded-lg border border-[var(--ds-primary)]/30 bg-[var(--ds-primary-light)] p-3">
            <div className="flex items-center gap-2">
              <Loader2 className="size-4 animate-spin text-[var(--ds-primary)]" />
              <span
                className="text-sm text-[var(--ds-text-secondary)]"
                style={{ fontFamily: "var(--font-body)" }}
              >
                {taskData.progress_message || t("fulltextAcquiringProgress")}
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

        {/* Summary counts */}
        {papers.length > 0 && (
          <div className="mb-3 flex items-center gap-3 text-xs">
            <span className="flex items-center gap-1 text-[var(--ds-success)]">
              <Check className="size-3" />
              {t("fulltextCountAvailable", { count: availableCount })}
            </span>
            <span className="flex items-center gap-1 text-amber-500">
              <AlertTriangle className="size-3" />
              {t("fulltextCountAbstract", { count: abstractOnlyCount })}
            </span>
          </div>
        )}

        {/* Per-paper list */}
        {papers.length > 0 ? (
          <div className="space-y-2">
            {papers.map((paper) => {
              const ftStatus = paper.metadata_?.full_text_status;
              const badge = statusBadge(ftStatus, t);
              const Icon = badge.icon;

              return (
                <div
                  key={paper.id}
                  className="flex items-center justify-between rounded-md bg-[var(--ds-bg-page)] px-3 py-2"
                >
                  <div className="flex items-center gap-2 overflow-hidden">
                    <span
                      className={`inline-flex shrink-0 items-center gap-1 rounded-md px-1.5 py-0.5 text-xs font-medium ${badge.bg} ${badge.color}`}
                    >
                      <Icon className="size-3" />
                      {badge.label}
                    </span>
                    <span
                      className="truncate text-sm text-[var(--ds-text-body)]"
                      style={{ fontFamily: "var(--font-body)" }}
                    >
                      {paper.title}
                    </span>
                  </div>
                  <div className="ml-2 flex shrink-0 items-center gap-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 px-2 text-xs text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
                      onClick={() => handleUploadClick(paper.id)}
                      disabled={uploadMutation.isPending}
                    >
                      {uploadMutation.isPending &&
                      uploadPaperId === paper.id ? (
                        <Loader2 className="mr-1 size-3 animate-spin" />
                      ) : (
                        <Upload className="mr-1 size-3" />
                      )}
                      {t("fulltextUpload")}
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 px-2 text-xs text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
                      onClick={() => openPasteDialog(paper.id)}
                      disabled={pasteMutation.isPending}
                    >
                      <ClipboardPaste className="mr-1 size-3" />
                      {t("fulltextPaste")}
                    </Button>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="py-6 text-center">
            <p
              className="text-sm text-[var(--ds-text-secondary)]"
              style={{ fontFamily: "var(--font-body)" }}
            >
              {t("fulltextNoPapers")}
            </p>
          </div>
        )}

        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.txt,.doc,.docx"
          className="hidden"
          onChange={handleFileChange}
        />
      </CardContent>

      {/* Paste Text Dialog */}
      <Dialog open={pasteOpen} onOpenChange={setPasteOpen}>
        <DialogContent className="border-[var(--ds-border)] bg-[var(--ds-bg-card)] sm:max-w-lg">
          <DialogHeader>
            <DialogTitle
              className="text-[var(--ds-text-heading)]"
              style={{ fontFamily: "var(--font-heading)" }}
            >
              {t("fulltextPasteDialogTitle")}
            </DialogTitle>
            <DialogDescription
              className="text-[var(--ds-text-secondary)]"
              style={{ fontFamily: "var(--font-body)" }}
            >
              {t("fulltextPasteDialogDescription")}
            </DialogDescription>
          </DialogHeader>
          <div className="py-2">
            <textarea
              value={pasteText}
              onChange={(e) => setPasteText(e.target.value)}
              placeholder={t("fulltextPastePlaceholder")}
              className="h-48 w-full resize-y rounded-md border border-[var(--ds-border)] bg-[var(--ds-bg-page)] px-3 py-2 text-sm text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)] focus:border-[var(--ds-primary)] focus:outline-none focus:ring-1 focus:ring-[var(--ds-primary)]"
              style={{ fontFamily: "var(--font-body)" }}
            />
          </div>
          <DialogFooter>
            <DialogClose
              render={
                <Button
                  variant="ghost"
                  className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
                />
              }
            >
              {t("fulltextCancel")}
            </DialogClose>
            <Button
              onClick={handlePasteSubmit}
              disabled={!pasteText.trim() || pasteMutation.isPending}
              className="bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
            >
              {pasteMutation.isPending && (
                <Loader2 className="mr-1 size-4 animate-spin" />
              )}
              {t("fulltextPasteSubmit")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}
