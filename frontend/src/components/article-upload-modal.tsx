"use client";

import { useState, useCallback, useEffect } from "react";
import { Upload, X, Loader2, FileText, Check } from "lucide-react";
import { toast } from "sonner";
import { useTranslations } from "next-intl";
import { useQueryClient } from "@tanstack/react-query";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  useUploadArticles,
  useConfirmUpload,
  type ReviewPaper,
} from "@/hooks/use-upload-articles";
import { useTaskPolling } from "@/hooks/use-task-polling";

// ── Constants ────────────────────────────────────────────────────────────────

const ACCEPTED_EXTENSIONS = [".pdf", ".docx", ".doc", ".bib"];
const ACCEPT_STRING = ACCEPTED_EXTENSIONS.join(",");
const MAX_FILES = 30;
const MAX_SIZE_MB = 50;
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024;

// ── Props ────────────────────────────────────────────────────────────────────

interface ArticleUploadModalProps {
  projectId: string;
  open: boolean;
  onClose: () => void;
}

// ── Component ────────────────────────────────────────────────────────────────

export function ArticleUploadModal({
  projectId,
  open,
  onClose,
}: ArticleUploadModalProps) {
  const t = useTranslations("papers");
  const queryClient = useQueryClient();

  // Upload mode state
  const [files, setFiles] = useState<File[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Task polling state
  const [taskId, setTaskId] = useState<string | null>(null);
  const [completedTaskId, setCompletedTaskId] = useState<string | null>(null);

  // Review mode state
  const [reviewPapers, setReviewPapers] = useState<ReviewPaper[]>([]);
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(new Set());
  const [mode, setMode] = useState<"upload" | "review">("upload");

  // Mutations
  const uploadArticles = useUploadArticles();
  const confirmUpload = useConfirmUpload(projectId);

  // Task polling — bounded, backed off.
  const { data: taskData } = useTaskPolling<{
    status: string;
    progress: number;
    progress_message: string | null;
    result: { papers: ReviewPaper[] } | null;
    error: string | null;
  }>({ taskId, kind: "article_extract" });

  const isExtracting =
    !!taskId &&
    taskData?.status !== "completed" &&
    taskData?.status !== "failed";

  // Handle extraction completion
  useEffect(() => {
    if (!taskData || !taskId) return;

    if (taskData.status === "completed" && taskData.result?.papers) {
      const papers = taskData.result.papers;
      setReviewPapers(papers);
      setSelectedIndices(new Set(papers.map((_, i) => i)));
      setMode("review");
      setCompletedTaskId(taskId);
      setTaskId(null);
      toast.success(t("extractComplete"));
    } else if (taskData.status === "failed") {
      setTaskId(null);
      toast.error(taskData.error || t("extractFailed"));
    }
  }, [taskData, taskId, t]);

  // Reset state when modal closes
  useEffect(() => {
    if (!open) {
      setFiles([]);
      setDragActive(false);
      setErrorMsg(null);
      setTaskId(null);
      setCompletedTaskId(null);
      setReviewPapers([]);
      setSelectedIndices(new Set());
      setMode("upload");
    }
  }, [open]);

  // ── File validation ────────────────────────────────────────────────────────

  function validateFile(file: File): string | null {
    const ext = "." + file.name.split(".").pop()?.toLowerCase();
    if (!ACCEPTED_EXTENSIONS.includes(ext)) {
      return t("invalidFileType", { name: file.name });
    }
    if (file.size > MAX_SIZE_BYTES) {
      return t("fileTooLarge", { name: file.name, max: MAX_SIZE_MB });
    }
    return null;
  }

  function addFiles(incoming: FileList | File[]) {
    const newFiles = Array.from(incoming);
    const errors: string[] = [];

    for (const f of newFiles) {
      const err = validateFile(f);
      if (err) errors.push(err);
    }

    if (errors.length > 0) {
      setErrorMsg(errors.join("; "));
      return;
    }

    setFiles((prev) => {
      const combined = [...prev, ...newFiles];
      if (combined.length > MAX_FILES) {
        setErrorMsg(t("maxFilesAllowed", { max: MAX_FILES }));
        return prev;
      }
      setErrorMsg(null);
      return combined;
    });
  }

  // ── Drag-and-drop handlers ────────────────────────────────────────────────

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        addFiles(e.dataTransfer.files);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  function handleFileInput(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files && e.target.files.length > 0) {
      addFiles(e.target.files);
    }
  }

  function removeFile(index: number) {
    setFiles((prev) => prev.filter((_, i) => i !== index));
    setErrorMsg(null);
  }

  // ── Upload & extract ──────────────────────────────────────────────────────

  function handleExtract() {
    if (files.length === 0) return;

    uploadArticles.mutate(
      { projectId, files },
      {
        onSuccess: (data) => {
          setTaskId(data.task_id);
          toast.info(t("extracting"));
        },
        onError: (err) => {
          toast.error(err.message || t("extractFailed"));
        },
      },
    );
  }

  // ── Review helpers ────────────────────────────────────────────────────────

  function toggleSelect(index: number) {
    setSelectedIndices((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  }

  function toggleSelectAll() {
    if (selectedIndices.size === reviewPapers.length) {
      setSelectedIndices(new Set());
    } else {
      setSelectedIndices(new Set(reviewPapers.map((_, i) => i)));
    }
  }

  function updatePaper(index: number, field: string, value: string) {
    setReviewPapers((prev) =>
      prev.map((p, i) => {
        if (i !== index) return p;
        if (field === "authors") {
          return {
            ...p,
            authors: value
              .split(",")
              .map((n) => ({ name: n.trim() }))
              .filter((a) => a.name),
          };
        }
        if (field === "year") {
          return { ...p, year: value ? parseInt(value) || null : null };
        }
        return { ...p, [field]: value || null };
      }),
    );
  }

  function handleConfirm() {
    const selected = reviewPapers
      .filter((_, i) => selectedIndices.has(i))
      .map((p) => ({
        index: p.index,
        title: p.title,
        authors: p.authors,
        year: p.year,
        journal_name: p.journal_name,
        doi: p.doi,
        abstract: p.abstract,
      }));

    if (selected.length === 0) return;

    confirmUpload.mutate(
      { taskId: completedTaskId || "", papers: selected },
      {
        onSuccess: (data) => {
          toast.success(t("papersImported", { count: data.imported_count }));
          queryClient.invalidateQueries({
            queryKey: ["projects", projectId, "papers"],
          });
          onClose();
        },
        onError: (err) => {
          toast.error(err.message || t("importFailed"));
        },
      },
    );
  }

  // ── File size formatter ───────────────────────────────────────────────────

  function formatSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <Dialog open={open} onOpenChange={(val) => !val && onClose()}>
      <DialogContent
        className="border-[var(--ds-border)] bg-[var(--ds-bg-card)] sm:max-w-2xl overflow-hidden"
        showCloseButton
      >
        <DialogHeader>
          <DialogTitle
            className="text-[var(--ds-text-heading)]"
            style={{ fontFamily: "var(--font-heading)" }}
          >
            {t("uploadTitle")}
          </DialogTitle>
          <DialogDescription
            className="text-[var(--ds-text-secondary)]"
            style={{ fontFamily: "var(--font-body)" }}
          >
            {t("uploadAcceptedFormats")}
          </DialogDescription>
        </DialogHeader>

        {mode === "upload" ? (
          <div className="min-w-0 space-y-4">
            {/* Drop zone */}
            <div
              className={`relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-colors ${
                dragActive
                  ? "border-[var(--ds-primary)] bg-[var(--ds-primary-light)]"
                  : "border-[var(--ds-border)] hover:border-[var(--ds-text-muted)]"
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <Upload className="mb-3 size-8 text-[var(--ds-text-muted)]" />
              <p
                className="text-sm text-[var(--ds-text-secondary)]"
                style={{ fontFamily: "var(--font-body)" }}
              >
                {t("uploadDragDrop")}
              </p>
              <p className="mt-1 text-xs text-[var(--ds-text-muted)]">
                {t("uploadAcceptedFormats")}
              </p>
              <input
                type="file"
                multiple
                className="absolute inset-0 cursor-pointer opacity-0"
                accept={ACCEPT_STRING}
                onChange={handleFileInput}
              />
            </div>

            {/* File list */}
            {files.length > 0 && (
              <div className="max-h-[300px] space-y-1 overflow-y-auto">
                {files.map((file, i) => (
                  <div
                    key={`${file.name}-${i}`}
                    className="flex min-w-0 items-center justify-between rounded-md border border-[var(--ds-border)] bg-[var(--ds-bg-page)] px-3 py-2"
                  >
                    <div className="flex min-w-0 items-center gap-2 overflow-hidden">
                      <FileText className="size-4 shrink-0 text-[var(--ds-primary)]" />
                      <span className="truncate text-sm text-[var(--ds-text-heading)]">
                        {file.name}
                      </span>
                      <span className="shrink-0 text-xs text-[var(--ds-text-muted)]">
                        {formatSize(file.size)}
                      </span>
                    </div>
                    <button
                      onClick={() => removeFile(i)}
                      className="ml-2 shrink-0 text-[var(--ds-text-muted)] hover:text-[var(--ds-text-heading)]"
                    >
                      <X className="size-4" />
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* Error */}
            {errorMsg && (
              <p className="text-sm text-[var(--ds-error)]">{errorMsg}</p>
            )}

            {/* Extraction progress */}
            {isExtracting && taskData && (
              <div className="space-y-2 rounded-lg border border-[var(--ds-primary)]/30 bg-[var(--ds-primary-light)] p-3">
                <div className="flex items-center gap-2">
                  <Loader2 className="size-4 animate-spin text-[var(--ds-primary)]" />
                  <span
                    className="text-sm text-[var(--ds-text-secondary)]"
                    style={{ fontFamily: "var(--font-body)" }}
                  >
                    {taskData.progress_message || t("extracting")}
                  </span>
                </div>
                <div className="h-1 overflow-hidden rounded-full bg-[var(--ds-border)]">
                  <div
                    className="h-full rounded-full bg-[var(--ds-primary)] transition-all duration-300"
                    style={{
                      width: `${Math.max(taskData.progress * 100, 5)}%`,
                    }}
                  />
                </div>
              </div>
            )}
          </div>
        ) : (
          /* Review mode */
          <div className="min-w-0 space-y-3">
            <div className="flex items-center justify-between">
              <h4
                className="text-sm font-medium text-[var(--ds-text-heading)]"
                style={{ fontFamily: "var(--font-heading)" }}
              >
                {t("reviewTitle")} ({reviewPapers.length})
              </h4>
              <Button
                variant="ghost"
                size="sm"
                className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
                onClick={toggleSelectAll}
              >
                {selectedIndices.size === reviewPapers.length
                  ? t("deselectAll" as Parameters<typeof t>[0])
                  : t("addAll")}
              </Button>
            </div>

            <div className="max-h-[400px] space-y-2 overflow-y-auto pr-1">
              {reviewPapers.map((paper, i) => (
                <div
                  key={i}
                  className="rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-page)] p-3"
                >
                  <div className="mb-2 flex items-start gap-2">
                    <input
                      type="checkbox"
                      checked={selectedIndices.has(i)}
                      onChange={() => toggleSelect(i)}
                      className="mt-1 size-4 shrink-0 rounded border-[var(--ds-border)] accent-[var(--ds-primary)]"
                    />
                    <div className="flex-1 space-y-2">
                      <Input
                        value={paper.title}
                        onChange={(e) => updatePaper(i, "title", e.target.value)}
                        placeholder={t("placeholderTitle")}
                        className="border-[var(--ds-border)] bg-[var(--ds-bg-card)] text-sm text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)]"
                      />
                      <div className="grid grid-cols-2 gap-2">
                        <Input
                          value={paper.authors.map((a) => a.name).join(", ")}
                          onChange={(e) =>
                            updatePaper(i, "authors", e.target.value)
                          }
                          placeholder={t("placeholderAuthors")}
                          className="border-[var(--ds-border)] bg-[var(--ds-bg-card)] text-sm text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)]"
                        />
                        <Input
                          value={paper.year?.toString() || ""}
                          onChange={(e) =>
                            updatePaper(i, "year", e.target.value)
                          }
                          placeholder={t("placeholderYear")}
                          type="number"
                          className="border-[var(--ds-border)] bg-[var(--ds-bg-card)] text-sm text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)]"
                        />
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        <Input
                          value={paper.journal_name || ""}
                          onChange={(e) =>
                            updatePaper(i, "journal_name", e.target.value)
                          }
                          placeholder={t("placeholderJournal")}
                          className="border-[var(--ds-border)] bg-[var(--ds-bg-card)] text-sm text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)]"
                        />
                        <Input
                          value={paper.doi || ""}
                          onChange={(e) =>
                            updatePaper(i, "doi", e.target.value)
                          }
                          placeholder={t("placeholderDoi")}
                          className="border-[var(--ds-border)] bg-[var(--ds-bg-card)] text-sm text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)]"
                        />
                      </div>
                    </div>
                  </div>
                  {paper.file_name && (
                    <p className="ml-6 text-xs text-[var(--ds-text-muted)]">
                      <FileText className="mr-1 inline size-3" />
                      {paper.file_name}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        <DialogFooter className="border-[var(--ds-border)] bg-[var(--ds-bg-subtle)]">
          <Button
            variant="ghost"
            className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
            onClick={onClose}
          >
            {t("clear" as Parameters<typeof t>[0])}
          </Button>

          {mode === "upload" ? (
            <Button
              onClick={handleExtract}
              disabled={
                files.length === 0 ||
                uploadArticles.isPending ||
                isExtracting
              }
              className="bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
            >
              {uploadArticles.isPending || isExtracting ? (
                <Loader2 className="mr-1 size-4 animate-spin" />
              ) : (
                <Upload className="mr-1 size-4" />
              )}
              {isExtracting
                ? t("extracting")
                : t("uploadButton")}
            </Button>
          ) : (
            <Button
              onClick={handleConfirm}
              disabled={
                selectedIndices.size === 0 || confirmUpload.isPending
              }
              className="bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
            >
              {confirmUpload.isPending ? (
                <Loader2 className="mr-1 size-4 animate-spin" />
              ) : (
                <Check className="mr-1 size-4" />
              )}
              {t("addSelected", { count: selectedIndices.size })}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
