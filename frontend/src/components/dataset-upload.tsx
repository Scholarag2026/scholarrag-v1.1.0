"use client";

import { useCallback, useState } from "react";
import { useTranslations } from "next-intl";
import { Loader2, Upload } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { useUploadDataset } from "@/hooks/use-datasets";

interface DatasetUploadProps {
  projectId: string;
  onUploadSuccess?: (datasetId: string) => void;
}

const ACCEPTED_EXTENSIONS = [".csv", ".xlsx", ".xls", ".tsv"];
const MAX_SIZE_MB = 50;
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024;

export function DatasetUpload({ projectId, onUploadSuccess }: DatasetUploadProps) {
  const t = useTranslations("dataAnalysis");
  const uploadDataset = useUploadDataset();

  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [availableSheets, setAvailableSheets] = useState<string[]>([]);
  const [selectedSheet, setSelectedSheet] = useState<string>("");

  function validateFile(file: File): string | null {
    const ext = "." + file.name.split(".").pop()?.toLowerCase();
    if (!ACCEPTED_EXTENSIONS.includes(ext)) {
      return t("upload.invalidType");
    }
    if (file.size > MAX_SIZE_BYTES) {
      return t("upload.fileTooLarge", { max: MAX_SIZE_MB });
    }
    return null;
  }

  function handleFile(file: File) {
    const error = validateFile(file);
    if (error) {
      setErrorMsg(error);
      setSelectedFile(null);
      return;
    }
    setErrorMsg(null);
    setSelectedFile(file);
    setAvailableSheets([]);
    setSelectedSheet("");
  }

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
        handleFile(e.dataTransfer.files[0]);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  function handleFileInput(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  }

  function handleUpload() {
    if (!selectedFile) return;

    uploadDataset.mutate(
      { projectId, file: selectedFile, sheet: selectedSheet || undefined },
      {
        onSuccess: (data) => {
          if (
            data.available_sheets &&
            data.available_sheets.length > 1 &&
            !selectedSheet
          ) {
            setAvailableSheets(data.available_sheets);
            setSelectedSheet(data.selected_sheet || data.available_sheets[0]);
            return;
          }
          setSelectedFile(null);
          setAvailableSheets([]);
          setSelectedSheet("");
          onUploadSuccess?.(data.id);
        },
        onError: (err) => {
          setErrorMsg(err.message);
        },
      },
    );
  }

  return (
    <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
      <CardContent className="pt-6 space-y-4">
        <div className="space-y-2">
          <label className="text-sm font-medium text-[var(--ds-text-body)]">
            {t("upload.title")}
          </label>

          {/* Drop Zone */}
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
            <Upload className="mb-3 h-8 w-8 text-[var(--ds-text-secondary)]" />
            <p className="text-sm text-[var(--ds-text-body)]">{t("upload.dragDrop")}</p>
            <p className="mt-1 text-xs text-[var(--ds-text-muted)]">
              {t("upload.acceptedFormats")}
            </p>
            <input
              type="file"
              className="absolute inset-0 cursor-pointer opacity-0"
              accept={ACCEPTED_EXTENSIONS.join(",")}
              onChange={handleFileInput}
            />
          </div>
        </div>

        {/* Selected File */}
        {selectedFile && (
          <div className="flex items-center justify-between rounded-md border border-[var(--ds-border)] bg-[var(--ds-bg-subtle)] px-3 py-2">
            <span className="text-sm text-[var(--ds-text-body)] truncate">
              {selectedFile.name}
            </span>
            <span className="ml-2 shrink-0 text-xs text-[var(--ds-text-secondary)]">
              {(selectedFile.size / 1024 / 1024).toFixed(1)} MB
            </span>
          </div>
        )}

        {/* Sheet Selector */}
        {availableSheets.length > 1 && (
          <div className="space-y-1">
            <label className="text-xs text-[var(--ds-text-secondary)]">
              {t("upload.selectSheet")}
            </label>
            <select
              value={selectedSheet}
              onChange={(e) => setSelectedSheet(e.target.value)}
              className="flex h-8 w-full rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-subtle)] px-3 text-sm text-[var(--ds-text-body)] outline-none focus:border-[var(--ds-primary)] focus:ring-1 focus:ring-[var(--ds-primary)]"
            >
              {availableSheets.map((sheet) => (
                <option key={sheet} value={sheet}>
                  {sheet}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Error */}
        {errorMsg && (
          <p className="text-sm text-[var(--ds-error)]">{errorMsg}</p>
        )}

        {/* Upload Button */}
        {selectedFile && (
          <div className="flex justify-end">
            <Button
              onClick={handleUpload}
              disabled={uploadDataset.isPending}
              className="gap-2"
            >
              {uploadDataset.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Upload className="h-4 w-4" />
              )}
              {uploadDataset.isPending
                ? t("upload.uploading")
                : t("upload.uploadButton")}
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
