"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { ChevronLeft, ChevronRight } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { ColumnInfo, DatasetDetail } from "@/hooks/use-datasets";
import { useUpdateDataset } from "@/hooks/use-datasets";

interface DataPreviewTableProps {
  dataset: DatasetDetail;
}

const ROWS_PER_PAGE = 50;

const dtypeColors: Record<string, string> = {
  numeric: "border-blue-500/50 text-blue-400",
  categorical: "border-amber-500/50 text-amber-400",
  text: "border-emerald-500/50 text-emerald-400",
  datetime: "border-purple-500/50 text-purple-400",
  boolean: "border-[var(--ds-border)] text-[var(--ds-text-secondary)]",
};

const roleOptions = [
  "independent",
  "dependent",
  "control",
  "participant_id",
  "text_data",
] as const;

export function DataPreviewTable({ dataset }: DataPreviewTableProps) {
  const t = useTranslations("dataAnalysis");
  const updateDataset = useUpdateDataset();

  const [page, setPage] = useState(0);

  const rows = dataset.preview_rows || [];
  const totalPages = Math.ceil(rows.length / ROWS_PER_PAGE);
  const pagedRows = rows.slice(
    page * ROWS_PER_PAGE,
    (page + 1) * ROWS_PER_PAGE,
  );

  function handleRoleChange(columnName: string, role: string) {
    const updatedColumns = dataset.columns.map((col) =>
      col.name === columnName
        ? { ...col, role: role as ColumnInfo["role"] }
        : col,
    );
    updateDataset.mutate({
      datasetId: dataset.id,
      columns: updatedColumns,
    });
  }

  return (
    <div className="space-y-3">
      {/* Column headers with metadata */}
      <div className="rounded-lg border border-[var(--ds-border)] overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow className="border-[var(--ds-border)] hover:bg-transparent">
              {dataset.columns.map((col) => (
                <TableHead
                  key={col.name}
                  className="border-r border-[var(--ds-border)] last:border-r-0 bg-[var(--ds-bg-subtle)] text-[var(--ds-text-body)]"
                >
                  <div className="space-y-1">
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs font-semibold">{col.name}</span>
                      <Badge
                        variant="outline"
                        className={dtypeColors[col.dtype] || "border-[var(--ds-border)] text-[var(--ds-text-secondary)]"}
                      >
                        {col.dtype}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      {col.missing_count > 0 && (
                        <span
                          className={`text-[10px] ${
                            col.missing_pct > 40
                              ? "text-[var(--ds-error)]"
                              : col.missing_pct > 20
                                ? "text-[var(--ds-warning)]"
                                : "text-[var(--ds-text-muted)]"
                          }`}
                        >
                          {col.missing_count} {t("preview.missing")}
                        </span>
                      )}
                    </div>
                    <select
                      value={col.role || ""}
                      onChange={(e) =>
                        handleRoleChange(col.name, e.target.value)
                      }
                      className="h-5 w-full rounded border border-[var(--ds-border)] bg-[var(--ds-bg-card)] px-1 text-[10px] text-[var(--ds-text-body)] outline-none"
                    >
                      <option value="">{t("preview.noRole")}</option>
                      {roleOptions.map((role) => (
                        <option key={role} value={role}>
                          {t(`preview.role.${role}`)}
                        </option>
                      ))}
                    </select>
                  </div>
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {pagedRows.map((row, i) => (
              <TableRow
                key={i}
                className="border-[var(--ds-border)]/50 hover:bg-[var(--ds-bg-hover)]"
              >
                {dataset.columns.map((col) => (
                  <TableCell
                    key={col.name}
                    className="text-xs text-[var(--ds-text-body)] border-r border-[var(--ds-border)]/30 last:border-r-0"
                  >
                    {row[col.name] != null ? String(row[col.name]) : ""}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <span className="text-xs text-[var(--ds-text-secondary)]">
            {t("preview.page", {
              current: page + 1,
              total: totalPages,
            })}
          </span>
          <div className="flex gap-1">
            <Button
              variant="ghost"
              size="sm"
              disabled={page === 0}
              onClick={() => setPage((p) => p - 1)}
              className="h-7 w-7 p-0 text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              disabled={page >= totalPages - 1}
              onClick={() => setPage((p) => p + 1)}
              className="h-7 w-7 p-0 text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
