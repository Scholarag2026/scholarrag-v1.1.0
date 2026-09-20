"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import {
  ChevronDown,
  ChevronRight,
  Columns,
  Rows3,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import type { DatasetDetail } from "@/hooks/use-datasets";

interface DataSummaryPanelProps {
  dataset: DatasetDetail;
}

const dtypeBadgeColors: Record<string, string> = {
  numeric: "border-blue-500/50 text-blue-400",
  categorical: "border-amber-500/50 text-amber-400",
  text: "border-emerald-500/50 text-emerald-400",
  datetime: "border-purple-500/50 text-purple-400",
  boolean: "border-[var(--ds-border)] text-[var(--ds-text-secondary)]",
};

export function DataSummaryPanel({ dataset }: DataSummaryPanelProps) {
  const t = useTranslations("dataAnalysis");
  const [open, setOpen] = useState(true);

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
        <CardHeader className="cursor-pointer pb-3">
          <CollapsibleTrigger className="w-full">
            <div className="flex items-center gap-2">
              <Rows3 className="h-4 w-4 text-[var(--ds-primary)]" />
              <span className="text-sm font-semibold text-[var(--ds-text-body)]">
                {t("summary.title")}
              </span>
              <div className="ml-2 flex items-center gap-2">
                <Badge variant="outline" className="border-blue-500/50 text-blue-400">
                  {dataset.row_count} {t("summary.rows")}
                </Badge>
                <Badge variant="outline" className="border-emerald-500/50 text-emerald-400">
                  {dataset.column_count} {t("summary.columns")}
                </Badge>
              </div>
              {open ? (
                <ChevronDown className="ml-auto h-4 w-4 text-[var(--ds-text-secondary)]" />
              ) : (
                <ChevronRight className="ml-auto h-4 w-4 text-[var(--ds-text-secondary)]" />
              )}
            </div>
          </CollapsibleTrigger>
        </CardHeader>
        <CollapsibleContent>
          <CardContent className="space-y-3 pt-0">
            <div className="text-xs text-[var(--ds-text-secondary)] mb-2">
              {dataset.filename}
              {dataset.selected_sheet && (
                <span className="ml-2 text-[var(--ds-text-muted)]">
                  ({t("summary.sheet")}: {dataset.selected_sheet})
                </span>
              )}
            </div>

            {/* Per-column stats */}
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {dataset.columns.map((col) => (
                <div
                  key={col.name}
                  className="rounded-lg border border-[var(--ds-border)] p-2 space-y-1"
                >
                  <div className="flex items-center gap-1.5">
                    <Columns className="h-3 w-3 text-[var(--ds-text-muted)]" />
                    <span className="text-xs font-medium text-[var(--ds-text-body)] truncate">
                      {col.name}
                    </span>
                    <Badge
                      variant="outline"
                      className={`ml-auto text-[10px] ${
                        dtypeBadgeColors[col.dtype] ||
                        "border-[var(--ds-border)] text-[var(--ds-text-secondary)]"
                      }`}
                    >
                      {col.dtype}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-2 text-[10px]">
                    <span className="text-[var(--ds-text-muted)]">
                      {col.unique_count} {t("summary.unique")}
                    </span>
                    {col.missing_count > 0 && (
                      <Badge
                        variant="outline"
                        className={`text-[10px] ${
                          col.missing_pct > 40
                            ? "border-[var(--ds-error)]/50 text-[var(--ds-error)]"
                            : col.missing_pct > 20
                              ? "border-amber-500/50 text-amber-400"
                              : "border-[var(--ds-border)] text-[var(--ds-text-secondary)]"
                        }`}
                      >
                        {col.missing_count} {t("summary.missingLabel")} ({col.missing_pct.toFixed(1)}%)
                      </Badge>
                    )}
                  </div>
                  {col.role && (
                    <Badge
                      variant="outline"
                      className="text-[10px] border-[var(--ds-primary)]/50 text-[var(--ds-primary)]"
                    >
                      {col.role.replace("_", " ")}
                    </Badge>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}
