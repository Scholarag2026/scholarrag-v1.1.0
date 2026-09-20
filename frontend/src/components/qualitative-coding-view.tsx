"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import {
  AlertTriangle,
  BookOpen,
  ChevronDown,
  ChevronRight,
  Info,
  Layers,
  Tags,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import type { CodingResult } from "@/hooks/use-qualitative";

interface QualitativeCodingViewProps {
  results: CodingResult;
}

function SectionHeader({
  icon: Icon,
  title,
  open,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  open: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      <Icon className="h-4 w-4 text-[var(--ds-primary)]" />
      <span className="text-sm font-semibold text-[var(--ds-text-body)]">{title}</span>
      {open ? (
        <ChevronDown className="ml-auto h-4 w-4 text-[var(--ds-text-secondary)]" />
      ) : (
        <ChevronRight className="ml-auto h-4 w-4 text-[var(--ds-text-secondary)]" />
      )}
    </div>
  );
}

const confidenceColor = (c: number) => {
  if (c >= 0.8) return "border-emerald-500/50 text-emerald-400";
  if (c >= 0.5) return "border-amber-500/50 text-amber-400";
  return "border-red-500/50 text-red-400";
};

export function QualitativeCodingView({ results }: QualitativeCodingViewProps) {
  const t = useTranslations("dataAnalysis");
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    codebook: true,
    segments: true,
    uncertainties: true,
    notes: true,
  });

  function toggleSection(key: string) {
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  const progressPct =
    results.total_segments > 0
      ? Math.round(
          (results.reviewed_segments / results.total_segments) * 100,
        )
      : 0;

  return (
    <div className="space-y-4">
      {/* Progress Bar */}
      <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
        <CardContent className="py-3">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-[var(--ds-text-secondary)]">
              {t("qualitative.progress")}
            </span>
            <span className="text-xs text-[var(--ds-text-body)]">
              {results.reviewed_segments}/{results.total_segments} ({progressPct}%)
            </span>
          </div>
          <div className="h-2 w-full rounded-full bg-[var(--ds-border)]">
            <div
              className="h-2 rounded-full bg-[var(--ds-primary)] transition-all"
              style={{ width: `${progressPct}%` }}
            />
          </div>
        </CardContent>
      </Card>

      {/* 1. Codebook Display */}
      <Collapsible
        open={openSections.codebook}
        onOpenChange={() => toggleSection("codebook")}
      >
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardHeader className="cursor-pointer pb-3">
            <CollapsibleTrigger className="w-full">
              <SectionHeader
                icon={BookOpen}
                title={t("qualitative.codebook")}
                open={openSections.codebook}
              />
            </CollapsibleTrigger>
          </CardHeader>
          <CollapsibleContent>
            <CardContent className="space-y-3 pt-0">
              {results.codebook.themes.map((theme, i) => (
                <div
                  key={i}
                  className="rounded-lg border border-[var(--ds-border)] p-3 space-y-2"
                >
                  <div className="flex items-center gap-2">
                    <Layers className="h-3.5 w-3.5 text-[var(--ds-primary)]" />
                    <span className="text-sm font-medium text-[var(--ds-text-body)]">
                      {theme.theme}
                    </span>
                  </div>
                  <p className="text-xs text-[var(--ds-text-secondary)]">{theme.description}</p>
                  <div className="ml-4 space-y-1.5">
                    {theme.codes.map((code, j) => (
                      <div
                        key={j}
                        className="rounded-md border border-[var(--ds-border)]/50 p-2 space-y-1"
                      >
                        <div className="flex items-center gap-2">
                          <Tags className="h-3 w-3 text-[var(--ds-text-muted)]" />
                          <span className="text-xs font-medium text-[var(--ds-text-body)]">
                            {code.code}
                          </span>
                        </div>
                        <p className="text-[10px] text-[var(--ds-text-secondary)]">
                          {code.definition}
                        </p>
                        {code.examples.length > 0 && (
                          <div className="flex flex-wrap gap-1 mt-1">
                            {code.examples.map((ex, k) => (
                              <span
                                key={k}
                                className="inline-block rounded border border-[var(--ds-border)] bg-[var(--ds-bg-subtle)] px-1.5 py-0.5 text-[10px] text-[var(--ds-text-secondary)] italic"
                              >
                                &ldquo;{ex}&rdquo;
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>

      {/* 2. Coding Segments */}
      <Collapsible
        open={openSections.segments}
        onOpenChange={() => toggleSection("segments")}
      >
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardHeader className="cursor-pointer pb-3">
            <CollapsibleTrigger className="w-full">
              <SectionHeader
                icon={Tags}
                title={`${t("qualitative.codedSegments")} (${results.segments.length})`}
                open={openSections.segments}
              />
            </CollapsibleTrigger>
          </CardHeader>
          <CollapsibleContent>
            <CardContent className="space-y-2 pt-0">
              {results.segments.map((seg) => (
                <div
                  key={seg.segment_id}
                  className="rounded-lg border border-[var(--ds-border)]/50 p-2 space-y-1"
                >
                  <div className="flex items-start justify-between gap-2">
                    <p className="text-xs text-[var(--ds-text-body)] flex-1 line-clamp-3">
                      &ldquo;{seg.text}&rdquo;
                    </p>
                    <Badge
                      variant="outline"
                      className={`shrink-0 text-[10px] ${confidenceColor(seg.confidence)}`}
                    >
                      {(seg.confidence * 100).toFixed(0)}%
                    </Badge>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {seg.codes.map((code) => (
                      <Badge
                        key={code}
                        variant="outline"
                        className="text-[10px] border-[var(--ds-primary)]/50 text-[var(--ds-primary)]"
                      >
                        {code}
                      </Badge>
                    ))}
                  </div>
                  <div className="flex items-center gap-2 text-[10px] text-[var(--ds-text-muted)]">
                    <span>
                      {t("qualitative.row")}: {seg.source_row}
                    </span>
                    <span>
                      {t("qualitative.column")}: {seg.source_column}
                    </span>
                  </div>
                  {seg.notes && (
                    <p className="text-[10px] text-[var(--ds-text-muted)] italic">
                      {seg.notes}
                    </p>
                  )}
                </div>
              ))}
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>

      {/* 3. Uncertainties */}
      {results.uncertainties.length > 0 && (
        <Collapsible
          open={openSections.uncertainties}
          onOpenChange={() => toggleSection("uncertainties")}
        >
          <Card className="border-amber-500/20 bg-[var(--ds-bg-card)]">
            <CardHeader className="cursor-pointer pb-3">
              <CollapsibleTrigger className="w-full">
                <SectionHeader
                  icon={AlertTriangle}
                  title={`${t("qualitative.uncertainties")} (${results.uncertainties.length})`}
                  open={openSections.uncertainties}
                />
              </CollapsibleTrigger>
            </CardHeader>
            <CollapsibleContent>
              <CardContent className="space-y-2 pt-0">
                {results.uncertainties.map((u) => (
                  <div
                    key={u.segment_id}
                    className="rounded-md border border-amber-500/30 bg-amber-500/5 p-2 space-y-1"
                  >
                    <p className="text-xs text-[var(--ds-text-body)]">
                      &ldquo;{u.text}&rdquo;
                    </p>
                    <p className="text-[10px] text-amber-400">{u.reason}</p>
                    <div className="flex flex-wrap gap-1">
                      {u.candidate_codes.map((code) => (
                        <Badge
                          key={code}
                          variant="outline"
                          className="text-[10px] border-amber-500/30 text-amber-400"
                        >
                          {code}
                        </Badge>
                      ))}
                    </div>
                  </div>
                ))}
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      )}

      {/* 4. Annotation Notes */}
      {results.annotation_notes.length > 0 && (
        <Collapsible
          open={openSections.notes}
          onOpenChange={() => toggleSection("notes")}
        >
          <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
            <CardHeader className="cursor-pointer pb-3">
              <CollapsibleTrigger className="w-full">
                <SectionHeader
                  icon={Info}
                  title={t("qualitative.annotationNotes")}
                  open={openSections.notes}
                />
              </CollapsibleTrigger>
            </CardHeader>
            <CollapsibleContent>
              <CardContent className="space-y-2 pt-0">
                {results.annotation_notes.map((note, i) => (
                  <div
                    key={i}
                    className="rounded-md border border-blue-500/20 bg-blue-500/5 p-2"
                  >
                    <div className="flex items-start gap-2">
                      <Info className="mt-0.5 h-3 w-3 shrink-0 text-blue-400" />
                      <p className="text-xs text-[var(--ds-text-body)]">{note}</p>
                    </div>
                  </div>
                ))}
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      )}
    </div>
  );
}
