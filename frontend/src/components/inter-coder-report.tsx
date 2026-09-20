"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import {
  ChevronDown,
  ChevronRight,
  Loader2,
  Scale,
  Users,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { InterCoderReport as InterCoderReportType } from "@/hooks/use-qualitative";

interface InterCoderReportProps {
  report: InterCoderReportType | undefined;
  isLoading: boolean;
  onRunReport: () => void;
  isTriggering: boolean;
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

function KappaValue({
  value,
  notComputableLabel,
  className,
}: {
  value: number | null;
  notComputableLabel: string;
  className?: string;
}) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return (
      <span className={`italic text-[var(--ds-text-muted)] ${className ?? ""}`}>
        {notComputableLabel}
      </span>
    );
  }
  return <span className={className}>{value.toFixed(3)}</span>;
}

export function InterCoderReportView({
  report,
  isLoading,
  onRunReport,
  isTriggering,
}: InterCoderReportProps) {
  const t = useTranslations("dataAnalysis");
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    perCode: true,
    disagreements: true,
  });
  const [expandedDisagreements, setExpandedDisagreements] = useState<
    Set<string>
  >(new Set());

  function toggleSection(key: string) {
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  function toggleDisagreement(segId: string) {
    setExpandedDisagreements((prev) => {
      const next = new Set(prev);
      if (next.has(segId)) {
        next.delete(segId);
      } else {
        next.add(segId);
      }
      return next;
    });
  }

  return (
    <div className="space-y-4">
      {/* Run Report Button */}
      <div className="flex justify-center">
        <Button
          onClick={onRunReport}
          disabled={isTriggering || isLoading}
          variant="outline"
          className="gap-2 border-[var(--ds-border)] text-[var(--ds-text-body)] hover:border-[var(--ds-primary)]/50 hover:text-[var(--ds-primary)]"
        >
          {isTriggering ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Scale className="h-4 w-4" />
          )}
          {t("interCoder.runReport")}
        </Button>
      </div>

      {isLoading && (
        <div className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 animate-spin text-[var(--ds-text-secondary)]" />
        </div>
      )}

      {report && (
        <>
          {/* Overall Summary */}
          <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
            <CardContent className="py-4">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <span className="text-xs text-[var(--ds-text-secondary)]">
                    {t("interCoder.overallKappa")}
                  </span>
                  <div className="flex items-center gap-3">
                    <KappaValue
                      value={report.overall_kappa}
                      notComputableLabel={t("interCoder.kappaNotComputable")}
                      className="text-2xl font-bold text-[var(--ds-text-heading)]"
                    />
                  </div>
                </div>
                <div className="text-right space-y-1">
                  <span className="text-xs text-[var(--ds-text-secondary)]">
                    {t("interCoder.agreementPct")}
                  </span>
                  <p className="text-2xl font-bold text-[var(--ds-text-heading)]">
                    {report.agreement_pct.toFixed(1)}%
                  </p>
                </div>
              </div>
              {report.overall_kappa_explanation && (
                <div className="mt-2 text-xs text-[var(--ds-text-muted)]">
                  {report.overall_kappa_explanation}
                </div>
              )}
              <div className="mt-2 text-xs text-[var(--ds-text-muted)]">
                {t("interCoder.disagreements")}: {report.disagreement_segments.length}
              </div>
            </CardContent>
          </Card>

          {/* Per-Code Kappa Table */}
          {report.per_code_kappa.length > 0 && (
            <Collapsible
              open={openSections.perCode}
              onOpenChange={() => toggleSection("perCode")}
            >
              <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
                <CardHeader className="cursor-pointer pb-3">
                  <CollapsibleTrigger className="w-full">
                    <SectionHeader
                      icon={Scale}
                      title={t("interCoder.perCodeKappa")}
                      open={openSections.perCode}
                    />
                  </CollapsibleTrigger>
                </CardHeader>
                <CollapsibleContent>
                  <CardContent className="pt-0">
                    <div className="rounded-lg border border-[var(--ds-border)] overflow-hidden">
                      <Table>
                        <TableHeader>
                          <TableRow className="border-[var(--ds-border)] hover:bg-transparent">
                            <TableHead className="bg-[var(--ds-bg-subtle)] text-xs text-[var(--ds-text-body)]">
                              {t("interCoder.code")}
                            </TableHead>
                            <TableHead className="bg-[var(--ds-bg-subtle)] text-xs text-[var(--ds-text-body)]">
                              {t("interCoder.kappa")}
                            </TableHead>
                            <TableHead className="bg-[var(--ds-bg-subtle)] text-xs text-[var(--ds-text-body)]">
                              {t("interCoder.agreement")}
                            </TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {report.per_code_kappa.map((pc) => (
                            <TableRow
                              key={pc.code_name}
                              className="border-[var(--ds-border)]/50 hover:bg-[var(--ds-bg-hover)]"
                            >
                              <TableCell className="text-xs font-medium text-[var(--ds-text-body)]">
                                {pc.code_name}
                              </TableCell>
                              <TableCell className="text-xs text-[var(--ds-text-body)]">
                                <KappaValue
                                  value={pc.kappa}
                                  notComputableLabel={t("interCoder.kappaNotComputable")}
                                />
                              </TableCell>
                              <TableCell className="text-xs text-[var(--ds-text-body)]">
                                {pc.agreement_pct.toFixed(1)}%
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </CardContent>
                </CollapsibleContent>
              </Card>
            </Collapsible>
          )}

          {/* Disagreement Review */}
          {report.disagreement_segments.length > 0 && (
            <Collapsible
              open={openSections.disagreements}
              onOpenChange={() => toggleSection("disagreements")}
            >
              <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
                <CardHeader className="cursor-pointer pb-3">
                  <CollapsibleTrigger className="w-full">
                    <SectionHeader
                      icon={Users}
                      title={`${t("interCoder.disagreements")} (${report.disagreement_segments.length})`}
                      open={openSections.disagreements}
                    />
                  </CollapsibleTrigger>
                </CardHeader>
                <CollapsibleContent>
                  <CardContent className="space-y-2 pt-0">
                    {report.disagreement_segments.map((d) => (
                      <div
                        key={d.segment_id}
                        className="rounded-lg border border-[var(--ds-border)]/50 overflow-hidden"
                      >
                        <button
                          className="w-full flex items-center gap-2 p-2 text-left hover:bg-[var(--ds-bg-card)]"
                          onClick={() => toggleDisagreement(d.segment_id)}
                        >
                          {expandedDisagreements.has(d.segment_id) ? (
                            <ChevronDown className="h-3 w-3 text-[var(--ds-text-muted)] shrink-0" />
                          ) : (
                            <ChevronRight className="h-3 w-3 text-[var(--ds-text-muted)] shrink-0" />
                          )}
                          <p className="text-xs text-[var(--ds-text-body)] truncate flex-1">
                            &ldquo;{d.segment_id}&rdquo;
                          </p>
                        </button>
                        {expandedDisagreements.has(d.segment_id) && (
                          <div className="border-t border-[var(--ds-border)]/50 p-2 space-y-2 bg-[var(--ds-bg-subtle)]">
                            <p className="text-xs text-[var(--ds-text-body)] whitespace-pre-line">
                              &ldquo;{d.segment_id}&rdquo;
                            </p>
                            <div className="grid grid-cols-2 gap-2">
                              <div>
                                <span className="text-[10px] text-[var(--ds-text-muted)]">
                                  {t("interCoder.coder1")}:
                                </span>
                                <div className="flex flex-wrap gap-1 mt-0.5">
                                  {d.ai_codes.map((c) => (
                                    <Badge
                                      key={c}
                                      variant="outline"
                                      className="text-[10px] border-blue-500/50 text-blue-400"
                                    >
                                      {c}
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                              <div>
                                <span className="text-[10px] text-[var(--ds-text-muted)]">
                                  {t("interCoder.coder2")}:
                                </span>
                                <div className="flex flex-wrap gap-1 mt-0.5">
                                  {d.human_codes.map((c) => (
                                    <Badge
                                      key={c}
                                      variant="outline"
                                      className="text-[10px] border-amber-500/50 text-amber-400"
                                    >
                                      {c}
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </CardContent>
                </CollapsibleContent>
              </Card>
            </Collapsible>
          )}
        </>
      )}

      {!report && !isLoading && (
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardContent className="py-8 text-center">
            <Scale className="mx-auto h-8 w-8 text-[var(--ds-text-muted)] mb-2" />
            <p className="text-sm text-[var(--ds-text-secondary)]">
              {t("interCoder.noReport")}
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
