"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import {
  BarChart3,
  Calculator,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  ClipboardCheck,
  Code2,
  Copy,
  FileText,
  FlaskConical,
  TrendingUp,
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
import type { AnalysisResults } from "@/hooks/use-quantitative";

interface QuantitativeViewProps {
  results: AnalysisResults;
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

const confidenceColors: Record<string, string> = {
  high: "border-emerald-500/50 text-emerald-400",
  medium: "border-amber-500/50 text-amber-400",
  low: "border-red-500/50 text-red-400",
};

export function QuantitativeView({ results }: QuantitativeViewProps) {
  const t = useTranslations("dataAnalysis");
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    plan: true,
    descriptive: true,
    correlations: true,
    assumptions: true,
    predictions: true,
    code: true,
    interpretation: true,
  });
  const [activeCodeTab, setActiveCodeTab] = useState<string>("python");
  const [copiedCode, setCopiedCode] = useState(false);

  function toggleSection(key: string) {
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  async function handleCopyCode(code: string) {
    try {
      await navigator.clipboard.writeText(code);
      setCopiedCode(true);
      setTimeout(() => setCopiedCode(false), 2000);
    } catch {
      // Clipboard API not available
    }
  }

  return (
    <div className="space-y-4">
      {/* 1. Analysis Plan */}
      <Collapsible
        open={openSections.plan}
        onOpenChange={() => toggleSection("plan")}
      >
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardHeader className="cursor-pointer pb-3">
            <CollapsibleTrigger className="w-full">
              <SectionHeader
                icon={FlaskConical}
                title={t("quantitative.analysisPlan")}
                open={openSections.plan}
              />
            </CollapsibleTrigger>
          </CardHeader>
          <CollapsibleContent>
            <CardContent className="space-y-3 pt-0">
              {results.plan.overall_strategy && (
                <div>
                  <span className="text-xs text-[var(--ds-text-secondary)]">
                    {t("quantitative.overallStrategy")}:
                  </span>
                  <p className="text-sm text-[var(--ds-text-body)] mt-0.5 whitespace-pre-line">
                    {results.plan.overall_strategy}
                  </p>
                </div>
              )}
              {results.plan.methods.map((method, i) => (
                <div
                  key={i}
                  className="rounded-lg border border-[var(--ds-border)] p-3 space-y-2"
                >
                  <span className="text-sm font-medium text-[var(--ds-text-body)]">
                    {method.method}
                  </span>
                  <p className="text-xs text-[var(--ds-text-body)]">
                    {method.justification}
                  </p>
                  {method.variables.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {method.variables.map((v, j) => (
                        <Badge
                          key={j}
                          variant="outline"
                          className="text-[10px] border-[var(--ds-border)] text-[var(--ds-text-secondary)]"
                        >
                          {v}
                        </Badge>
                      ))}
                    </div>
                  )}
                  {method.assumptions.length > 0 && (
                    <div>
                      <span className="text-[10px] text-[var(--ds-text-muted)]">
                        {t("quantitative.assumptions")}:
                      </span>
                      <ul className="mt-0.5 list-disc list-inside space-y-0.5">
                        {method.assumptions.map((a, j) => (
                          <li key={j} className="text-xs text-[var(--ds-text-secondary)]">
                            {a}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ))}
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>

      {/* 2. Descriptive Statistics */}
      <Collapsible
        open={openSections.descriptive}
        onOpenChange={() => toggleSection("descriptive")}
      >
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardHeader className="cursor-pointer pb-3">
            <CollapsibleTrigger className="w-full">
              <SectionHeader
                icon={Calculator}
                title={t("quantitative.descriptiveStats")}
                open={openSections.descriptive}
              />
            </CollapsibleTrigger>
          </CardHeader>
          <CollapsibleContent>
            <CardContent className="space-y-4 pt-0">
              {/* Numeric Stats Table */}
              {results.descriptive_stats.numeric.length > 0 && (
                <div>
                  <span className="text-xs text-[var(--ds-text-secondary)] mb-1 block">
                    {t("quantitative.numericVariables")}
                  </span>
                  <div className="rounded-lg border border-[var(--ds-border)] overflow-hidden">
                    <Table>
                      <TableHeader>
                        <TableRow className="border-[var(--ds-border)] hover:bg-transparent">
                          <TableHead className="bg-[var(--ds-bg-subtle)] text-xs text-[var(--ds-text-body)]">
                            {t("quantitative.variable")}
                          </TableHead>
                          <TableHead className="bg-[var(--ds-bg-subtle)] text-xs text-[var(--ds-text-body)]">M</TableHead>
                          <TableHead className="bg-[var(--ds-bg-subtle)] text-xs text-[var(--ds-text-body)]">Mdn</TableHead>
                          <TableHead className="bg-[var(--ds-bg-subtle)] text-xs text-[var(--ds-text-body)]">SD</TableHead>
                          <TableHead className="bg-[var(--ds-bg-subtle)] text-xs text-[var(--ds-text-body)]">Min</TableHead>
                          <TableHead className="bg-[var(--ds-bg-subtle)] text-xs text-[var(--ds-text-body)]">Max</TableHead>
                          <TableHead className="bg-[var(--ds-bg-subtle)] text-xs text-[var(--ds-text-body)]">Skew</TableHead>
                          <TableHead className="bg-[var(--ds-bg-subtle)] text-xs text-[var(--ds-text-body)]">Kurt</TableHead>
                          <TableHead className="bg-[var(--ds-bg-subtle)] text-xs text-[var(--ds-text-body)]">N</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {results.descriptive_stats.numeric.map((stat) => (
                          <TableRow
                            key={stat.column}
                            className="border-[var(--ds-border)]/50 hover:bg-[var(--ds-bg-hover)]"
                          >
                            <TableCell className="text-xs font-medium text-[var(--ds-text-body)]">
                              {stat.column}
                            </TableCell>
                            <TableCell className="text-xs text-[var(--ds-text-body)]">
                              {stat.mean.toFixed(2)}
                            </TableCell>
                            <TableCell className="text-xs text-[var(--ds-text-body)]">
                              {stat.median.toFixed(2)}
                            </TableCell>
                            <TableCell className="text-xs text-[var(--ds-text-body)]">
                              {stat.std_dev.toFixed(2)}
                            </TableCell>
                            <TableCell className="text-xs text-[var(--ds-text-body)]">
                              {stat.min.toFixed(2)}
                            </TableCell>
                            <TableCell className="text-xs text-[var(--ds-text-body)]">
                              {stat.max.toFixed(2)}
                            </TableCell>
                            <TableCell className="text-xs text-[var(--ds-text-body)]">
                              {stat.skewness.toFixed(2)}
                            </TableCell>
                            <TableCell className="text-xs text-[var(--ds-text-body)]">
                              {stat.kurtosis.toFixed(2)}
                            </TableCell>
                            <TableCell className="text-xs text-[var(--ds-text-body)]">
                              {stat.n}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </div>
              )}

              {/* Categorical Stats */}
              {results.descriptive_stats.categorical.length > 0 && (
                <div>
                  <span className="text-xs text-[var(--ds-text-secondary)] mb-1 block">
                    {t("quantitative.categoricalVariables")}
                  </span>
                  <div className="grid gap-2 sm:grid-cols-2">
                    {results.descriptive_stats.categorical.map((stat) => (
                      <div
                        key={stat.column}
                        className="rounded-lg border border-[var(--ds-border)] p-2 space-y-1"
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-medium text-[var(--ds-text-body)]">
                            {stat.column}
                          </span>
                          <span className="text-[10px] text-[var(--ds-text-muted)]">
                            N={stat.n}, {t("quantitative.mode")}: {stat.mode}
                          </span>
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {Object.entries(stat.frequencies)
                            .sort(([, a], [, b]) => b - a)
                            .slice(0, 8)
                            .map(([val, count]) => (
                              <Badge
                                key={val}
                                variant="outline"
                                className="text-[10px] border-amber-500/30 text-amber-400"
                              >
                                {val}: {count}
                              </Badge>
                            ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>

      {/* 3. Correlation Matrix */}
      {results.correlations.length > 0 && (
        <Collapsible
          open={openSections.correlations}
          onOpenChange={() => toggleSection("correlations")}
        >
          <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
            <CardHeader className="cursor-pointer pb-3">
              <CollapsibleTrigger className="w-full">
                <SectionHeader
                  icon={TrendingUp}
                  title={t("quantitative.correlationMatrix")}
                  open={openSections.correlations}
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
                          {t("quantitative.var1")}
                        </TableHead>
                        <TableHead className="bg-[var(--ds-bg-subtle)] text-xs text-[var(--ds-text-body)]">
                          {t("quantitative.var2")}
                        </TableHead>
                        <TableHead className="bg-[var(--ds-bg-subtle)] text-xs text-[var(--ds-text-body)]">r</TableHead>
                        <TableHead className="bg-[var(--ds-bg-subtle)] text-xs text-[var(--ds-text-body)]">p</TableHead>
                        <TableHead className="bg-[var(--ds-bg-subtle)] text-xs text-[var(--ds-text-body)]">
                          {t("quantitative.significant")}
                        </TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {results.correlations.map((corr, i) => (
                        <TableRow
                          key={i}
                          className="border-[var(--ds-border)]/50 hover:bg-[var(--ds-bg-hover)]"
                        >
                          <TableCell className="text-xs text-[var(--ds-text-body)]">
                            {corr.var1}
                          </TableCell>
                          <TableCell className="text-xs text-[var(--ds-text-body)]">
                            {corr.var2}
                          </TableCell>
                          <TableCell className="text-xs text-[var(--ds-text-body)]">
                            {corr.r.toFixed(3)}
                          </TableCell>
                          <TableCell className="text-xs text-[var(--ds-text-body)]">
                            {corr.p_value < 0.001
                              ? "< .001"
                              : corr.p_value.toFixed(3)}
                          </TableCell>
                          <TableCell>
                            {corr.significant ? (
                              <Badge
                                variant="outline"
                                className="border-emerald-500/50 text-emerald-400 text-[10px]"
                              >
                                {t("quantitative.yes")}
                              </Badge>
                            ) : (
                              <Badge
                                variant="outline"
                                className="border-[var(--ds-border)] text-[var(--ds-text-secondary)] text-[10px]"
                              >
                                {t("quantitative.no")}
                              </Badge>
                            )}
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

      {/* 4. Assumption Checks */}
      {results.assumption_checks.length > 0 && (
        <Collapsible
          open={openSections.assumptions}
          onOpenChange={() => toggleSection("assumptions")}
        >
          <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
            <CardHeader className="cursor-pointer pb-3">
              <CollapsibleTrigger className="w-full">
                <SectionHeader
                  icon={CheckCircle2}
                  title={t("quantitative.assumptionChecks")}
                  open={openSections.assumptions}
                />
              </CollapsibleTrigger>
            </CardHeader>
            <CollapsibleContent>
              <CardContent className="space-y-2 pt-0">
                {results.assumption_checks.map((check, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-3 rounded-md border border-[var(--ds-border)]/50 p-2"
                  >
                    <Badge
                      variant="outline"
                      className={`shrink-0 text-[10px] ${
                        check.met
                          ? "border-emerald-500/50 text-emerald-400"
                          : "border-red-500/50 text-red-400"
                      }`}
                    >
                      {check.met
                        ? t("quantitative.met")
                        : t("quantitative.violated")}
                    </Badge>
                    <div className="flex-1 space-y-0.5">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-medium text-[var(--ds-text-body)]">
                          {check.test_name}
                        </span>
                        <span className="text-[10px] text-[var(--ds-text-muted)]">
                          ({check.variable})
                        </span>
                      </div>
                      <p className="text-xs text-[var(--ds-text-secondary)]">
                        {check.interpretation}
                      </p>
                      <span className="text-[10px] text-[var(--ds-text-muted)]">
                        stat={check.statistic.toFixed(3)}, p=
                        {check.p_value < 0.001
                          ? "< .001"
                          : check.p_value.toFixed(3)}
                      </span>
                    </div>
                  </div>
                ))}
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      )}

      {/* 5. Result Predictions */}
      {results.result_predictions.length > 0 && (
        <Collapsible
          open={openSections.predictions}
          onOpenChange={() => toggleSection("predictions")}
        >
          <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
            <CardHeader className="cursor-pointer pb-3">
              <CollapsibleTrigger className="w-full">
                <SectionHeader
                  icon={BarChart3}
                  title={t("quantitative.resultPredictions")}
                  open={openSections.predictions}
                />
              </CollapsibleTrigger>
            </CardHeader>
            <CollapsibleContent>
              <CardContent className="space-y-2 pt-0">
                {results.result_predictions.map((pred, i) => (
                  <div
                    key={i}
                    className="rounded-lg border border-[var(--ds-border)] p-3 space-y-1"
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-medium text-[var(--ds-text-body)]">
                        {pred.hypothesis}
                      </span>
                      <Badge
                        variant="outline"
                        className={`ml-auto text-[10px] ${
                          confidenceColors[pred.confidence] ||
                          "border-[var(--ds-border)] text-[var(--ds-text-secondary)]"
                        }`}
                      >
                        {t(`quantitative.confidence.${pred.confidence}`)}
                      </Badge>
                    </div>
                    <p className="text-xs text-[var(--ds-text-body)]">
                      {pred.predicted_outcome}
                    </p>
                    <p className="text-[10px] text-[var(--ds-text-muted)]">
                      {pred.rationale}
                    </p>
                  </div>
                ))}
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      )}

      {/* 6. Code Templates */}
      {results.code_templates.length > 0 && (
        <Collapsible
          open={openSections.code}
          onOpenChange={() => toggleSection("code")}
        >
          <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
            <CardHeader className="cursor-pointer pb-3">
              <CollapsibleTrigger className="w-full">
                <SectionHeader
                  icon={Code2}
                  title={t("quantitative.codeTemplates")}
                  open={openSections.code}
                />
              </CollapsibleTrigger>
            </CardHeader>
            <CollapsibleContent>
              <CardContent className="space-y-3 pt-0">
                {/* Language tabs */}
                <div className="flex gap-1 rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-subtle)] p-1">
                  {Array.from(
                    new Set(results.code_templates.map((ct) => ct.language)),
                  ).map((lang) => (
                    <button
                      key={lang}
                      onClick={(e) => {
                        e.stopPropagation();
                        setActiveCodeTab(lang);
                      }}
                      className={`rounded-md px-3 py-1 text-xs transition-colors ${
                        activeCodeTab === lang
                          ? "bg-[var(--ds-primary-light)] text-[var(--ds-primary)]"
                          : "text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-body)]"
                      }`}
                    >
                      {lang.toUpperCase()}
                    </button>
                  ))}
                </div>

                {/* Code display */}
                {results.code_templates
                  .filter((ct) => ct.language === activeCodeTab)
                  .map((ct, i) => (
                    <div key={i} className="space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-[var(--ds-text-secondary)]">
                          {ct.description}
                        </span>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 gap-1 text-xs text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-body)]"
                          onClick={() => handleCopyCode(ct.code)}
                        >
                          {copiedCode ? (
                            <>
                              <ClipboardCheck className="h-3 w-3" />
                              {t("common.copied")}
                            </>
                          ) : (
                            <>
                              <Copy className="h-3 w-3" />
                              {t("common.copy")}
                            </>
                          )}
                        </Button>
                      </div>
                      <div className="rounded-md border border-[var(--ds-border)] bg-[var(--ds-bg-subtle)] p-3 overflow-x-auto">
                        <pre className="text-xs text-[var(--ds-text-body)] whitespace-pre">
                          {ct.code}
                        </pre>
                      </div>
                    </div>
                  ))}
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      )}

      {/* 7. Interpretation Guidance */}
      {results.interpretation_guides.length > 0 && (
        <Collapsible
          open={openSections.interpretation}
          onOpenChange={() => toggleSection("interpretation")}
        >
          <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
            <CardHeader className="cursor-pointer pb-3">
              <CollapsibleTrigger className="w-full">
                <SectionHeader
                  icon={FileText}
                  title={t("quantitative.interpretationGuidance")}
                  open={openSections.interpretation}
                />
              </CollapsibleTrigger>
            </CardHeader>
            <CollapsibleContent>
              <CardContent className="space-y-3 pt-0">
                {results.interpretation_guides.map((guide, i) => (
                  <div
                    key={i}
                    className="rounded-lg border border-[var(--ds-border)] p-3 space-y-2"
                  >
                    <span className="text-xs font-medium text-[var(--ds-text-body)]">
                      {guide.section}
                    </span>
                    <div className="rounded-md border border-[var(--ds-border)] bg-[var(--ds-bg-subtle)] p-2">
                      <span className="text-[10px] text-[var(--ds-text-muted)] block mb-1">
                        {t("quantitative.apaExample")}:
                      </span>
                      <p className="text-xs text-[var(--ds-text-body)] italic whitespace-pre-line">
                        {guide.apa_example}
                      </p>
                    </div>
                    {guide.notes && (
                      <p className="text-xs text-[var(--ds-text-secondary)]">{guide.notes}</p>
                    )}
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
