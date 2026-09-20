"use client";

import { useTranslations } from "next-intl";
import {
  BarChart3,
  CheckCircle2,
  TrendingDown,
  Shield,
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import type { CoverageMetrics } from "@/hooks/use-deep-search";

interface CoverageReportProps {
  coverage: CoverageMetrics;
}

const confidenceColors: Record<string, string> = {
  high: "border-[var(--ds-success)]/50 text-[var(--ds-success)]",
  medium: "border-yellow-500/50 text-yellow-400",
  low: "border-red-500/50 text-red-400",
};

export function CoverageReport({ coverage }: CoverageReportProps) {
  const t = useTranslations("search");

  const maxSourceCount = Math.max(...Object.values(coverage.sources), 1);
  const maxYield = Math.max(...coverage.yield_curve, 1);

  // Strategy checklist: mark rounds where new_papers > 0
  const strategies = coverage.rounds.map((r) => ({
    name: r.strategy,
    round: r.round,
    hasResults: r.new_papers > 0,
    stopped: r.stopped,
  }));

  return (
    <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle
            className="flex items-center gap-2 text-base text-[var(--ds-text-heading)]"
            style={{ fontFamily: "var(--font-heading)" }}
          >
            <BarChart3 className="size-4 text-[var(--ds-primary)]" />
            {t("coverage.title")}
          </CardTitle>
          <Badge
            variant="outline"
            className={
              confidenceColors[coverage.confidence] ||
              "border-[var(--ds-border)] text-[var(--ds-text-secondary)]"
            }
          >
            <Shield className="mr-1 size-3" />
            {t("coverage.confidence")}: {coverage.confidence}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* Summary Bar */}
        <div className="flex gap-6">
          <div>
            <p className="text-2xl font-bold text-[var(--ds-text-heading)]">
              {coverage.total_unique}
            </p>
            <p
              className="text-xs text-[var(--ds-text-secondary)]"
              style={{ fontFamily: "var(--font-body)" }}
            >
              {t("coverage.uniquePapers")}
            </p>
          </div>
          <div>
            <p className="text-2xl font-bold text-[var(--ds-text-secondary)]">
              {coverage.total_scanned}
            </p>
            <p
              className="text-xs text-[var(--ds-text-secondary)]"
              style={{ fontFamily: "var(--font-body)" }}
            >
              {t("coverage.totalScanned")}
            </p>
          </div>
          <div>
            <p className="text-2xl font-bold text-[var(--ds-primary)]">
              {coverage.rounds.length}
            </p>
            <p
              className="text-xs text-[var(--ds-text-secondary)]"
              style={{ fontFamily: "var(--font-body)" }}
            >
              {t("coverage.rounds")}
            </p>
          </div>
        </div>

        <Separator className="bg-[var(--ds-border)]" />

        {/* Source Breakdown */}
        <div className="space-y-3">
          <h4
            className="text-xs font-medium text-[var(--ds-text-secondary)]"
            style={{ fontFamily: "var(--font-heading)" }}
          >
            {t("coverage.sourceBreakdown")}
          </h4>
          <div className="space-y-2">
            {Object.entries(coverage.sources).map(([source, count]) => (
              <div key={source} className="space-y-1">
                <div className="flex items-center justify-between">
                  <span
                    className="text-xs text-[var(--ds-text-body)]"
                    style={{ fontFamily: "var(--font-body)" }}
                  >
                    {source}
                  </span>
                  <span className="text-xs text-[var(--ds-text-secondary)]">{count}</span>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-[var(--ds-border)]">
                  <div
                    className="h-full rounded-full bg-[var(--ds-primary)] transition-all duration-300"
                    style={{
                      width: `${(count / maxSourceCount) * 100}%`,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        <Separator className="bg-[var(--ds-border)]" />

        {/* Yield Curve */}
        {coverage.yield_curve.length > 0 && (
          <div className="space-y-3">
            <h4
              className="flex items-center gap-1.5 text-xs font-medium text-[var(--ds-text-secondary)]"
              style={{ fontFamily: "var(--font-heading)" }}
            >
              <TrendingDown className="size-3" />
              {t("coverage.yieldCurve")}
            </h4>
            <div className="flex items-end gap-1 h-20">
              {coverage.yield_curve.map((value, i) => (
                <div
                  key={i}
                  className="flex-1 rounded-t bg-[var(--ds-primary)]/70 transition-all duration-300 hover:bg-[var(--ds-primary)]"
                  style={{
                    height: `${(value / maxYield) * 100}%`,
                    minHeight: value > 0 ? "4px" : "0px",
                  }}
                  title={`R${i + 1}: ${value} ${t("coverage.newPapers")}`}
                />
              ))}
            </div>
            <div className="flex gap-1">
              {coverage.yield_curve.map((_, i) => (
                <span
                  key={i}
                  className="flex-1 text-center text-[10px] text-[var(--ds-text-muted)]"
                >
                  R{i + 1}
                </span>
              ))}
            </div>
          </div>
        )}

        <Separator className="bg-[var(--ds-border)]" />

        {/* Strategy Checklist */}
        <div className="space-y-3">
          <h4
            className="text-xs font-medium text-[var(--ds-text-secondary)]"
            style={{ fontFamily: "var(--font-heading)" }}
          >
            {t("coverage.strategyChecklist")}
          </h4>
          <div className="space-y-1.5">
            {strategies.map((s) => (
              <div
                key={s.round}
                className="flex items-center gap-2 rounded-md bg-[var(--ds-bg-page)] px-3 py-1.5"
              >
                <CheckCircle2
                  className={`size-3.5 ${
                    s.hasResults ? "text-[var(--ds-success)]" : "text-[var(--ds-text-muted)]"
                  }`}
                />
                <span
                  className="text-xs text-[var(--ds-text-body)]"
                  style={{ fontFamily: "var(--font-body)" }}
                >
                  R{s.round}: {s.name}
                </span>
                <span className="text-xs text-[var(--ds-text-muted)]">
                  {s.stopped
                    ? t("coverage.stopped")
                    : `+${coverage.rounds[s.round - 1]?.new_papers ?? 0}`}
                </span>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
