"use client";

import { FileText, PenTool, Search, Download } from "lucide-react";
import { useTranslations } from "next-intl";

import type { UserAnalytics } from "@/hooks/use-analytics";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

const SOURCE_COLORS: Record<string, string> = {
  openalex: "#3B82F6",
  semantic_scholar: "#F97316",
  core: "#22C55E",
  crossref: "#A855F7",
  unpaywall: "#EC4899",
  manual: "#94A3B8",
};

function getSourceColor(source: string): string {
  return SOURCE_COLORS[source] || "#8B5CF6";
}

interface StatCardProps {
  icon: React.ReactNode;
  label: string;
  value: number;
  color: string;
}

function StatCard({ icon, label, value, color }: StatCardProps) {
  return (
    <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
      <CardContent className="flex items-center gap-4 py-4">
        <div
          className="flex size-10 items-center justify-center rounded-lg"
          style={{ backgroundColor: `${color}26` }}
        >
          <div style={{ color }}>{icon}</div>
        </div>
        <div>
          <p
            className="text-2xl font-bold text-[var(--ds-text-heading)]"
            style={{ fontFamily: "var(--font-heading)" }}
          >
            {value}
          </p>
          <p
            className="text-xs text-[var(--ds-text-secondary)]"
            style={{ fontFamily: "var(--font-body)" }}
          >
            {label}
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

interface AnalyticsChartsProps {
  data: UserAnalytics;
}

export function AnalyticsCharts({ data }: AnalyticsChartsProps) {
  const t = useTranslations("analytics");

  const maxMonthCount = Math.max(
    ...data.papers_by_month.map((m) => m.count),
    1,
  );

  const sourceEntries = Object.entries(data.papers_by_source);
  const maxSourceCount = Math.max(
    ...sourceEntries.map(([, count]) => count),
    1,
  );

  return (
    <div className="space-y-6">
      {/* Stat cards */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard
          icon={<FileText className="size-5" />}
          label={t("totalPapers")}
          value={data.total_papers}
          color="#8B5CF6"
        />
        <StatCard
          icon={<PenTool className="size-5" />}
          label={t("totalDrafts")}
          value={data.total_drafts}
          color="#22C55E"
        />
        <StatCard
          icon={<Search className="size-5" />}
          label={t("totalSearches")}
          value={data.total_searches}
          color="#3B82F6"
        />
        <StatCard
          icon={<Download className="size-5" />}
          label={t("totalExports")}
          value={data.total_exports}
          color="#F97316"
        />
      </div>

      {/* Papers by month */}
      {data.papers_by_month.length > 0 && (
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardHeader>
            <CardTitle
              className="text-sm font-medium text-[var(--ds-text-heading)]"
              style={{ fontFamily: "var(--font-heading)" }}
            >
              {t("papersByMonth")}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {data.papers_by_month.map((item) => (
              <div key={item.month} className="flex items-center gap-3">
                <span
                  className="w-20 shrink-0 text-xs text-[var(--ds-text-secondary)]"
                  style={{ fontFamily: "var(--font-body)" }}
                >
                  {item.month}
                </span>
                <div className="relative h-6 flex-1 overflow-hidden rounded bg-[var(--ds-bg-page)]">
                  <div
                    className="absolute inset-y-0 left-0 rounded bg-[var(--ds-primary)] transition-all"
                    style={{
                      width: `${(item.count / maxMonthCount) * 100}%`,
                      minWidth: item.count > 0 ? "2px" : "0",
                    }}
                  />
                </div>
                <span
                  className="w-8 shrink-0 text-right text-xs font-medium text-[var(--ds-text-heading)]"
                  style={{ fontFamily: "var(--font-body)" }}
                >
                  {item.count}
                </span>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Papers by source */}
      {sourceEntries.length > 0 && (
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardHeader>
            <CardTitle
              className="text-sm font-medium text-[var(--ds-text-heading)]"
              style={{ fontFamily: "var(--font-heading)" }}
            >
              {t("papersBySource")}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {sourceEntries.map(([source, count]) => (
              <div key={source} className="flex items-center gap-3">
                <span
                  className="w-32 shrink-0 truncate text-xs text-[var(--ds-text-secondary)]"
                  style={{ fontFamily: "var(--font-body)" }}
                >
                  {source}
                </span>
                <div className="relative h-6 flex-1 overflow-hidden rounded bg-[var(--ds-bg-page)]">
                  <div
                    className="absolute inset-y-0 left-0 rounded transition-all"
                    style={{
                      width: `${(count / maxSourceCount) * 100}%`,
                      minWidth: count > 0 ? "2px" : "0",
                      backgroundColor: getSourceColor(source),
                    }}
                  />
                </div>
                <span
                  className="w-8 shrink-0 text-right text-xs font-medium text-[var(--ds-text-heading)]"
                  style={{ fontFamily: "var(--font-body)" }}
                >
                  {count}
                </span>
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
