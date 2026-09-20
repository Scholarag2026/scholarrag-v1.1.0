"use client";

import { CheckCircle2, XCircle, AlertTriangle } from "lucide-react";
import { useTranslations } from "next-intl";
import type { ComplianceReport as ComplianceReportType } from "@/hooks/use-drafts";

interface ComplianceReportProps {
  report: ComplianceReportType;
}

const statusIcon = {
  pass: CheckCircle2,
  fail: XCircle,
  warning: AlertTriangle,
};

const statusColor = {
  pass: "text-[var(--ds-success)]",
  fail: "text-[var(--ds-error)]",
  warning: "text-[var(--ds-warning)]",
};

export function ComplianceReportView({ report }: ComplianceReportProps) {
  const t = useTranslations("compliance");

  const OverallIcon = statusIcon[report.overall_status as keyof typeof statusIcon] || AlertTriangle;
  const overallColor = statusColor[report.overall_status as keyof typeof statusColor] || "text-[var(--ds-text-secondary)]";

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <OverallIcon className={`size-5 ${overallColor}`} />
        <span
          className={`text-sm font-medium ${overallColor}`}
          style={{ fontFamily: "var(--font-heading)" }}
        >
          {t(`status.${report.overall_status}`)}
        </span>
        <span className="text-xs text-[var(--ds-text-muted)]">
          {t("wordCount", { count: report.total_word_count })}
        </span>
      </div>
      <div className="space-y-2">
        {report.checks.map((check, i) => {
          const Icon = statusIcon[check.status as keyof typeof statusIcon] || AlertTriangle;
          const color = statusColor[check.status as keyof typeof statusColor] || "text-[var(--ds-text-secondary)]";
          return (
            <div key={i} className="flex items-start gap-2 rounded-md bg-[var(--ds-bg-page)] px-3 py-2">
              <Icon className={`mt-0.5 size-4 shrink-0 ${color}`} />
              <span
                className="text-sm text-[var(--ds-text-body)]"
                style={{ fontFamily: "var(--font-body)" }}
              >
                {check.message}
              </span>
            </div>
          );
        })}
      </div>
      {report.sections_found.length > 0 && (
        <div className="text-xs text-[var(--ds-text-muted)]" style={{ fontFamily: "var(--font-body)" }}>
          {t("sectionsFound")}: {report.sections_found.map(s => s.charAt(0).toUpperCase() + s.slice(1)).join(", ")}
        </div>
      )}
    </div>
  );
}
