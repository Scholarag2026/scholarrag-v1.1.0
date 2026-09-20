"use client";

import { Loader2, BarChart3 } from "lucide-react";
import { useTranslations } from "next-intl";

import { useMyAnalytics } from "@/hooks/use-analytics";
import { AnalyticsCharts } from "@/components/analytics-charts";

export default function AnalyticsPage() {
  const t = useTranslations("analytics");
  const { data, isLoading, error } = useMyAnalytics();

  return (
    <div className="p-6 md:p-8">
      <div className="mb-8">
        <h1
          className="text-2xl font-bold text-[var(--ds-text-heading)]"
          style={{ fontFamily: "var(--font-heading)" }}
        >
          {t("title")}
        </h1>
        <p
          className="mt-1 text-sm text-[var(--ds-text-secondary)]"
          style={{ fontFamily: "var(--font-body)" }}
        >
          {t("myAnalytics")}
        </p>
      </div>

      <div className="max-w-4xl">
        {isLoading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="size-8 animate-spin text-[var(--ds-primary)]" />
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center py-20 text-center">
            <BarChart3 className="mb-3 size-10 text-[var(--ds-text-muted)]" />
            <p
              className="text-sm text-[var(--ds-text-secondary)]"
              style={{ fontFamily: "var(--font-body)" }}
            >
              {t("noPapers")}
            </p>
          </div>
        ) : data ? (
          <AnalyticsCharts data={data} />
        ) : null}
      </div>
    </div>
  );
}
