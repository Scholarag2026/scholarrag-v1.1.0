"use client";

import { useTranslations } from "next-intl";
import {
  AlertCircle,
  Calendar,
  CheckCircle2,
  Database,
  Flag,
  ListChecks,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { DataCollectionPlan } from "@/hooks/use-research-design";

interface CollectionPlanViewProps {
  plan: DataCollectionPlan;
}

export function CollectionPlanView({ plan }: CollectionPlanViewProps) {
  const t = useTranslations("design");

  return (
    <div className="space-y-4">
      {/* Collection Phases */}
      {plan.phases.length > 0 && (
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm text-[var(--ds-text-heading)]">
              <ListChecks className="h-4 w-4 text-[var(--ds-primary)]" />
              {t("phases")} ({plan.phases.length})
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {plan.phases.map((phase) => (
              <div
                key={phase.phase_number}
                className="rounded-lg border border-[var(--ds-border)] p-3 space-y-2"
              >
                <div className="flex items-center gap-2">
                  <Badge
                    variant="outline"
                    className="border-[var(--ds-primary)]/50 text-[var(--ds-primary)]"
                  >
                    {phase.phase_number}
                  </Badge>
                  <span className="text-sm font-medium text-[var(--ds-text-heading)]">
                    {phase.name}
                  </span>
                  <span className="ml-auto text-xs text-[var(--ds-text-secondary)]">
                    {t("duration")}: {phase.duration}
                  </span>
                </div>
                <p className="text-sm text-[var(--ds-text-heading)]">{phase.description}</p>
                {phase.steps.length > 0 && (
                  <div>
                    <span className="text-xs text-[var(--ds-text-secondary)]">{t("steps")}:</span>
                    <ol className="mt-1 list-decimal list-inside space-y-0.5">
                      {phase.steps.map((step, j) => (
                        <li key={j} className="text-xs text-[var(--ds-text-heading)]">
                          {step}
                        </li>
                      ))}
                    </ol>
                  </div>
                )}
                {phase.deliverables.length > 0 && (
                  <div>
                    <span className="text-xs text-[var(--ds-text-secondary)]">
                      {t("deliverables")}:
                    </span>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {phase.deliverables.map((d, j) => (
                        <Badge
                          key={j}
                          variant="secondary"
                          className="text-xs"
                        >
                          {d}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Timeline */}
      {plan.timeline.length > 0 && (
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm text-[var(--ds-text-heading)]">
              <Calendar className="h-4 w-4 text-[var(--ds-primary)]" />
              {t("timeline")}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {plan.timeline.map((item, i) => (
                <div
                  key={i}
                  className="flex items-start gap-3 rounded-md border border-[var(--ds-border)]/50 p-2"
                >
                  <span className="shrink-0 text-xs font-medium text-[var(--ds-primary)] min-w-[60px]">
                    {item.week}
                  </span>
                  <span className="text-sm text-[var(--ds-text-heading)] flex-1">
                    {item.activity}
                  </span>
                  {item.milestone && (
                    <Badge
                      variant="outline"
                      className="shrink-0 border-emerald-500/50 text-emerald-400 gap-1"
                    >
                      <Flag className="h-3 w-3" />
                      {item.milestone}
                    </Badge>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Quality Checks */}
      {plan.quality_checks.length > 0 && (
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm text-[var(--ds-text-heading)]">
              <CheckCircle2 className="h-4 w-4 text-[var(--ds-primary)]" />
              {t("qualityChecks")} ({plan.quality_checks.length})
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {plan.quality_checks.map((check, i) => (
              <div
                key={i}
                className="rounded-lg border border-[var(--ds-border)] p-3 space-y-1"
              >
                <p className="text-sm font-medium text-[var(--ds-text-heading)]">
                  {check.check_name}
                </p>
                <p className="text-xs text-[var(--ds-text-secondary)]">
                  <span className="text-[var(--ds-text-secondary)]">{t("week")}:</span>{" "}
                  {check.when}
                </p>
                <p className="text-xs text-[var(--ds-text-heading)]">{check.how}</p>
                <p className="text-xs text-amber-400/80">
                  {check.action_if_failed}
                </p>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Data Storage */}
      <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm text-[var(--ds-text-heading)]">
            <Database className="h-4 w-4 text-[var(--ds-primary)]" />
            {t("dataStorage")}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            <div>
              <span className="text-xs text-[var(--ds-text-secondary)]">{t("storageMethod")}:</span>
              <p className="text-sm text-[var(--ds-text-heading)]">
                {plan.data_storage.storage_method}
              </p>
            </div>
            <div>
              <span className="text-xs text-[var(--ds-text-secondary)]">{t("backupStrategy")}:</span>
              <p className="text-sm text-[var(--ds-text-heading)]">
                {plan.data_storage.backup_strategy}
              </p>
            </div>
            <div>
              <span className="text-xs text-[var(--ds-text-secondary)]">{t("accessControl")}:</span>
              <p className="text-sm text-[var(--ds-text-heading)]">
                {plan.data_storage.access_control}
              </p>
            </div>
            <div>
              <span className="text-xs text-[var(--ds-text-secondary)]">{t("anonymization")}:</span>
              <p className="text-sm text-[var(--ds-text-heading)]">
                {plan.data_storage.anonymization}
              </p>
            </div>
          </div>
          <div>
            <span className="text-xs text-[var(--ds-text-secondary)]">{t("retentionPeriod")}:</span>
            <p className="text-sm text-[var(--ds-text-heading)]">
              {plan.data_storage.retention_period}
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Ethical Reminders */}
      {plan.ethical_reminders.length > 0 && (
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm text-[var(--ds-text-heading)]">
              <AlertCircle className="h-4 w-4 text-amber-400" />
              {t("ethicalReminders")}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="list-disc list-inside space-y-1">
              {plan.ethical_reminders.map((reminder, i) => (
                <li key={i} className="text-sm text-[var(--ds-text-heading)]">
                  {reminder}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
