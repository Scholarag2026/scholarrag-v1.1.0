"use client";

import { Users, FolderOpen, FileText, PenTool, Activity } from "lucide-react";
import { useTranslations } from "next-intl";

import { Card, CardContent } from "@/components/ui/card";
import { type SystemStats } from "@/hooks/use-admin";

interface AdminStatsProps {
  stats: SystemStats | undefined;
  isLoading: boolean;
}

const statDefs = [
  { key: "totalUsers" as const, field: "total_users" as const, icon: Users },
  { key: "totalProjects" as const, field: "total_projects" as const, icon: FolderOpen },
  { key: "totalPapers" as const, field: "total_papers" as const, icon: FileText },
  { key: "totalDrafts" as const, field: "total_drafts" as const, icon: PenTool },
  { key: "totalJobs" as const, field: "total_jobs" as const, icon: Activity },
];

export function AdminStats({ stats, isLoading }: AdminStatsProps) {
  const t = useTranslations("admin");

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
      {statDefs.map(({ key, field, icon: Icon }) => (
        <Card key={key} className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardContent className="flex items-center gap-4">
            <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-[var(--ds-primary-light)]">
              <Icon className="size-5 text-[var(--ds-primary)]" />
            </div>
            <div>
              <p
                className="text-xs text-[var(--ds-text-secondary)]"
                style={{ fontFamily: "var(--font-body)" }}
              >
                {t(key)}
              </p>
              <p
                className="text-2xl font-bold text-[var(--ds-text-heading)]"
                style={{ fontFamily: "var(--font-heading)" }}
              >
                {isLoading ? "..." : (stats?.[field] ?? 0)}
              </p>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
