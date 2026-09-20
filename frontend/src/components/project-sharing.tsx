"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { Loader2, Share2 } from "lucide-react";

import { useTeams, useShareProject } from "@/hooks/use-teams";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

interface ProjectSharingProps {
  projectId: string;
}

export function ProjectSharing({ projectId }: ProjectSharingProps) {
  const t = useTranslations("team");
  const tCommon = useTranslations("common");

  const { data: teams, isLoading } = useTeams();
  const shareProject = useShareProject();

  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());

  function toggleTeam(teamId: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(teamId)) {
        next.delete(teamId);
      } else {
        next.add(teamId);
      }
      return next;
    });
  }

  async function handleShare() {
    try {
      for (const teamId of selected) {
        await shareProject.mutateAsync({ teamId, projectId });
      }
      setSelected(new Set());
      setOpen(false);
    } catch {
      // Error handled by TanStack Query
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger
        render={
          <Button
            variant="ghost"
            size="sm"
            className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
          />
        }
      >
        <Share2 className="mr-1.5 size-3.5" />
        <span style={{ fontFamily: "var(--font-body)" }}>
          {t("shareProject")}
        </span>
      </DialogTrigger>
      <DialogContent className="border-[var(--ds-border)] bg-[var(--ds-bg-card)] sm:max-w-sm">
        <DialogHeader>
          <DialogTitle
            className="text-[var(--ds-text-heading)]"
            style={{ fontFamily: "var(--font-heading)" }}
          >
            {t("shareProject")}
          </DialogTitle>
          <DialogDescription className="text-[var(--ds-text-secondary)]">
            {t("sharedWith")}
          </DialogDescription>
        </DialogHeader>

        <div className="max-h-60 space-y-2 overflow-y-auto">
          {isLoading && (
            <p className="py-4 text-center text-sm text-[var(--ds-text-secondary)]">
              {tCommon("loading")}
            </p>
          )}
          {!isLoading && (!teams || teams.length === 0) && (
            <p className="py-4 text-center text-sm text-[var(--ds-text-secondary)]">
              {t("noTeams")}
            </p>
          )}
          {teams?.map((team) => (
            <label
              key={team.id}
              className="flex cursor-pointer items-center gap-3 rounded-lg border border-[var(--ds-border)] px-4 py-3 transition-colors hover:border-[var(--ds-primary)]/40"
            >
              <input
                type="checkbox"
                checked={selected.has(team.id)}
                onChange={() => toggleTeam(team.id)}
                className="size-4 rounded border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-primary)] accent-[var(--ds-primary)]"
              />
              <div className="flex flex-col">
                <span
                  className="text-sm font-medium text-[var(--ds-text-heading)]"
                  style={{ fontFamily: "var(--font-body)" }}
                >
                  {team.name}
                </span>
                <span
                  className="text-xs text-[var(--ds-text-secondary)]"
                  style={{ fontFamily: "var(--font-body)" }}
                >
                  {team.member_count} {t("members")}
                </span>
              </div>
            </label>
          ))}
        </div>

        <DialogFooter className="border-[var(--ds-border)] bg-[var(--ds-bg-subtle)]">
          <DialogClose
            render={
              <Button
                variant="ghost"
                className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
              />
            }
          >
            {tCommon("cancel")}
          </DialogClose>
          <Button
            disabled={selected.size === 0 || shareProject.isPending}
            onClick={handleShare}
            className="bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
          >
            {shareProject.isPending && (
              <Loader2 className="mr-1.5 size-4 animate-spin" />
            )}
            {t("shareProject")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
