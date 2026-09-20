"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import {
  Plus,
  Users,
  Loader2,
  ChevronDown,
  ChevronRight,
} from "lucide-react";

import {
  useTeams,
  useTeam,
  useCreateTeam,
} from "@/hooks/use-teams";
import { useAuthStore } from "@/lib/auth";
import { TeamMembers } from "@/components/team-members";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";
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
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";

/* ---------- Expandable team card ---------- */

function TeamCard({ teamId }: { teamId: string }) {
  const user = useAuthStore((s) => s.user);
  const { data: team, isLoading } = useTeam(teamId);

  if (isLoading || !team) {
    return (
      <div className="space-y-2 p-4">
        <Skeleton className="h-4 w-2/3 bg-[var(--ds-border)]" />
        <Skeleton className="h-4 w-1/2 bg-[var(--ds-border)]" />
      </div>
    );
  }

  const isOwner = team.created_by === user?.id;

  return (
    <CardContent className="pt-0">
      <TeamMembers team={team} isOwner={isOwner} />
    </CardContent>
  );
}

/* ---------- Teams page ---------- */

export default function TeamsPage() {
  const t = useTranslations("team");
  const tCommon = useTranslations("common");

  const { data: teams, isLoading } = useTeams();
  const createTeam = useCreateTeam();

  const [createOpen, setCreateOpen] = useState(false);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [formData, setFormData] = useState({ name: "", description: "" });

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    try {
      await createTeam.mutateAsync({
        name: formData.name,
        description: formData.description || undefined,
      });
      setFormData({ name: "", description: "" });
      setCreateOpen(false);
    } catch {
      // Error handled by TanStack Query
    }
  }

  return (
    <div className="p-6 md:p-8">
      {/* Header */}
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1
            className="text-2xl font-bold text-[var(--ds-text-heading)]"
            style={{ fontFamily: "var(--font-heading)" }}
          >
            {t("title")}
          </h1>
        </div>

        <Dialog open={createOpen} onOpenChange={setCreateOpen}>
          <DialogTrigger
            render={
              <Button className="bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]" />
            }
          >
            <Plus className="mr-1.5 size-4" />
            <span style={{ fontFamily: "var(--font-body)" }}>
              {t("createTeam")}
            </span>
          </DialogTrigger>
          <DialogContent className="border-[var(--ds-border)] bg-[var(--ds-bg-card)] sm:max-w-md">
            <DialogHeader>
              <DialogTitle
                className="text-[var(--ds-text-heading)]"
                style={{ fontFamily: "var(--font-heading)" }}
              >
                {t("createTeam")}
              </DialogTitle>
              <DialogDescription
                className="text-[var(--ds-text-secondary)]"
                style={{ fontFamily: "var(--font-body)" }}
              >
                {t("description")}
              </DialogDescription>
            </DialogHeader>
            <form onSubmit={handleCreate} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="team-name" className="text-[var(--ds-text-heading)]">
                  {t("teamName")}
                </Label>
                <Input
                  id="team-name"
                  placeholder={t("teamName")}
                  value={formData.name}
                  onChange={(e) =>
                    setFormData({ ...formData, name: e.target.value })
                  }
                  required
                  className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)]"
                  style={{ fontFamily: "var(--font-body)" }}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="team-desc" className="text-[var(--ds-text-heading)]">
                  {t("description")}
                </Label>
                <Input
                  id="team-desc"
                  placeholder={t("description")}
                  value={formData.description}
                  onChange={(e) =>
                    setFormData({ ...formData, description: e.target.value })
                  }
                  className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)]"
                  style={{ fontFamily: "var(--font-body)" }}
                />
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
                  type="submit"
                  disabled={createTeam.isPending || !formData.name.trim()}
                  className="bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
                >
                  {createTeam.isPending && (
                    <Loader2 className="mr-1.5 size-4 animate-spin" />
                  )}
                  {t("createTeam")}
                </Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      </div>

      {/* Loading */}
      {isLoading && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
              <CardHeader>
                <Skeleton className="h-5 w-3/4 bg-[var(--ds-border)]" />
                <Skeleton className="h-4 w-full bg-[var(--ds-border)]" />
              </CardHeader>
            </Card>
          ))}
        </div>
      )}

      {/* Empty */}
      {!isLoading && (!teams || teams.length === 0) && (
        <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-[var(--ds-border)] bg-[var(--ds-bg-card)]/50 py-16">
          <Users className="mb-4 size-12 text-[var(--ds-text-secondary)]" />
          <h3
            className="text-lg font-medium text-[var(--ds-text-heading)]"
            style={{ fontFamily: "var(--font-heading)" }}
          >
            {t("noTeams")}
          </h3>
          <Button
            className="mt-6 bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
            onClick={() => setCreateOpen(true)}
          >
            <Plus className="mr-1.5 size-4" />
            <span style={{ fontFamily: "var(--font-body)" }}>
              {t("createTeam")}
            </span>
          </Button>
        </div>
      )}

      {/* Team list */}
      {!isLoading && teams && teams.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {teams.map((team) => {
            const isExpanded = expandedId === team.id;
            return (
              <Card
                key={team.id}
                className="border-[var(--ds-border)] bg-[var(--ds-bg-card)] transition-colors hover:border-[var(--ds-primary)]/40"
              >
                <CardHeader
                  className="cursor-pointer"
                  onClick={() =>
                    setExpandedId(isExpanded ? null : team.id)
                  }
                >
                  <div className="flex items-center justify-between">
                    <CardTitle
                      className="text-[var(--ds-text-heading)]"
                      style={{ fontFamily: "var(--font-heading)" }}
                    >
                      {team.name}
                    </CardTitle>
                    {isExpanded ? (
                      <ChevronDown className="size-4 text-[var(--ds-text-secondary)]" />
                    ) : (
                      <ChevronRight className="size-4 text-[var(--ds-text-secondary)]" />
                    )}
                  </div>
                  {team.description && (
                    <CardDescription
                      className="line-clamp-2 text-[var(--ds-text-secondary)]"
                      style={{ fontFamily: "var(--font-body)" }}
                    >
                      {team.description}
                    </CardDescription>
                  )}
                  <span
                    className="text-xs text-[var(--ds-text-secondary)]"
                    style={{ fontFamily: "var(--font-body)" }}
                  >
                    {team.member_count} {t("members")}
                  </span>
                </CardHeader>

                {isExpanded && <TeamCard teamId={team.id} />}
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
