"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { Loader2, UserMinus, UserPlus } from "lucide-react";

import {
  type Team,
  useAddMember,
  useRemoveMember,
} from "@/hooks/use-teams";
import { Badge } from "@/components/ui/badge";
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
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

const roleBadgeColors: Record<string, string> = {
  owner: "bg-[var(--ds-primary-light)] text-[var(--ds-primary)]",
  editor: "bg-[var(--ds-accent)]/15 text-[var(--ds-accent)]",
  viewer: "bg-[var(--ds-text-secondary)]/15 text-[var(--ds-text-secondary)]",
};

interface TeamMembersProps {
  team: Team;
  isOwner: boolean;
}

export function TeamMembers({ team, isOwner }: TeamMembersProps) {
  const t = useTranslations("team");
  const tCommon = useTranslations("common");

  const addMember = useAddMember();
  const removeMember = useRemoveMember();

  const [addOpen, setAddOpen] = useState(false);
  const [email, setEmail] = useState("");
  const [role, setRole] = useState("editor");
  const [confirmUserId, setConfirmUserId] = useState<string | null>(null);

  async function handleAdd(e: React.FormEvent) {
    e.preventDefault();
    try {
      await addMember.mutateAsync({ teamId: team.id, email, role });
      setEmail("");
      setRole("editor");
      setAddOpen(false);
    } catch {
      // Error handled by TanStack Query
    }
  }

  async function handleRemove(userId: string) {
    try {
      await removeMember.mutateAsync({ teamId: team.id, userId });
      setConfirmUserId(null);
    } catch {
      // Error handled by TanStack Query
    }
  }

  function getRoleLabel(r: string) {
    if (r === "owner") return t("owner");
    if (r === "editor") return t("editor");
    return t("viewer");
  }

  return (
    <div className="space-y-3">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <h4
          className="text-sm font-medium text-[var(--ds-text-heading)]"
          style={{ fontFamily: "var(--font-heading)" }}
        >
          {t("members")} ({team.members.length})
        </h4>

        {isOwner && (
          <Dialog open={addOpen} onOpenChange={setAddOpen}>
            <DialogTrigger
              render={
                <Button
                  size="sm"
                  className="bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
                />
              }
            >
              <UserPlus className="mr-1.5 size-3.5" />
              <span style={{ fontFamily: "var(--font-body)" }}>
                {t("addMember")}
              </span>
            </DialogTrigger>
            <DialogContent className="border-[var(--ds-border)] bg-[var(--ds-bg-card)] sm:max-w-sm">
              <DialogHeader>
                <DialogTitle
                  className="text-[var(--ds-text-heading)]"
                  style={{ fontFamily: "var(--font-heading)" }}
                >
                  {t("addMember")}
                </DialogTitle>
                <DialogDescription className="text-[var(--ds-text-secondary)]">
                  {t("email")}
                </DialogDescription>
              </DialogHeader>
              <form onSubmit={handleAdd} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="member-email" className="text-[var(--ds-text-heading)]">
                    {t("email")}
                  </Label>
                  <Input
                    id="member-email"
                    type="email"
                    placeholder="colleague@example.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)]"
                    style={{ fontFamily: "var(--font-body)" }}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="member-role" className="text-[var(--ds-text-heading)]">
                    {t("role")}
                  </Label>
                  <select
                    id="member-role"
                    value={role}
                    onChange={(e) => setRole(e.target.value)}
                    className="flex h-8 w-full rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-page)] px-3 text-sm text-[var(--ds-text-heading)] outline-none focus:border-[var(--ds-primary)] focus:ring-1 focus:ring-[var(--ds-primary)]"
                    style={{ fontFamily: "var(--font-body)" }}
                  >
                    <option value="editor">{t("editor")}</option>
                    <option value="viewer">{t("viewer")}</option>
                  </select>
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
                    disabled={addMember.isPending || !email.trim()}
                    className="bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
                  >
                    {addMember.isPending && (
                      <Loader2 className="mr-1.5 size-4 animate-spin" />
                    )}
                    {t("addMember")}
                  </Button>
                </DialogFooter>
              </form>
            </DialogContent>
          </Dialog>
        )}
      </div>

      {/* Member list */}
      <div className="divide-y divide-[var(--ds-border)] rounded-lg border border-[var(--ds-border)]">
        {team.members.map((member) => (
          <div
            key={member.id}
            className="flex items-center justify-between px-4 py-3"
          >
            <div className="flex flex-col gap-0.5">
              <span
                className="text-sm font-medium text-[var(--ds-text-heading)]"
                style={{ fontFamily: "var(--font-body)" }}
              >
                {member.name}
              </span>
              <span
                className="text-xs text-[var(--ds-text-secondary)]"
                style={{ fontFamily: "var(--font-body)" }}
              >
                {member.email}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Badge
                className={roleBadgeColors[member.role] ?? roleBadgeColors.viewer}
              >
                {getRoleLabel(member.role)}
              </Badge>
              {isOwner && member.role !== "owner" && (
                <Button
                  variant="ghost"
                  size="icon-sm"
                  className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-error)]"
                  onClick={() => setConfirmUserId(member.user_id)}
                  title={t("removeMember")}
                >
                  <UserMinus className="size-4" />
                </Button>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Remove member confirmation */}
      <Dialog
        open={confirmUserId !== null}
        onOpenChange={(open) => {
          if (!open) setConfirmUserId(null);
        }}
      >
        <DialogContent className="border-[var(--ds-border)] bg-[var(--ds-bg-card)] sm:max-w-sm">
          <DialogHeader>
            <DialogTitle
              className="text-[var(--ds-text-heading)]"
              style={{ fontFamily: "var(--font-heading)" }}
            >
              {t("removeMember")}
            </DialogTitle>
            <DialogDescription className="text-[var(--ds-text-secondary)]">
              {t("confirmRemove")}
            </DialogDescription>
          </DialogHeader>
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
              variant="destructive"
              disabled={removeMember.isPending}
              onClick={() => confirmUserId && handleRemove(confirmUserId)}
            >
              {removeMember.isPending && (
                <Loader2 className="mr-1.5 size-4 animate-spin" />
              )}
              {t("removeMember")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
