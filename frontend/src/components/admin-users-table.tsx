"use client";

import { useState } from "react";
import { MoreHorizontal, Shield, User, Trash2 } from "lucide-react";
import { useTranslations } from "next-intl";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  type AdminUser,
  useUpdateUserRole,
  useDeleteUser,
} from "@/hooks/use-admin";

interface AdminUsersTableProps {
  users: AdminUser[] | undefined;
  isLoading: boolean;
  currentUserId: string | undefined;
}

export function AdminUsersTable({
  users,
  isLoading,
  currentUserId,
}: AdminUsersTableProps) {
  const t = useTranslations("admin");
  const tCommon = useTranslations("common");
  const updateRole = useUpdateUserRole();
  const deleteUser = useDeleteUser();
  const [deletingId, setDeletingId] = useState<string | null>(null);

  function handleRoleChange(userId: string, newRole: string) {
    updateRole.mutate({ userId, role: newRole });
  }

  function handleDelete(userId: string) {
    if (window.confirm(t("confirmDelete"))) {
      setDeletingId(userId);
      deleteUser.mutate(userId, {
        onSettled: () => setDeletingId(null),
      });
    }
  }

  if (isLoading) {
    return (
      <div
        className="py-12 text-center text-[var(--ds-text-secondary)]"
        style={{ fontFamily: "var(--font-body)" }}
      >
        {tCommon("loading")}
      </div>
    );
  }

  if (!users || users.length === 0) {
    return (
      <div
        className="py-12 text-center text-[var(--ds-text-secondary)]"
        style={{ fontFamily: "var(--font-body)" }}
      >
        {t("noUsersFound")}
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
      <Table>
        <TableHeader>
          <TableRow className="border-[var(--ds-border)] hover:bg-transparent">
            <TableHead className="text-[var(--ds-text-secondary)]">{t("name")}</TableHead>
            <TableHead className="text-[var(--ds-text-secondary)]">{t("email")}</TableHead>
            <TableHead className="text-[var(--ds-text-secondary)]">{t("role")}</TableHead>
            <TableHead className="text-[var(--ds-text-secondary)]">{t("expertise")}</TableHead>
            <TableHead className="text-[var(--ds-text-secondary)]">{t("projects")}</TableHead>
            <TableHead className="text-[var(--ds-text-secondary)]">{t("joined")}</TableHead>
            <TableHead className="text-right text-[var(--ds-text-secondary)]">
              {t("actions")}
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {users.map((user) => {
            const isSelf = user.id === currentUserId;
            return (
              <TableRow
                key={user.id}
                className="border-[var(--ds-border)] hover:bg-[var(--ds-bg-hover)]"
              >
                <TableCell
                  className="font-medium text-[var(--ds-text-heading)]"
                  style={{ fontFamily: "var(--font-body)" }}
                >
                  {user.name}
                </TableCell>
                <TableCell
                  className="text-[var(--ds-text-secondary)]"
                  style={{ fontFamily: "var(--font-body)" }}
                >
                  {user.email}
                </TableCell>
                <TableCell>
                  <Badge
                    className={
                      user.role === "admin"
                        ? "bg-[var(--ds-primary-light)] text-[var(--ds-primary)] border-[var(--ds-primary)]/30"
                        : "bg-[var(--ds-border)] text-[var(--ds-text-secondary)] border-[var(--ds-text-secondary)]"
                    }
                  >
                    {user.role === "admin" ? (
                      <Shield className="mr-1 size-3" />
                    ) : (
                      <User className="mr-1 size-3" />
                    )}
                    {t(`role.${user.role}`)}
                  </Badge>
                </TableCell>
                <TableCell
                  className="text-[var(--ds-text-secondary)]"
                  style={{ fontFamily: "var(--font-body)" }}
                >
                  {t(`expertise.${user.expertise_level}`)}
                </TableCell>
                <TableCell className="text-[var(--ds-text-secondary)]">
                  {user.project_count}
                </TableCell>
                <TableCell
                  className="text-[var(--ds-text-secondary)]"
                  style={{ fontFamily: "var(--font-body)" }}
                >
                  {new Date(user.created_at).toLocaleDateString(undefined)}
                </TableCell>
                <TableCell className="text-right">
                  {!isSelf && (
                    <DropdownMenu>
                      <DropdownMenuTrigger
                        className="inline-flex h-8 w-8 items-center justify-center rounded-md text-[var(--ds-text-secondary)] hover:bg-[var(--ds-bg-hover)] hover:text-[var(--ds-text-heading)]"
                      >
                        <MoreHorizontal className="size-4" />
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        {user.role === "admin" ? (
                          <DropdownMenuItem
                            onClick={() =>
                              handleRoleChange(user.id, "user")
                            }
                          >
                            <User className="size-4" />
                            {t("makeUser")}
                          </DropdownMenuItem>
                        ) : (
                          <DropdownMenuItem
                            onClick={() =>
                              handleRoleChange(user.id, "admin")
                            }
                          >
                            <Shield className="size-4" />
                            {t("makeAdmin")}
                          </DropdownMenuItem>
                        )}
                        <DropdownMenuSeparator />
                        <DropdownMenuItem
                          variant="destructive"
                          onClick={() => handleDelete(user.id)}
                          disabled={deletingId === user.id}
                        >
                          <Trash2 className="size-4" />
                          {t("deleteUser")}
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  )}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
