"use client";

import { ShieldAlert } from "lucide-react";
import { useTranslations } from "next-intl";

import { useAuthStore } from "@/lib/auth";
import { ApiError } from "@/lib/api";
import { useAdminUsers, useAdminStats } from "@/hooks/use-admin";
import { AdminStats } from "@/components/admin-stats";
import { AdminUsersTable } from "@/components/admin-users-table";

export default function AdminPage() {
  const t = useTranslations("admin");
  const user = useAuthStore((s) => s.user);

  const {
    data: users,
    isLoading: usersLoading,
    error: usersError,
  } = useAdminUsers();

  const {
    data: stats,
    isLoading: statsLoading,
  } = useAdminStats();

  // If the admin endpoints return 403, user is not an admin
  const isForbidden =
    usersError instanceof ApiError && usersError.status === 403;

  if (isForbidden) {
    return (
      <div className="flex min-h-[60vh] flex-col items-center justify-center gap-4 p-6">
        <ShieldAlert className="size-12 text-[var(--ds-text-secondary)]" />
        <p
          className="text-lg text-[var(--ds-text-secondary)]"
          style={{ fontFamily: "var(--font-body)" }}
        >
          {t("accessDenied")}
        </p>
      </div>
    );
  }

  return (
    <div className="p-6 md:p-8">
      <div className="mb-8">
        <h1
          className="text-2xl font-bold text-[var(--ds-text-heading)]"
          style={{ fontFamily: "var(--font-heading)" }}
        >
          {t("title")}
        </h1>
      </div>

      {/* System Statistics */}
      <div className="mb-8">
        <h2
          className="mb-4 text-lg font-semibold text-[var(--ds-text-heading)]"
          style={{ fontFamily: "var(--font-heading)" }}
        >
          {t("stats")}
        </h2>
        <AdminStats stats={stats} isLoading={statsLoading} />
      </div>

      {/* Users Table */}
      <div>
        <h2
          className="mb-4 text-lg font-semibold text-[var(--ds-text-heading)]"
          style={{ fontFamily: "var(--font-heading)" }}
        >
          {t("users")}
        </h2>
        <AdminUsersTable
          users={users}
          isLoading={usersLoading}
          currentUserId={user?.id}
        />
      </div>
    </div>
  );
}
