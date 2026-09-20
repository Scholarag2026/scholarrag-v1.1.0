"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

import { useAuthStore } from "@/lib/auth";
import { AppSidebar } from "@/components/app-sidebar";
import {
  SidebarProvider,
  SidebarInset,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { Skeleton } from "@/components/ui/skeleton";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const user = useAuthStore((s) => s.user);
  const isLoading = useAuthStore((s) => s.isLoading);

  useEffect(() => {
    if (!isLoading && !user) {
      router.replace("/login");
    }
  }, [user, isLoading, router]);

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[var(--ds-bg-page)]">
        <div className="flex flex-col items-center gap-4">
          <Skeleton className="h-8 w-48 bg-[var(--ds-bg-subtle)]" />
          <Skeleton className="h-4 w-32 bg-[var(--ds-bg-subtle)]" />
        </div>
      </div>
    );
  }

  if (!user) {
    return null;
  }

  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset className="bg-[var(--ds-bg-page)]">
        <header className="flex h-12 items-center gap-2 border-b border-[var(--ds-border)] px-4 md:hidden">
          <SidebarTrigger className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]" />
        </header>
        <div className="flex-1 overflow-auto">{children}</div>
      </SidebarInset>
    </SidebarProvider>
  );
}
