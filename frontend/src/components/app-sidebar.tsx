"use client";

import { useState, useEffect, useCallback } from "react";
import { usePathname, useRouter, useParams, useSearchParams } from "next/navigation";
import {
  FolderOpen,
  Users,
  BarChart3,
  Shield,
  Settings,
  LogOut,
  ArrowLeft,
  LayoutDashboard,
  Search,
  MessageSquare,
  FlaskConical,
  Network,
  PenTool,
  Compass,
  ChevronDown,
} from "lucide-react";
import { useTranslations } from "next-intl";

import { useAuthStore } from "@/lib/auth";
import { LanguageSwitcher } from "@/components/language-switcher";
import { useProject } from "@/hooks/use-projects";
import { useProjectPapers } from "@/hooks/use-papers";
import { useProjectDrafts } from "@/hooks/use-drafts";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarSeparator,
} from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";

// ── Mode A: Global Navigation Items ─────────────────────────────────────────

interface GlobalNavItem {
  key: string;
  href: string;
  icon: typeof FolderOpen;
  adminOnly?: boolean;
}

const globalNavItems: GlobalNavItem[] = [
  { key: "projects", href: "/projects", icon: FolderOpen },
  { key: "teams", href: "/teams", icon: Users },
  { key: "analytics", href: "/analytics", icon: BarChart3 },
  { key: "admin", href: "/admin", icon: Shield, adminOnly: true },
  { key: "settings", href: "/settings", icon: Settings },
];

// ── Mode B: Project Workspace Phase Definitions ─────────────────────────────

interface PhaseItem {
  key: string;
  view: string;
  icon: typeof Search;
  counter?: "papers" | "drafts" | "fulltexts";
}

interface Phase {
  number: number;
  labelKey: string;
  items: PhaseItem[];
}

const phases: Phase[] = [
  {
    number: 1,
    labelKey: "find",
    items: [
      { key: "papers", view: "papers", icon: Search, counter: "papers" },
    ],
  },
  {
    number: 2,
    labelKey: "explore",
    items: [
      { key: "paperChat", view: "paper-chat", icon: MessageSquare },
      { key: "researchGaps", view: "gap-analysis", icon: FlaskConical },
      { key: "citationMap", view: "citation-graph", icon: Network },
    ],
  },
  {
    number: 3,
    labelKey: "create",
    items: [
      { key: "studyDesign", view: "research-design", icon: Compass },
      { key: "dataAnalysis", view: "data-analysis", icon: BarChart3 },
      { key: "manuscript", view: "drafts", icon: PenTool, counter: "drafts" },
    ],
  },
];

// ── Helper: find which phase a view belongs to ──────────────────────────────

function findPhaseForView(view: string): number | null {
  for (const phase of phases) {
    if (phase.items.some((item) => item.view === view || item.view.startsWith(view + "&"))) {
      return phase.number;
    }
  }
  return null;
}

// ── Collapsed state persistence ─────────────────────────────────────────────

function getStorageKey(projectId: string) {
  return `sidebar-collapsed-${projectId}`;
}

function loadCollapsedState(projectId: string): Record<number, boolean> {
  if (typeof window === "undefined") return {};
  try {
    const stored = localStorage.getItem(getStorageKey(projectId));
    return stored ? JSON.parse(stored) : {};
  } catch {
    return {};
  }
}

function saveCollapsedState(projectId: string, state: Record<number, boolean>) {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(getStorageKey(projectId), JSON.stringify(state));
  } catch {
    // ignore storage errors
  }
}

// ── Component ───────────────────────────────────────────────────────────────

export function AppSidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const params = useParams();
  const searchParams = useSearchParams();
  const user = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);
  const tApp = useTranslations("app");
  const tNav = useTranslations("nav");
  const tSidebar = useTranslations("sidebar");

  // Determine mode
  const isProjectView = pathname.startsWith("/projects/") && !!params.id;
  const projectId = (params.id as string) || "";

  // Project data (only fetched in Mode B)
  const { data: project } = useProject(projectId);
  const { data: papersData } = useProjectPapers(projectId);
  const { data: draftsData } = useProjectDrafts(projectId);

  // Active view from URL
  const activeView = searchParams.get("view") || "dashboard";

  // Collapsed phases state
  const [collapsed, setCollapsed] = useState<Record<number, boolean>>({});

  // Load collapsed state from localStorage on mount / project change
  useEffect(() => {
    if (projectId) {
      setCollapsed(loadCollapsedState(projectId));
    }
  }, [projectId]);

  // Auto-expand phase when navigating to a sub-item
  useEffect(() => {
    const phaseNum = findPhaseForView(activeView);
    if (phaseNum !== null && collapsed[phaseNum]) {
      setCollapsed((prev) => {
        const next = { ...prev, [phaseNum]: false };
        if (projectId) saveCollapsedState(projectId, next);
        return next;
      });
    }
  }, [activeView, projectId]); // eslint-disable-line react-hooks/exhaustive-deps

  const togglePhase = useCallback(
    (phaseNumber: number) => {
      setCollapsed((prev) => {
        const next = { ...prev, [phaseNumber]: !prev[phaseNumber] };
        if (projectId) saveCollapsedState(projectId, next);
        return next;
      });
    },
    [projectId]
  );

  // Counters
  const paperCount = papersData?.total ?? 0;
  const draftCount = draftsData?.drafts?.length ?? 0;
  const fulltextCount = papersData?.papers
    ? papersData.papers.filter(
        (p) => p.paper?.metadata?.fulltext_status === "acquired"
      ).length
    : 0;
  const fulltextTotal = papersData?.total ?? 0;

  function getCounter(type?: "papers" | "drafts" | "fulltexts"): string | null {
    if (!type) return null;
    if (type === "papers") return String(paperCount);
    if (type === "drafts") return String(draftCount);
    if (type === "fulltexts") return `${fulltextCount}/${fulltextTotal}`;
    return null;
  }

  function handleLogout() {
    logout();
    router.push("/login");
  }

  function navigateToView(view: string) {
    router.push(`/projects/${projectId}?view=${view}`);
  }

  function isViewActive(view: string): boolean {
    // Handle "paper-library&upload=true" → matches activeView "paper-library"
    const baseView = view.split("&")[0];
    return activeView === baseView;
  }

  // ── Render: Logo ──────────────────────────────────────────────────────────

  const logo = (
    <div className="flex items-center gap-2">
      <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[var(--ds-primary)] text-white font-bold text-sm">
        D
      </div>
      <span
        className="text-lg font-bold text-[var(--ds-text-heading)]"
        style={{ fontFamily: "Georgia, serif" }}
      >
        {tApp("name")}
      </span>
    </div>
  );

  // ── Render: Footer (shared between modes) ─────────────────────────────────

  const footer = (
    <SidebarFooter className="p-4 space-y-3">
      <LanguageSwitcher />
      {user && (
        <div className="flex items-center justify-between gap-2">
          <div className="min-w-0 flex-1">
            <div className="truncate text-sm font-medium text-[var(--ds-text-heading)]">
              {user.name}
            </div>
            <div className="truncate text-xs text-[var(--ds-text-muted)]">
              {user.email}
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={handleLogout}
            className="shrink-0 text-[var(--ds-text-muted)] hover:bg-[var(--ds-bg-hover)] hover:text-red-500"
            title={tNav("signOut")}
          >
            <LogOut className="size-4" />
          </Button>
        </div>
      )}
    </SidebarFooter>
  );

  // ── Mode A: Global Navigation ─────────────────────────────────────────────

  if (!isProjectView) {
    return (
      <Sidebar className="border-r border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
        <SidebarHeader className="p-4">{logo}</SidebarHeader>

        <SidebarSeparator className="bg-[var(--ds-border)]" />

        <SidebarContent>
          <SidebarGroup>
            <SidebarGroupLabel className="text-[var(--ds-text-muted)] text-[11px] uppercase tracking-wider">
              {tNav("navigation")}
            </SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {globalNavItems.map((item) => {
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  if (item.adminOnly && (user as any)?.role !== "admin") return null;
                  const isActive = pathname.startsWith(item.href);
                  const title = tNav(item.key);
                  return (
                    <SidebarMenuItem key={item.key}>
                      <SidebarMenuButton
                        isActive={isActive}
                        tooltip={title}
                        onClick={() => router.push(item.href)}
                        className={
                          isActive
                            ? "bg-[var(--ds-primary-light)] text-[var(--ds-primary)] hover:bg-[var(--ds-primary-light)] hover:text-[var(--ds-primary)]"
                            : "text-[var(--ds-text-secondary)] hover:bg-[var(--ds-bg-hover)] hover:text-[var(--ds-text-heading)]"
                        }
                      >
                        <item.icon className="size-4" />
                        <span>{title}</span>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  );
                })}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>

        <SidebarSeparator className="bg-[var(--ds-border)]" />

        {footer}
      </Sidebar>
    );
  }

  // ── Mode B: Project Workspace ─────────────────────────────────────────────

  return (
    <Sidebar className="border-r border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
      {/* Header: Logo + Back link + Project name */}
      <SidebarHeader className="p-4 space-y-3">
        {logo}

        {/* Back to projects */}
        <button
          onClick={() => router.push("/projects")}
          className="flex items-center gap-1 text-sm text-[var(--ds-primary)] hover:underline"
        >
          <ArrowLeft className="size-3.5" />
          <span>{tNav("projects")}</span>
        </button>

        {/* Project name */}
        {project && (
          <div
            className="truncate text-sm font-semibold text-[var(--ds-text-heading)]"
            title={project.title}
          >
            {project.title}
          </div>
        )}
      </SidebarHeader>

      <SidebarSeparator className="bg-[var(--ds-border)]" />

      <SidebarContent className="overflow-y-auto">
        {/* Dashboard item */}
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton
                  isActive={activeView === "dashboard"}
                  tooltip={tSidebar("dashboard")}
                  onClick={() => navigateToView("dashboard")}
                  className={
                    activeView === "dashboard"
                      ? "bg-[var(--ds-primary-light)] text-[var(--ds-primary)] hover:bg-[var(--ds-primary-light)] hover:text-[var(--ds-primary)]"
                      : "text-[var(--ds-text-secondary)] hover:bg-[var(--ds-bg-hover)] hover:text-[var(--ds-text-heading)]"
                  }
                >
                  <LayoutDashboard className="size-4" />
                  <span>{tSidebar("dashboard")}</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* Phase groups */}
        {phases.map((phase) => {
          const isCollapsed = !!collapsed[phase.number];
          const hasActiveItem = phase.items.some((item) => isViewActive(item.view));

          return (
            <SidebarGroup key={phase.number} className="py-1">
              {/* Phase header */}
              <button
                onClick={() => togglePhase(phase.number)}
                className="flex w-full items-center gap-2 px-3 py-1.5 group"
              >
                {/* Numbered circle */}
                <span className="flex h-[18px] w-[18px] shrink-0 items-center justify-center rounded-full bg-[var(--ds-primary)] text-[10px] font-bold text-white">
                  {phase.number}
                </span>
                <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--ds-text-muted)]">
                  {tSidebar(phase.labelKey)}
                </span>
                <ChevronDown
                  className={`ml-auto size-3.5 text-[var(--ds-text-muted)] transition-transform duration-200 ${
                    isCollapsed ? "-rotate-90" : ""
                  }`}
                />
              </button>

              {/* Phase items */}
              {!isCollapsed && (
                <SidebarGroupContent className="ml-[22px] border-l-2 border-[var(--ds-border)] pl-2">
                  <SidebarMenu>
                    {phase.items.map((item) => {
                      const active = isViewActive(item.view);
                      const counter = getCounter(item.counter);
                      return (
                        <SidebarMenuItem key={item.key}>
                          <SidebarMenuButton
                            isActive={active}
                            tooltip={tSidebar(item.key)}
                            onClick={() => navigateToView(item.view)}
                            className={
                              active
                                ? "bg-[var(--ds-primary-light)] text-[var(--ds-primary)] hover:bg-[var(--ds-primary-light)] hover:text-[var(--ds-primary)]"
                                : "text-[var(--ds-text-secondary)] hover:bg-[var(--ds-bg-hover)] hover:text-[var(--ds-text-heading)]"
                            }
                          >
                            <item.icon className="size-4" />
                            <span className="flex-1 truncate">
                              {tSidebar(item.key)}
                              {counter !== null && (
                                <span className="text-[10px] font-normal" style={{ color: "var(--ds-text-muted)" }}>{` (${counter})`}</span>
                              )}
                            </span>
                          </SidebarMenuButton>
                        </SidebarMenuItem>
                      );
                    })}
                  </SidebarMenu>
                </SidebarGroupContent>
              )}

              {/* Show a subtle active indicator even when collapsed */}
              {isCollapsed && hasActiveItem && (
                <div className="ml-[22px] h-0.5 w-8 rounded-full bg-[var(--ds-primary)]" />
              )}
            </SidebarGroup>
          );
        })}

        {/* Settings at the bottom of content */}
        <SidebarGroup className="mt-auto">
          <SidebarSeparator className="bg-[var(--ds-border)]" />
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton
                  isActive={activeView === "settings"}
                  tooltip={tNav("settings")}
                  onClick={() => navigateToView("settings")}
                  className={
                    activeView === "settings"
                      ? "bg-[var(--ds-primary-light)] text-[var(--ds-primary)] hover:bg-[var(--ds-primary-light)] hover:text-[var(--ds-primary)]"
                      : "text-[var(--ds-text-secondary)] hover:bg-[var(--ds-bg-hover)] hover:text-[var(--ds-text-heading)]"
                  }
                >
                  <Settings className="size-4" />
                  <span>{tNav("settings")}</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarSeparator className="bg-[var(--ds-border)]" />

      {footer}
    </Sidebar>
  );
}
