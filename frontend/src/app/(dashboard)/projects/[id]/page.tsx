"use client";

import { useState, useEffect } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import dynamic from "next/dynamic";
import {
  ArrowLeft,
  Trash2,
  Loader2,
} from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { toast } from "sonner";

import {
  useProject,
  useUpdateProject,
  useDeleteProject,
} from "@/hooks/use-projects";
import { fetchJournalGuidelines } from "@/lib/journal-guidelines";
import { PapersTab } from "@/components/papers-tab";
import { SmartSearchPanel } from "@/components/smart-search-panel";
import { PaperChat } from "@/components/paper-chat";
import { AnalysisTab } from "@/components/analysis-tab";
import { ProjectDashboard } from "@/components/project-dashboard";
import { ArticleUploadModal } from "@/components/article-upload-modal";
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
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";

const citationStyles = [
  "APA",
  "MLA",
  "Chicago",
  "Harvard",
  "IEEE",
  "Vancouver",
];

/** Shared fallback while a lazily-loaded tab chunk is fetched. */
function TabLoading() {
  return (
    <div className="flex items-center justify-center py-16">
      <Loader2 className="size-6 animate-spin text-[var(--ds-text-secondary)]" />
    </div>
  );
}

// These four tabs own the heavyweight dependencies (d3-force/d3-zoom for the citation graph,
// ProseMirror/Tiptap for the editor, the analysis views). They are client-only anyway — every
// one of them is a "use client" component that reads from TanStack Query — so ssr:false costs
// nothing and keeps ~246KB gzip out of the initial bundle for users who never open them.
const GraphTab = dynamic(
  () => import("@/components/graph-tab").then((m) => m.GraphTab),
  { ssr: false, loading: () => <TabLoading /> },
);
const DraftsTab = dynamic(
  () => import("@/components/drafts-tab").then((m) => m.DraftsTab),
  { ssr: false, loading: () => <TabLoading /> },
);
const DataAnalysisTab = dynamic(
  () => import("@/components/data-analysis-tab").then((m) => m.DataAnalysisTab),
  { ssr: false, loading: () => <TabLoading /> },
);
const ResearchDesignTab = dynamic(
  () => import("@/components/research-design-tab").then((m) => m.ResearchDesignTab),
  { ssr: false, loading: () => <TabLoading /> },
);

export default function ProjectDetailPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;

  const queryClient = useQueryClient();
  const { data: project, isLoading } = useProject(id);
  const updateProject = useUpdateProject();
  const deleteProject = useDeleteProject();
  const tDetail = useTranslations("projectDetail");
  const tProjects = useTranslations("projects");
  const tCommon = useTranslations("common");
  const tDash = useTranslations("dashboard");

  const searchParams = useSearchParams();
  const view = searchParams.get("view") ?? "dashboard";
  const [uploadOpen, setUploadOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [editTitle, setEditTitle] = useState("");
  const [editDescription, setEditDescription] = useState("");
  const [editJournal, setEditJournal] = useState("");
  const [editCitation, setEditCitation] = useState("");
  const [settingsDirty, setSettingsDirty] = useState(false);
  const [isFetchingGuidelines, setIsFetchingGuidelines] = useState(false);
  const [guidelinesInput, setGuidelinesInput] = useState("");

  async function handleFetchGuidelines() {
    setIsFetchingGuidelines(true);
    try {
      await fetchJournalGuidelines(id, guidelinesInput);
      // Only the project row changed (target_journal_guidelines).
      queryClient.invalidateQueries({ queryKey: ["projects", id], exact: true });
      toast.success(tDetail("guidelinesExtracted"));
      setGuidelinesInput("");
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : tDetail("guidelinesExtractFailed");
      toast.error(message);
    } finally {
      setIsFetchingGuidelines(false);
    }
  }

  // Populate settings form when project loads
  function initSettings() {
    if (project) {
      setEditTitle(project.title);
      setEditDescription(project.description ?? "");
      setEditJournal(project.target_journal ?? "");
      setEditCitation(project.citation_style);
      setSettingsDirty(false);
    }
  }

  function handleNavigate(target: string) {
    router.push(`/projects/${id}?view=${target}`);
  }

  // Auto-open upload modal when ?upload=true is in the URL
  useEffect(() => {
    if (searchParams.get("upload") === "true") {
      setUploadOpen(true);
    }
  }, [searchParams]);

  // Init settings form when navigating to settings view
  useEffect(() => {
    if (view === "settings" && project) {
      initSettings();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, project]);

  async function handleSaveSettings() {
    try {
      await updateProject.mutateAsync({
        id,
        title: editTitle,
        description: editDescription || undefined,
        target_journal: editJournal || undefined,
        citation_style: editCitation,
      });
      setSettingsDirty(false);
    } catch {
      // Error handled by TanStack Query
    }
  }

  async function handleDelete() {
    try {
      await deleteProject.mutateAsync(id);
      router.push("/projects");
    } catch {
      // Error handled by TanStack Query
    }
  }

  if (isLoading) {
    return (
      <div className="p-6 md:p-8">
        <Skeleton className="mb-6 h-8 w-24 bg-[var(--ds-bg-card)]" />
        <Skeleton className="mb-2 h-8 w-64 bg-[var(--ds-bg-card)]" />
        <Skeleton className="mb-8 h-4 w-96 bg-[var(--ds-bg-card)]" />
        <div className="flex gap-2">
          {[1, 2, 3, 4, 5].map((i) => (
            <Skeleton key={i} className="h-9 w-28 bg-[var(--ds-bg-card)]" />
          ))}
        </div>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="flex flex-col items-center justify-center p-16">
        <h2
          className="text-lg font-medium text-[var(--ds-text-heading)]"
          style={{ fontFamily: "var(--font-heading)" }}
        >
          {tProjects("projectNotFound")}
        </h2>
        <Button
          variant="ghost"
          className="mt-4 text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
          onClick={() => router.push("/projects")}
        >
          <ArrowLeft className="mr-1.5 size-4" />
          {tProjects("backToProjects")}
        </Button>
      </div>
    );
  }

  return (
    <div className="p-6 md:p-8">
      {/* Back Button */}
      <Button
        variant="ghost"
        size="sm"
        className="mb-6 text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
        onClick={() => router.push("/projects")}
      >
        <ArrowLeft className="mr-1.5 size-4" />
        <span style={{ fontFamily: "var(--font-body)" }}>{tProjects("backToProjects")}</span>
      </Button>

      {/* Project Header */}
      <div className="mb-6">
        <h1
          className="text-2xl font-bold text-[var(--ds-text-heading)]"
          style={{ fontFamily: "var(--font-heading)" }}
        >
          {project.title}
        </h1>
        {project.description && (
          <p
            className="mt-1 text-sm text-[var(--ds-text-secondary)]"
            style={{ fontFamily: "var(--font-body)" }}
          >
            {project.description}
          </p>
        )}
        {project.refined_topic && (
          <div className="mt-2 rounded-lg border border-[var(--ds-primary)]/30 bg-[var(--ds-primary-light)] p-3">
            <p className="text-xs font-medium text-[var(--ds-primary)]">{tDetail("refinedResearchTopic")}</p>
            <p className="mt-1 text-sm text-[var(--ds-text-heading)]" style={{ fontFamily: "var(--font-body)" }}>
              {String(project.refined_topic)}
            </p>
            {Array.isArray(project.inclusion_criteria) && (project.inclusion_criteria as string[]).length > 0 && (
              <div className="mt-2">
                <p className="text-xs text-[var(--ds-success)]">{tDetail("include")}:</p>
                {(project.inclusion_criteria as string[]).map((c, i) => (
                  <p key={i} className="text-xs text-[var(--ds-text-secondary)]">• {c}</p>
                ))}
              </div>
            )}
            {Array.isArray(project.exclusion_criteria) && (project.exclusion_criteria as string[]).length > 0 && (
              <div className="mt-1">
                <p className="text-xs text-[var(--ds-error)]">{tDetail("exclude")}:</p>
                {(project.exclusion_criteria as string[]).map((c, i) => (
                  <p key={i} className="text-xs text-[var(--ds-text-secondary)]">• {c}</p>
                ))}
              </div>
            )}
          </div>
        )}
        <div className="mt-3 flex items-center gap-2">
          <span
            className="inline-flex rounded-md bg-[var(--ds-primary-light)] px-2 py-0.5 text-xs font-medium text-[var(--ds-primary)]"
            style={{ fontFamily: "var(--font-body)" }}
          >
            {project.citation_style}
          </span>
          {project.target_journal && (
            <span
              className="inline-flex rounded-md bg-[var(--ds-success)]/15 px-2 py-0.5 text-xs font-medium text-[var(--ds-success)]"
              style={{ fontFamily: "var(--font-body)" }}
            >
              {project.target_journal}
            </span>
          )}
          <span
            className="inline-flex rounded-md bg-[var(--ds-border)] px-2 py-0.5 text-xs text-[var(--ds-text-secondary)]"
            style={{ fontFamily: "var(--font-body)" }}
          >
            {tDetail(`status.${project.status}`)}
          </span>
        </div>
      </div>

      {/* View Content — driven by ?view= search param */}
      {(() => {
        switch (view) {
          case "dashboard":
            return <ProjectDashboard projectId={id} onNavigate={handleNavigate} />;
          case "papers":
          case "smart-search":
          case "paper-library":
          case "full-texts":
          case "deep-search":
          case "seed-expansion":
          case "field-foundations":
            return (
              <div className="space-y-4">
                <SmartSearchPanel projectId={id} />
                <PapersTab projectId={id} />
              </div>
            );
          case "paper-chat":
            return <PaperChat projectId={id} />;
          case "gap-analysis":
            return <AnalysisTab projectId={id} />;
          case "citation-graph":
            return <GraphTab projectId={id} />;
          case "data-analysis":
            return <DataAnalysisTab projectId={id} />;
          case "drafts":
            return <DraftsTab projectId={id} />;
          case "research-design":
            return <ResearchDesignTab projectId={id} />;
          case "settings":
            return (
              <div className="space-y-6">
                <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
                  <CardHeader>
                    <CardTitle
                      className="text-[var(--ds-text-heading)]"
                      style={{ fontFamily: "var(--font-heading)" }}
                    >
                      {tDetail("projectSettings")}
                    </CardTitle>
                    <CardDescription
                      className="text-[var(--ds-text-secondary)]"
                      style={{ fontFamily: "var(--font-body)" }}
                    >
                      {tDetail("settingsDescription")}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="edit-title" className="text-[var(--ds-text-heading)]">
                        {tCommon("title")}
                      </Label>
                      <Input
                        id="edit-title"
                        value={editTitle}
                        onChange={(e) => {
                          setEditTitle(e.target.value);
                          setSettingsDirty(true);
                        }}
                        className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)]"
                        style={{ fontFamily: "var(--font-body)" }}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="edit-description" className="text-[var(--ds-text-heading)]">
                        {tCommon("description")}
                      </Label>
                      <Input
                        id="edit-description"
                        value={editDescription}
                        onChange={(e) => {
                          setEditDescription(e.target.value);
                          setSettingsDirty(true);
                        }}
                        className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)]"
                        style={{ fontFamily: "var(--font-body)" }}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="edit-citation" className="text-[var(--ds-text-heading)]">
                        {tProjects("citationStyle")}
                      </Label>
                      <select
                        id="edit-citation"
                        value={editCitation}
                        onChange={(e) => {
                          setEditCitation(e.target.value);
                          setSettingsDirty(true);
                        }}
                        className="flex h-8 w-full rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-page)] px-3 text-sm text-[var(--ds-text-heading)] outline-none focus:border-[var(--ds-primary)] focus:ring-1 focus:ring-[var(--ds-primary)]"
                        style={{ fontFamily: "var(--font-body)" }}
                      >
                        {citationStyles.map((style) => (
                          <option key={style} value={style}>
                            {style}
                          </option>
                        ))}
                      </select>
                      <p
                        className="text-xs text-[var(--ds-text-muted)]"
                        style={{ fontFamily: "var(--font-body)" }}
                      >
                        {tProjects("citationStyleDraftNote")}
                      </p>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="edit-journal" className="text-[var(--ds-text-heading)]">
                        {tProjects("targetJournal")}
                      </Label>
                      <Input
                        id="edit-journal"
                        value={editJournal}
                        onChange={(e) => {
                          setEditJournal(e.target.value);
                          setSettingsDirty(true);
                        }}
                        className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)]"
                        style={{ fontFamily: "var(--font-body)" }}
                      />
                    </div>

                    <div className="flex justify-end pt-2">
                      <Button
                        disabled={
                          !settingsDirty ||
                          updateProject.isPending ||
                          !editTitle.trim()
                        }
                        className="bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
                        onClick={handleSaveSettings}
                      >
                        {updateProject.isPending && (
                          <Loader2 className="mr-1.5 size-4 animate-spin" />
                        )}
                        {tCommon("save")}
                      </Button>
                    </div>
                  </CardContent>
                </Card>

                {/* Journal Guidelines */}
                <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
                  <CardHeader>
                    <CardTitle
                      className="text-base text-[var(--ds-text-heading)]"
                      style={{ fontFamily: "var(--font-heading)" }}
                    >
                      {tDetail("journalGuidelines")}
                    </CardTitle>
                    <p
                      className="text-sm text-[var(--ds-text-secondary)]"
                      style={{ fontFamily: "var(--font-body)" }}
                    >
                      {tDetail("journalGuidelinesDesc")}
                    </p>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {project.target_journal ? (
                      <>
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-[var(--ds-text-secondary)]">{tDetail("target")}: </span>
                          <span className="text-sm font-medium text-[var(--ds-text-heading)]">{project.target_journal}</span>
                        </div>
                        <div className="space-y-2">
                          <textarea
                            value={guidelinesInput}
                            onChange={(e) => setGuidelinesInput(e.target.value)}
                            placeholder={tDetail("journalGuidelinesPlaceholder")}
                            rows={6}
                            className="w-full rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-page)] p-3 text-sm text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)] outline-none focus-visible:border-[var(--ds-primary)]"
                            style={{ fontFamily: "var(--font-body)" }}
                          />
                          <Button
                            onClick={handleFetchGuidelines}
                            disabled={isFetchingGuidelines || !guidelinesInput.trim()}
                            className="bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
                          >
                            {isFetchingGuidelines ? <Loader2 className="mr-2 size-4 animate-spin" /> : null}
                            {tDetail("extractGuidelines")}
                          </Button>
                        </div>

                        {project.target_journal_guidelines && (() => {
                          const g = project.target_journal_guidelines as Record<string, string | number | string[] | null>;
                          return (
                            <div className="space-y-2 rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-page)] p-4 text-sm">
                              {g.citation_style && (
                                <div><span className="text-[var(--ds-text-muted)]">{tDetail("citationStyleLabel")}:</span>{" "}<span className="text-[var(--ds-text-heading)]">{String(g.citation_style)}</span></div>
                              )}
                              {g.word_limit_total && (
                                <div><span className="text-[var(--ds-text-muted)]">{tDetail("wordLimit")}:</span>{" "}<span className="text-[var(--ds-text-heading)]">{String(g.word_limit_total)} {tDetail("words")}</span></div>
                              )}
                              {g.word_limit_abstract && (
                                <div><span className="text-[var(--ds-text-muted)]">{tDetail("abstractLimit")}:</span>{" "}<span className="text-[var(--ds-text-heading)]">{String(g.word_limit_abstract)} {tDetail("words")}</span></div>
                              )}
                              {g.abstract_structure && (
                                <div><span className="text-[var(--ds-text-muted)]">{tDetail("abstractType")}:</span>{" "}<span className="text-[var(--ds-text-heading)]">{String(g.abstract_structure)}</span></div>
                              )}
                              {Array.isArray(g.required_sections) && g.required_sections.length > 0 && (
                                <div><span className="text-[var(--ds-text-muted)]">{tDetail("requiredSections")}:</span>{" "}<span className="text-[var(--ds-text-heading)]">{(g.required_sections as string[]).join(", ")}</span></div>
                              )}
                              {g.heading_style && (
                                <div><span className="text-[var(--ds-text-muted)]">{tDetail("headingStyle")}:</span>{" "}<span className="text-[var(--ds-text-heading)]">{String(g.heading_style)}</span></div>
                              )}
                              {g.line_spacing && (
                                <div><span className="text-[var(--ds-text-muted)]">{tDetail("lineSpacing")}:</span>{" "}<span className="text-[var(--ds-text-heading)]">{String(g.line_spacing)}</span></div>
                              )}
                              {g.font_requirements && (
                                <div><span className="text-[var(--ds-text-muted)]">{tDetail("font")}:</span>{" "}<span className="text-[var(--ds-text-heading)]">{String(g.font_requirements)}</span></div>
                              )}
                              {g.reference_format && (
                                <div><span className="text-[var(--ds-text-muted)]">{tDetail("referenceFormat")}:</span>{" "}<span className="text-[var(--ds-text-heading)]">{String(g.reference_format)}</span></div>
                              )}
                              {g.keyword_requirements && (
                                <div><span className="text-[var(--ds-text-muted)]">{tDetail("keywords")}:</span>{" "}<span className="text-[var(--ds-text-heading)]">{String(g.keyword_requirements)}</span></div>
                              )}
                              {g.figure_table_rules && (
                                <div><span className="text-[var(--ds-text-muted)]">{tDetail("figuresTables")}:</span>{" "}<span className="text-[var(--ds-text-heading)]">{String(g.figure_table_rules)}</span></div>
                              )}
                              {g.special_requirements && (
                                <div><span className="text-[var(--ds-text-muted)]">{tDetail("special")}:</span>{" "}<span className="text-[var(--ds-text-heading)]">{String(g.special_requirements)}</span></div>
                              )}
                              {g.source_url && (
                                <div>
                                  <span className="text-[var(--ds-text-muted)]">{tDetail("source")}:</span>{" "}
                                  <a href={String(g.source_url)} target="_blank" rel="noopener noreferrer" className="text-[var(--ds-primary)] hover:underline">
                                    {String(g.source_url)}
                                  </a>
                                </div>
                              )}
                            </div>
                          );
                        })()}
                      </>
                    ) : (
                      <p className="text-sm text-[var(--ds-text-muted)]">{tDetail("setJournalFirst")}</p>
                    )}
                  </CardContent>
                </Card>

                <Separator className="bg-[var(--ds-border)]" />

                {/* Danger Zone */}
                <Card className="border-[var(--ds-error)]/30 bg-[var(--ds-bg-card)]">
                  <CardHeader>
                    <CardTitle
                      className="text-[var(--ds-error)]"
                      style={{ fontFamily: "var(--font-heading)" }}
                    >
                      {tDetail("dangerZone")}
                    </CardTitle>
                    <CardDescription
                      className="text-[var(--ds-text-secondary)]"
                      style={{ fontFamily: "var(--font-body)" }}
                    >
                      {tDetail("dangerDescription")}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Button
                      variant="destructive"
                      onClick={() => setDeleteOpen(true)}
                    >
                      <Trash2 className="mr-1.5 size-4" />
                      {tDetail("deleteProject")}
                    </Button>
                  </CardContent>
                </Card>
              </div>
            );
          default:
            return <ProjectDashboard projectId={id} onNavigate={handleNavigate} />;
        }
      })()}

      {/* Upload Modal — triggered via ?upload=true */}
      <ArticleUploadModal
        projectId={id}
        open={uploadOpen}
        onClose={() => {
          setUploadOpen(false);
          // Remove the upload param from the URL without navigation
          router.replace(`/projects/${id}?view=${view}`);
        }}
      />

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <DialogContent className="border-[var(--ds-border)] bg-[var(--ds-bg-card)] sm:max-w-sm">
          <DialogHeader>
            <DialogTitle
              className="text-[var(--ds-text-heading)]"
              style={{ fontFamily: "var(--font-heading)" }}
            >
              {tProjects("deleteTitle")}
            </DialogTitle>
            <DialogDescription
              className="text-[var(--ds-text-secondary)]"
              style={{ fontFamily: "var(--font-body)" }}
            >
              {tProjects("deleteConfirm", { title: project.title })}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="border-[var(--ds-border)] bg-[var(--ds-bg-subtle)]">
            <DialogClose
              render={<Button variant="ghost" className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]" />}
            >
              {tCommon("cancel")}
            </DialogClose>
            <Button
              variant="destructive"
              disabled={deleteProject.isPending}
              onClick={handleDelete}
            >
              {deleteProject.isPending && (
                <Loader2 className="mr-1.5 size-4 animate-spin" />
              )}
              {tDetail("deleteProject")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
