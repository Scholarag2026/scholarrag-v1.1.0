"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import {
  Plus,
  FolderOpen,
  Trash2,
  Loader2,
  Calendar,
} from "lucide-react";
import { useTranslations } from "next-intl";
import { fetchJournalGuidelines } from "@/lib/journal-guidelines";

import {
  useProjects,
  useCreateProject,
  useDeleteProject,
} from "@/hooks/use-projects";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from "@/components/ui/card";
import {
  Dialog,
  DialogTrigger,
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

const citationStyles = [
  "APA",
  "MLA",
  "Chicago",
  "Harvard",
  "IEEE",
  "Vancouver",
];

export default function ProjectsPage() {
  const router = useRouter();
  const { data, isLoading } = useProjects();
  const createProject = useCreateProject();
  const deleteProject = useDeleteProject();
  const t = useTranslations("projects");
  const tCommon = useTranslations("common");

  const [createOpen, setCreateOpen] = useState(false);
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [formData, setFormData] = useState({
    title: "",
    description: "",
    citation_style: "APA",
    target_journal: "",
    guidelines_text: "",
  });

  const projects = data?.projects ?? [];

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    try {
      const result = await createProject.mutateAsync({
        title: formData.title,
        description: formData.description || undefined,
        citation_style: formData.citation_style,
        target_journal: formData.target_journal || undefined,
      });

      // If user pasted guidelines text and has a journal, extract them
      const projectId = (result as { id: string }).id;
      if (formData.guidelines_text.trim() && formData.target_journal.trim()) {
        try {
          await fetchJournalGuidelines(projectId, formData.guidelines_text);
        } catch {
          // Guidelines extraction is best-effort — don't block project creation
        }
      }

      setFormData({
        title: "",
        description: "",
        citation_style: "APA",
        target_journal: "",
        guidelines_text: "",
      });
      setCreateOpen(false);
    } catch {
      // Error is handled by TanStack Query
    }
  }

  async function handleDelete(id: string) {
    try {
      await deleteProject.mutateAsync(id);
      setDeleteId(null);
    } catch {
      // Error is handled by TanStack Query
    }
  }

  function formatDate(dateStr: string) {
    return new Date(dateStr).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
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
          <p
            className="mt-1 text-sm text-[var(--ds-text-secondary)]"
            style={{ fontFamily: "var(--font-body)" }}
          >
            {t("subtitle")}
          </p>
        </div>

        <Dialog open={createOpen} onOpenChange={setCreateOpen}>
          <DialogTrigger
            render={
              <Button className="bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]" />
            }
          >
            <Plus className="mr-1.5 size-4" />
            <span style={{ fontFamily: "var(--font-body)" }}>{t("newProject")}</span>
          </DialogTrigger>
          <DialogContent className="border-[var(--ds-border)] bg-[var(--ds-bg-card)] sm:max-w-md">
            <DialogHeader>
              <DialogTitle
                className="text-[var(--ds-text-heading)]"
                style={{ fontFamily: "var(--font-heading)" }}
              >
                {t("createTitle")}
              </DialogTitle>
              <DialogDescription
                className="text-[var(--ds-text-secondary)]"
                style={{ fontFamily: "var(--font-body)" }}
              >
                {t("createDescription")}
              </DialogDescription>
            </DialogHeader>
            <form onSubmit={handleCreate} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="title" className="text-[var(--ds-text-heading)]">
                  {tCommon("title")}
                </Label>
                <Input
                  id="title"
                  placeholder={t("titlePlaceholder")}
                  value={formData.title}
                  onChange={(e) =>
                    setFormData({ ...formData, title: e.target.value })
                  }
                  required
                  className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)]"
                  style={{ fontFamily: "var(--font-body)" }}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="description" className="text-[var(--ds-text-heading)]">
                  {tCommon("description")}
                </Label>
                <Input
                  id="description"
                  placeholder={t("descriptionPlaceholder")}
                  value={formData.description}
                  onChange={(e) =>
                    setFormData({ ...formData, description: e.target.value })
                  }
                  className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)]"
                  style={{ fontFamily: "var(--font-body)" }}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="citation_style" className="text-[var(--ds-text-heading)]">
                  {t("citationStyle")}
                </Label>
                <select
                  id="citation_style"
                  value={formData.citation_style}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      citation_style: e.target.value,
                    })
                  }
                  className="flex h-8 w-full rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-page)] px-3 text-sm text-[var(--ds-text-heading)] outline-none focus:border-[var(--ds-primary)] focus:ring-1 focus:ring-[var(--ds-primary)]"
                  style={{ fontFamily: "var(--font-body)" }}
                >
                  {citationStyles.map((style) => (
                    <option key={style} value={style}>
                      {style}
                    </option>
                  ))}
                </select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="target_journal" className="text-[var(--ds-text-heading)]">
                  {t("targetJournal")}
                </Label>
                <Input
                  id="target_journal"
                  placeholder={t("journalPlaceholder")}
                  value={formData.target_journal}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      target_journal: e.target.value,
                    })
                  }
                  className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)]"
                  style={{ fontFamily: "var(--font-body)" }}
                />
              </div>

              {formData.target_journal.trim() && (
                <div className="space-y-2">
                  <Label htmlFor="guidelines_text" className="text-[var(--ds-text-heading)]">
                    {t("guidelinesLabel")}
                  </Label>
                  <p className="text-xs" style={{ color: "var(--ds-text-muted)" }}>
                    {t("guidelinesHint")}
                  </p>
                  <textarea
                    id="guidelines_text"
                    placeholder={t("guidelinesPlaceholder")}
                    value={formData.guidelines_text}
                    onChange={(e) =>
                      setFormData({
                        ...formData,
                        guidelines_text: e.target.value,
                      })
                    }
                    rows={4}
                    className="w-full rounded-md border px-3 py-2 text-sm border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)]"
                    style={{ fontFamily: "var(--font-body)", resize: "vertical" }}
                  />
                </div>
              )}

              <DialogFooter className="border-[var(--ds-border)] bg-[var(--ds-bg-subtle)]">
                <DialogClose
                  render={<Button variant="ghost" className="text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]" />}
                >
                  {tCommon("cancel")}
                </DialogClose>
                <Button
                  type="submit"
                  disabled={createProject.isPending || !formData.title.trim()}
                  className="bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
                >
                  {createProject.isPending && (
                    <Loader2 className="mr-1.5 size-4 animate-spin" />
                  )}
                  {t("createProject")}
                </Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
              <CardHeader>
                <Skeleton className="h-5 w-3/4 bg-[var(--ds-border)]" />
                <Skeleton className="h-4 w-full bg-[var(--ds-border)]" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-3 w-1/3 bg-[var(--ds-border)]" />
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Empty State */}
      {!isLoading && projects.length === 0 && (
        <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-[var(--ds-border)] bg-[var(--ds-bg-card)]/50 py-16">
          <FolderOpen className="mb-4 size-12 text-[var(--ds-text-secondary)]" />
          <h3
            className="text-lg font-medium text-[var(--ds-text-heading)]"
            style={{ fontFamily: "var(--font-heading)" }}
          >
            {t("noProjects")}
          </h3>
          <p
            className="mt-1 text-sm text-[var(--ds-text-secondary)]"
            style={{ fontFamily: "var(--font-body)" }}
          >
            {t("noProjectsHint")}
          </p>
          <Button
            className="mt-6 bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
            onClick={() => setCreateOpen(true)}
          >
            <Plus className="mr-1.5 size-4" />
            <span style={{ fontFamily: "var(--font-body)" }}>
              {t("newProject")}
            </span>
          </Button>
        </div>
      )}

      {/* Project Grid */}
      {!isLoading && projects.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {projects.map((project) => (
            <Card
              key={project.id}
              className="group cursor-pointer border-[var(--ds-border)] bg-[var(--ds-bg-card)] transition-colors hover:border-[var(--ds-primary)]/40"
              onClick={() => router.push(`/projects/${project.id}`)}
            >
              <CardHeader>
                <CardTitle
                  className="text-[var(--ds-text-heading)]"
                  style={{ fontFamily: "var(--font-heading)" }}
                >
                  {project.title}
                </CardTitle>
                {project.description && (
                  <CardDescription
                    className="line-clamp-2 text-[var(--ds-text-secondary)]"
                    style={{ fontFamily: "var(--font-body)" }}
                  >
                    {project.description}
                  </CardDescription>
                )}
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-2">
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
                </div>
              </CardContent>
              <CardFooter className="border-[var(--ds-border)] bg-[var(--ds-bg-subtle)]">
                <div className="flex w-full items-center justify-between">
                  <div
                    className="flex items-center gap-1.5 text-xs text-[var(--ds-text-secondary)]"
                    style={{ fontFamily: "var(--font-body)" }}
                  >
                    <Calendar className="size-3" />
                    {formatDate(project.updated_at)}
                  </div>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    className="text-[var(--ds-text-secondary)] opacity-0 transition-opacity hover:text-[var(--ds-error)] group-hover:opacity-100"
                    onClick={(e) => {
                      e.stopPropagation();
                      setDeleteId(project.id);
                    }}
                    title={t("deleteTitle")}
                  >
                    <Trash2 className="size-4" />
                  </Button>
                </div>
              </CardFooter>
            </Card>
          ))}
        </div>
      )}

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteId !== null}
        onOpenChange={(open) => {
          if (!open) setDeleteId(null);
        }}
      >
        <DialogContent className="border-[var(--ds-border)] bg-[var(--ds-bg-card)] sm:max-w-sm">
          <DialogHeader>
            <DialogTitle
              className="text-[var(--ds-text-heading)]"
              style={{ fontFamily: "var(--font-heading)" }}
            >
              {t("deleteTitle")}
            </DialogTitle>
            <DialogDescription
              className="text-[var(--ds-text-secondary)]"
              style={{ fontFamily: "var(--font-body)" }}
            >
              {t("deleteConfirmShort")}
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
              onClick={() => deleteId && handleDelete(deleteId)}
            >
              {deleteProject.isPending && (
                <Loader2 className="mr-1.5 size-4 animate-spin" />
              )}
              {tCommon("delete")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
