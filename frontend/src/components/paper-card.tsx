// frontend/src/components/paper-card.tsx
"use client";

import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { ExternalLink, Plus, Trash2, BookOpen, ChevronDown, ChevronUp } from "lucide-react";
import { useTranslations } from "next-intl";
import { Button } from "@/components/ui/button";

interface PaperCardProps {
  title: string;
  authors: { name: string }[];
  year: number | null;
  journalName: string | null;
  citationCount: number | null;
  doi: string | null;
  abstract: string | null;
  sourceApi: string;
  fullTextUrl: string | null;
  onAdd?: () => void;
  onRemove?: () => void;
  isAdding?: boolean;
  isRemoving?: boolean;
  isInLibrary?: boolean;
  verificationBadge?: ReactNode;
  /** Optional line rendered under the abstract (e.g. the screener's stated reason). */
  footer?: ReactNode;
}

export function PaperCard({
  title,
  authors,
  year,
  journalName,
  citationCount,
  doi,
  abstract,
  sourceApi,
  fullTextUrl,
  onAdd,
  onRemove,
  isAdding,
  isRemoving,
  isInLibrary,
  verificationBadge,
  footer,
}: PaperCardProps) {
  const t = useTranslations("papers");
  // useId() is unique per mounted card even when the same paper appears in two lists.
  const abstractId = `${useId()}-abstract`;

  // Abstract is collapsed to two lines by default; the toggle only appears when the
  // collapsed text is actually clamped (measured), or once it has been expanded.
  const [expanded, setExpanded] = useState(false);
  const [isClamped, setIsClamped] = useState(false);
  const abstractRef = useRef<HTMLParagraphElement>(null);

  useEffect(() => {
    const el = abstractRef.current;
    if (!el || expanded) return;
    const measure = () => setIsClamped(el.scrollHeight > el.clientHeight + 1);
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [abstract, expanded]);

  const showToggle = !!abstract && (expanded || isClamped);

  const authorStr =
    authors.length > 3
      ? `${authors.slice(0, 3).map((a) => a.name).join(", ")} et al.`
      : authors.map((a) => a.name).join(", ");

  return (
    <div className="rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-card)] p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <h4
            className="text-sm font-medium leading-snug text-[var(--ds-text-heading)]"
            style={{ fontFamily: "var(--font-heading)" }}
          >
            {title}
          </h4>
          {authorStr && (
            <p
              className="mt-1 text-xs text-[var(--ds-text-secondary)]"
              style={{ fontFamily: "var(--font-body)" }}
            >
              {authorStr}
            </p>
          )}
          <div className="mt-2 flex flex-wrap items-center gap-2">
            {year && (
              <span className="rounded bg-[var(--ds-border)] px-1.5 py-0.5 text-xs text-[var(--ds-text-body)]">
                {year}
              </span>
            )}
            {journalName && (
              <span className="rounded bg-[var(--ds-primary-light)] px-1.5 py-0.5 text-xs text-[var(--ds-primary)]">
                {journalName}
              </span>
            )}
            {citationCount != null && (
              <span className="rounded bg-[rgba(5,150,105,0.08)] px-1.5 py-0.5 text-xs text-[var(--ds-success)]">
                {citationCount} citations
              </span>
            )}
            {verificationBadge}
          </div>
          {abstract && (
            <p
              id={abstractId}
              ref={abstractRef}
              className={`mt-2 text-xs leading-relaxed text-[var(--ds-text-secondary)] ${
                expanded ? "" : "line-clamp-2"
              }`}
              style={{ fontFamily: "var(--font-body)" }}
            >
              {abstract}
            </p>
          )}
          {showToggle && (
            <button
              type="button"
              onClick={() => setExpanded((v) => !v)}
              aria-expanded={expanded}
              aria-controls={abstractId}
              className="mt-1 inline-flex items-center gap-1 rounded text-xs font-medium text-[var(--ds-primary)] hover:underline focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--ds-primary)]"
            >
              {expanded ? (
                <ChevronUp className="size-3" aria-hidden="true" />
              ) : (
                <ChevronDown className="size-3" aria-hidden="true" />
              )}
              {expanded ? t("showLess") : t("showMore")}
            </button>
          )}
          {footer && (
            <div
              className="mt-2 text-xs text-[var(--ds-text-muted)]"
              style={{ fontFamily: "var(--font-body)" }}
            >
              {footer}
            </div>
          )}
        </div>
        <div className="flex shrink-0 flex-col gap-1">
          {onAdd && !isInLibrary && (
            <Button
              size="sm"
              variant="ghost"
              className="h-8 text-[var(--ds-success)] hover:bg-[rgba(5,150,105,0.08)] hover:text-[var(--ds-success)]"
              onClick={onAdd}
              disabled={isAdding}
            >
              <Plus className="mr-1 size-3.5" />
              Add
            </Button>
          )}
          {isInLibrary && (
            <span className="flex h-8 items-center px-2 text-xs text-[var(--ds-success)]">
              <BookOpen className="mr-1 size-3.5" />
              Added
            </span>
          )}
          {onRemove && (
            <Button
              size="sm"
              variant="ghost"
              className="h-8 text-[var(--ds-error)] hover:bg-red-400/15 hover:text-[var(--ds-error)]"
              onClick={onRemove}
              disabled={isRemoving}
            >
              <Trash2 className="mr-1 size-3.5" />
              Remove
            </Button>
          )}
          {doi && (
            <a
              href={`https://doi.org/${doi}`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex h-8 items-center px-2 text-xs text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
            >
              <ExternalLink className="mr-1 size-3.5" />
              DOI
            </a>
          )}
          {fullTextUrl && (
            <a
              href={fullTextUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="flex h-8 items-center px-2 text-xs text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-heading)]"
            >
              <ExternalLink className="mr-1 size-3.5" />
              PDF
            </a>
          )}
        </div>
      </div>
    </div>
  );
}
