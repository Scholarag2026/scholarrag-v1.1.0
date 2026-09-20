"use client";

import { Button } from "@/components/ui/button";
import { QualityBadge } from "@/components/quality-badge";
import type { GraphNode } from "@/hooks/use-graph";
import { Expand, Library, X } from "lucide-react";
import { useTranslations } from "next-intl";

interface GraphSidePanelProps {
  node: GraphNode | null;
  onClose: () => void;
  onExpand: (paperId: string) => void;
  onAddToLibrary: (node: GraphNode) => void;
  isExpanding: boolean;
}

export function GraphSidePanel({
  node,
  onClose,
  onExpand,
  onAddToLibrary,
  isExpanding,
}: GraphSidePanelProps) {
  const t = useTranslations("graph");

  if (!node) return null;

  return (
    <div className="absolute right-0 top-0 h-full w-80 border-l border-[var(--ds-border)] bg-[var(--ds-bg-card)]/95 backdrop-blur-sm overflow-y-auto z-10">
      <div className="p-4 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {node.quality_score != null && (
              <QualityBadge score={node.quality_score} />
            )}
            {node.in_library && (
              <span className="text-xs text-[var(--ds-success)] border border-[var(--ds-success)]/30 rounded px-1.5 py-0.5">
                {t("inLibrary")}
              </span>
            )}
          </div>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-[var(--ds-text-heading)] leading-tight">
            {node.title}
          </h3>
          {node.authors.length > 0 && (
            <p className="text-xs text-[var(--ds-text-secondary)] mt-1">
              {node.authors.join(", ")}
            </p>
          )}
          <p className="text-xs text-[var(--ds-text-muted)] mt-1">
            {node.year && `${node.year} · `}
            {node.citation_count != null &&
              `${node.citation_count} ${t("citations")}`}
          </p>
        </div>

        {node.abstract && (
          <div>
            <h4 className="text-xs font-semibold text-[var(--ds-text-secondary)] uppercase mb-1">
              {t("abstract")}
            </h4>
            <p className="text-xs text-[var(--ds-text-body)] leading-relaxed line-clamp-6">
              {node.abstract}
            </p>
          </div>
        )}

        <div className="flex flex-col gap-2">
          <Button
            variant="outline"
            size="sm"
            className="w-full gap-2"
            onClick={() => onExpand(node.id)}
            disabled={isExpanding}
          >
            <Expand className="h-3.5 w-3.5" />
            {isExpanding ? t("expanding") : t("expandCitations")}
          </Button>

          {!node.in_library && (
            <Button
              variant="default"
              size="sm"
              className="w-full gap-2"
              onClick={() => onAddToLibrary(node)}
            >
              <Library className="h-3.5 w-3.5" />
              {t("addToLibrary")}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
