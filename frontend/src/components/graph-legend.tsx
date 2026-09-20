"use client";

import { useTranslations } from "next-intl";

export type ColorMode = "year" | "cluster" | "quality";

// Viridis 10-stop palette for year encoding
export const VIRIDIS = [
  "#440154", "#482878", "#3e4989", "#31688e", "#26828e",
  "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725",
] as const;

// Tableau 10 palette for cluster encoding
export const TABLEAU_10 = [
  "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
  "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
] as const;

interface GraphLegendProps {
  colorMode: ColorMode;
  yearRange: [number, number];
  clusterCount: number;
}

export function GraphLegend({ colorMode, yearRange, clusterCount }: GraphLegendProps) {
  const t = useTranslations("graph");

  return (
    <div className="absolute bottom-3 left-3 z-20 min-w-[200px] rounded-xl border border-[var(--ds-border)] bg-[var(--ds-bg-card)]/95 p-3 text-xs backdrop-blur-sm">
      {/* Mode-specific content */}
      {colorMode === "year" && (
        <>
          <div className="mb-2 font-semibold text-[var(--ds-text-heading)]">{t("legendYear")}</div>
          <div
            className="mb-1 h-2.5 rounded-full"
            style={{
              background: `linear-gradient(to right, ${VIRIDIS.join(", ")})`,
            }}
          />
          <div className="mb-2.5 flex justify-between text-[10px] text-[var(--ds-text-secondary)]">
            <span>{yearRange[0]}</span>
            <span>{Math.round((yearRange[0] + yearRange[1]) / 2)}</span>
            <span>{yearRange[1]}</span>
          </div>
        </>
      )}

      {colorMode === "cluster" && (
        <>
          <div className="mb-2 font-semibold text-[var(--ds-text-heading)]">{t("legendCluster")}</div>
          <div className="mb-2.5 flex flex-wrap gap-1.5">
            {Array.from({ length: Math.min(clusterCount, 10) }, (_, i) => (
              <div key={i} className="flex items-center gap-1">
                <div
                  className="h-2.5 w-2.5 rounded-full"
                  style={{ background: TABLEAU_10[i % TABLEAU_10.length] }}
                />
                <span className="text-[10px] text-[var(--ds-text-secondary)]">#{i + 1}</span>
              </div>
            ))}
          </div>
        </>
      )}

      {colorMode === "quality" && (
        <>
          <div className="mb-2 font-semibold text-[var(--ds-text-heading)]">{t("legendQuality")}</div>
          <div className="mb-2.5 space-y-1">
            <div className="flex items-center gap-2">
              <div className="h-2.5 w-2.5 rounded-full bg-[var(--ds-success)]" />
              <span className="text-[var(--ds-text-secondary)]">&gt; 0.6</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-2.5 w-2.5 rounded-full bg-[var(--ds-warning)]" />
              <span className="text-[var(--ds-text-secondary)]">0.3 – 0.6</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-2.5 w-2.5 rounded-full bg-[var(--ds-error)]" />
              <span className="text-[var(--ds-text-secondary)]">&lt; 0.3</span>
            </div>
          </div>
        </>
      )}

      {/* Common items */}
      <div className="space-y-1 border-t border-[var(--ds-border)] pt-2">
        <div className="flex items-center gap-2">
          <div className="h-2.5 w-2.5 rounded-full border-2 border-[var(--ds-text-heading)] bg-[var(--ds-primary)]" />
          <span className="text-[var(--ds-text-secondary)]">{t("legendFocus")}</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-2.5 w-2.5 rounded-full border border-dashed border-[#9CA3AF]" />
          <span className="text-[var(--ds-text-secondary)]">{t("legendNotInLibrary")}</span>
        </div>
        <div className="flex items-center gap-1.5 pt-1">
          <div className="h-1.5 w-1.5 rounded-full bg-[#6B7280]" />
          <div className="h-2.5 w-2.5 rounded-full bg-[#6B7280]" />
          <div className="h-4 w-4 rounded-full bg-[#6B7280]" />
          <span className="ml-1 text-[var(--ds-text-secondary)]">{t("legendCitationCount")}</span>
        </div>
      </div>
    </div>
  );
}
