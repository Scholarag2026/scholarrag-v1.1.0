"use client";

import { Badge } from "@/components/ui/badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { QualityBadge } from "@/components/quality-badge";
import type { PaperAnalysis } from "@/hooks/use-analysis";
import { ChevronDown } from "lucide-react";
import { useTranslations } from "next-intl";
import { useState } from "react";

interface PaperAnalysisDetailProps {
  analysis: PaperAnalysis;
}

export function PaperAnalysisDetail({ analysis }: PaperAnalysisDetailProps) {
  const t = useTranslations("analysis");
  const [open, setOpen] = useState(false);

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="flex w-full items-center justify-between rounded-lg border border-slate-700 bg-slate-800/50 p-3 text-left hover:bg-slate-800">
        <div className="flex items-center gap-3">
          <QualityBadge score={analysis.quality_score} />
          <div>
            <p className="text-sm font-medium text-slate-50">
              {analysis.paper_title}
            </p>
            <p className="text-xs text-slate-400">
              {analysis.paper_year} &middot; {analysis.methodology_rigor} {t("rigor")}
            </p>
          </div>
        </div>
        <ChevronDown
          className={`h-4 w-4 text-slate-400 transition-transform ${open ? "rotate-180" : ""}`}
        />
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-1 rounded-lg border border-slate-700 bg-slate-800/30 p-4 space-y-3">
        <div>
          <h4 className="text-xs font-semibold text-slate-400 uppercase mb-1">
            {t("keyFindings")}
          </h4>
          <ul className="list-disc list-inside text-sm text-slate-300 space-y-1">
            {analysis.key_findings.map((f, i) => (
              <li key={i}>{f}</li>
            ))}
          </ul>
        </div>

        <div>
          <h4 className="text-xs font-semibold text-slate-400 uppercase mb-1">
            {t("methodology")}
          </h4>
          <p className="text-sm text-slate-300">{analysis.methodology}</p>
        </div>

        {analysis.limitations.length > 0 && (
          <div>
            <h4 className="text-xs font-semibold text-slate-400 uppercase mb-1">
              {t("limitations")}
            </h4>
            <ul className="list-disc list-inside text-sm text-slate-300 space-y-1">
              {analysis.limitations.map((l, i) => (
                <li key={i}>{l}</li>
              ))}
            </ul>
          </div>
        )}

        {analysis.theories_used.length > 0 && (
          <div>
            <h4 className="text-xs font-semibold text-slate-400 uppercase mb-1">
              {t("theories")}
            </h4>
            <div className="flex flex-wrap gap-1">
              {analysis.theories_used.map((theory, i) => (
                <Badge key={i} variant="outline" className="text-xs">
                  {theory}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {analysis.sample_info && (
          <div>
            <h4 className="text-xs font-semibold text-slate-400 uppercase mb-1">
              {t("sampleInfo")}
            </h4>
            <p className="text-sm text-slate-300">{analysis.sample_info}</p>
          </div>
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}
