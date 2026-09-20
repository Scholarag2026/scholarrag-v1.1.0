"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import {
  ChevronDown,
  ChevronRight,
  ClipboardCheck,
  Copy,
  FlaskConical,
  ShieldCheck,
  Target,
  Users,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import type { ResearchDesign } from "@/hooks/use-research-design";

interface ResearchDesignViewProps {
  design: ResearchDesign;
}

function SectionHeader({
  icon: Icon,
  title,
  open,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  open: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      <Icon className="h-4 w-4 text-[var(--ds-primary)]" />
      <span className="text-sm font-semibold text-[var(--ds-text-body)]">{title}</span>
      {open ? (
        <ChevronDown className="ml-auto h-4 w-4 text-[var(--ds-text-secondary)]" />
      ) : (
        <ChevronRight className="ml-auto h-4 w-4 text-[var(--ds-text-secondary)]" />
      )}
    </div>
  );
}

const approachColors: Record<string, string> = {
  qualitative: "border-emerald-500/50 text-emerald-400",
  quantitative: "border-blue-500/50 text-blue-400",
  mixed_methods: "border-purple-500/50 text-purple-400",
};

const instrumentTypeColors: Record<string, string> = {
  survey: "border-blue-500/50 text-blue-400",
  interview: "border-emerald-500/50 text-emerald-400",
  observation: "border-amber-500/50 text-amber-400",
  focus_group: "border-purple-500/50 text-purple-400",
  document_analysis: "border-[var(--ds-border)] text-[var(--ds-text-secondary)]",
};

export function ResearchDesignView({ design }: ResearchDesignViewProps) {
  const t = useTranslations("design");
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    methodology: true,
    ethics: true,
    instruments: true,
    sampling: true,
    validity: true,
  });
  const [copied, setCopied] = useState(false);

  function toggleSection(key: string) {
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  async function handleCopyConsent() {
    try {
      await navigator.clipboard.writeText(design.ethics.consent_template);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard API not available
    }
  }

  return (
    <div className="space-y-4">
      {/* Methodology */}
      <Collapsible
        open={openSections.methodology}
        onOpenChange={() => toggleSection("methodology")}
      >
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardHeader className="cursor-pointer pb-3">
            <CollapsibleTrigger className="w-full">
              <SectionHeader
                icon={FlaskConical}
                title={t("methodology")}
                open={openSections.methodology}
              />
            </CollapsibleTrigger>
          </CardHeader>
          <CollapsibleContent>
            <CardContent className="space-y-3 pt-0">
              <div className="flex items-center gap-2">
                <span className="text-xs text-[var(--ds-text-secondary)]">{t("approach")}:</span>
                <Badge
                  variant="outline"
                  className={approachColors[design.methodology.approach] || ""}
                >
                  {design.methodology.approach.replace("_", " ")}
                </Badge>
              </div>
              <div>
                <span className="text-xs text-[var(--ds-text-secondary)]">{t("designType")}:</span>
                <p className="text-sm text-[var(--ds-text-body)] mt-0.5">
                  {design.methodology.design_type}
                </p>
              </div>
              <div>
                <span className="text-xs text-[var(--ds-text-secondary)]">{t("justification")}:</span>
                <p className="text-sm text-[var(--ds-text-body)] mt-0.5 whitespace-pre-line">
                  {design.methodology.justification}
                </p>
              </div>
              {design.methodology.research_questions.length > 0 && (
                <div>
                  <span className="text-xs text-[var(--ds-text-secondary)]">{t("researchQuestions")}:</span>
                  <ol className="mt-1 list-decimal list-inside space-y-1">
                    {design.methodology.research_questions.map((rq, i) => (
                      <li key={i} className="text-sm text-[var(--ds-text-body)]">
                        {rq}
                      </li>
                    ))}
                  </ol>
                </div>
              )}
              <div>
                <span className="text-xs text-[var(--ds-text-secondary)]">
                  {t("theoreticalFramework")}:
                </span>
                <p className="text-sm text-[var(--ds-text-body)] mt-0.5">
                  {design.methodology.theoretical_framework}
                </p>
              </div>
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>

      {/* Ethics */}
      <Collapsible
        open={openSections.ethics}
        onOpenChange={() => toggleSection("ethics")}
      >
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardHeader className="cursor-pointer pb-3">
            <CollapsibleTrigger className="w-full">
              <SectionHeader
                icon={ShieldCheck}
                title={t("ethics")}
                open={openSections.ethics}
              />
            </CollapsibleTrigger>
          </CardHeader>
          <CollapsibleContent>
            <CardContent className="space-y-3 pt-0">
              {design.ethics.considerations.length > 0 && (
                <div>
                  <span className="text-xs text-[var(--ds-text-secondary)]">{t("considerations")}:</span>
                  <ul className="mt-1 list-disc list-inside space-y-0.5">
                    {design.ethics.considerations.map((c, i) => (
                      <li key={i} className="text-sm text-[var(--ds-text-body)]">
                        {c}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-[var(--ds-text-secondary)]">{t("consentTemplate")}:</span>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 gap-1.5 text-xs text-[var(--ds-text-secondary)] hover:text-[var(--ds-text-body)]"
                    onClick={handleCopyConsent}
                  >
                    {copied ? (
                      <>
                        <ClipboardCheck className="h-3 w-3" />
                        {t("copied")}
                      </>
                    ) : (
                      <>
                        <Copy className="h-3 w-3" />
                        {t("copyToClipboard")}
                      </>
                    )}
                  </Button>
                </div>
                <div className="mt-1 rounded-md border border-[var(--ds-border)] bg-[var(--ds-bg-subtle)] p-3">
                  <p className="text-sm text-[var(--ds-text-body)] whitespace-pre-line">
                    {design.ethics.consent_template}
                  </p>
                </div>
              </div>
              <div>
                <span className="text-xs text-[var(--ds-text-secondary)]">{t("dataProtection")}:</span>
                <p className="text-sm text-[var(--ds-text-body)] mt-0.5 whitespace-pre-line">
                  {design.ethics.data_protection}
                </p>
              </div>
              <div>
                <span className="text-xs text-[var(--ds-text-secondary)]">{t("irbNotes")}:</span>
                <p className="text-sm text-[var(--ds-text-body)] mt-0.5 whitespace-pre-line">
                  {design.ethics.irb_notes}
                </p>
              </div>
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>

      {/* Instruments */}
      <Collapsible
        open={openSections.instruments}
        onOpenChange={() => toggleSection("instruments")}
      >
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardHeader className="cursor-pointer pb-3">
            <CollapsibleTrigger className="w-full">
              <SectionHeader
                icon={ClipboardCheck}
                title={`${t("instruments")} (${design.instruments.length})`}
                open={openSections.instruments}
              />
            </CollapsibleTrigger>
          </CardHeader>
          <CollapsibleContent>
            <CardContent className="space-y-3 pt-0">
              {design.instruments.map((inst, i) => (
                <div
                  key={i}
                  className="rounded-lg border border-[var(--ds-border)] p-3 space-y-2"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-[var(--ds-text-body)]">
                      {inst.name}
                    </span>
                    <Badge
                      variant="outline"
                      className={
                        instrumentTypeColors[inst.type] ||
                        "border-[var(--ds-border)] text-[var(--ds-text-secondary)]"
                      }
                    >
                      {inst.type.replace("_", " ")}
                    </Badge>
                  </div>
                  <p className="text-sm text-[var(--ds-text-body)]">{inst.description}</p>
                  {inst.sample_questions.length > 0 && (
                    <div>
                      <span className="text-xs text-[var(--ds-text-secondary)]">
                        {t("sampleQuestions")}:
                      </span>
                      <ol className="mt-1 list-decimal list-inside space-y-0.5">
                        {inst.sample_questions.map((q, j) => (
                          <li key={j} className="text-xs text-[var(--ds-text-body)]">
                            {q}
                          </li>
                        ))}
                      </ol>
                    </div>
                  )}
                  <div>
                    <span className="text-xs text-[var(--ds-text-secondary)]">
                      {t("administration")}:
                    </span>
                    <p className="text-xs text-[var(--ds-text-body)] mt-0.5">
                      {inst.administration}
                    </p>
                  </div>
                </div>
              ))}
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>

      {/* Sampling */}
      <Collapsible
        open={openSections.sampling}
        onOpenChange={() => toggleSection("sampling")}
      >
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardHeader className="cursor-pointer pb-3">
            <CollapsibleTrigger className="w-full">
              <SectionHeader
                icon={Users}
                title={t("sampling")}
                open={openSections.sampling}
              />
            </CollapsibleTrigger>
          </CardHeader>
          <CollapsibleContent>
            <CardContent className="space-y-3 pt-0">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <span className="text-xs text-[var(--ds-text-secondary)]">
                    {t("sampling")}:
                  </span>
                  <p className="text-sm text-[var(--ds-text-body)] mt-0.5">
                    {design.sampling.strategy_type}
                  </p>
                </div>
                <div>
                  <span className="text-xs text-[var(--ds-text-secondary)]">
                    {t("sampleSize")}:
                  </span>
                  <p className="text-sm text-[var(--ds-text-body)] mt-0.5">
                    {design.sampling.sample_size}
                  </p>
                </div>
              </div>
              <div>
                <span className="text-xs text-[var(--ds-text-secondary)]">
                  {t("targetPopulation")}:
                </span>
                <p className="text-sm text-[var(--ds-text-body)] mt-0.5">
                  {design.sampling.target_population}
                </p>
              </div>
              <div>
                <span className="text-xs text-[var(--ds-text-secondary)]">
                  {t("justification")}:
                </span>
                <p className="text-sm text-[var(--ds-text-body)] mt-0.5">
                  {design.sampling.justification}
                </p>
              </div>
              <div>
                <span className="text-xs text-[var(--ds-text-secondary)]">
                  {t("recruitmentPlan")}:
                </span>
                <p className="text-sm text-[var(--ds-text-body)] mt-0.5 whitespace-pre-line">
                  {design.sampling.recruitment_plan}
                </p>
              </div>
              {design.sampling.inclusion_criteria.length > 0 && (
                <div>
                  <span className="text-xs text-[var(--ds-text-secondary)]">
                    {t("inclusionCriteria")}:
                  </span>
                  <ul className="mt-1 list-disc list-inside space-y-0.5">
                    {design.sampling.inclusion_criteria.map((c, i) => (
                      <li key={i} className="text-sm text-[var(--ds-text-body)]">
                        {c}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {design.sampling.exclusion_criteria.length > 0 && (
                <div>
                  <span className="text-xs text-[var(--ds-text-secondary)]">
                    {t("exclusionCriteria")}:
                  </span>
                  <ul className="mt-1 list-disc list-inside space-y-0.5">
                    {design.sampling.exclusion_criteria.map((c, i) => (
                      <li key={i} className="text-sm text-[var(--ds-text-body)]">
                        {c}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>

      {/* Validity */}
      <Collapsible
        open={openSections.validity}
        onOpenChange={() => toggleSection("validity")}
      >
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardHeader className="cursor-pointer pb-3">
            <CollapsibleTrigger className="w-full">
              <SectionHeader
                icon={Target}
                title={t("validity")}
                open={openSections.validity}
              />
            </CollapsibleTrigger>
          </CardHeader>
          <CollapsibleContent>
            <CardContent className="space-y-3 pt-0">
              {design.validity.internal_validity.length > 0 && (
                <div>
                  <span className="text-xs text-[var(--ds-text-secondary)]">
                    {t("internalValidity")}:
                  </span>
                  <ul className="mt-1 list-disc list-inside space-y-0.5">
                    {design.validity.internal_validity.map((v, i) => (
                      <li key={i} className="text-sm text-[var(--ds-text-body)]">
                        {v}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {design.validity.external_validity.length > 0 && (
                <div>
                  <span className="text-xs text-[var(--ds-text-secondary)]">
                    {t("externalValidity")}:
                  </span>
                  <ul className="mt-1 list-disc list-inside space-y-0.5">
                    {design.validity.external_validity.map((v, i) => (
                      <li key={i} className="text-sm text-[var(--ds-text-body)]">
                        {v}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {design.validity.reliability_measures.length > 0 && (
                <div>
                  <span className="text-xs text-[var(--ds-text-secondary)]">
                    {t("reliabilityMeasures")}:
                  </span>
                  <ul className="mt-1 list-disc list-inside space-y-0.5">
                    {design.validity.reliability_measures.map((m, i) => (
                      <li key={i} className="text-sm text-[var(--ds-text-body)]">
                        {m}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              <div>
                <span className="text-xs text-[var(--ds-text-secondary)]">
                  {t("triangulation")}:
                </span>
                <p className="text-sm text-[var(--ds-text-body)] mt-0.5 whitespace-pre-line">
                  {design.validity.triangulation}
                </p>
              </div>
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>
    </div>
  );
}
