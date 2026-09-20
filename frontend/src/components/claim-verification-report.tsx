"use client";

import { useState, useCallback } from "react";
import {
  Check,
  AlertTriangle,
  X,
  HelpCircle,
  Loader2,
  ShieldCheck,
  ChevronDown,
  ChevronRight,
  Quote,
  Lightbulb,
  Download,
} from "lucide-react";
import { toast } from "sonner";
import { useTranslations } from "next-intl";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  useClaimVerification,
  useTriggerVerifyClaims,
  type ClaimVerification,
} from "@/hooks/use-fulltext";
import { fetchWithAuth } from "@/lib/api";
import { claimRecordFilename, downloadBlob } from "@/lib/download";
import {
  citationCoverageSummary,
  claimDiagnosticLabelKey,
  claimEvidenceQuotes,
  claimReasonLabelKey,
  claimTextView,
  formatCoveragePercent,
  summarizeProvenance,
  unescapeVerificationText,
} from "@/lib/provenance";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

// ── Types ──────────────────────────────────────────────────────────────────

interface ClaimVerificationReportProps {
  draftId: string;
  projectId: string;
}

// ── Status config ──────────────────────────────────────────────────────────

const statusConfig = {
  verified: {
    icon: Check,
    color: "text-[var(--ds-success)]",
    bg: "bg-[rgba(5,150,105,0.08)]",
    borderColor: "border-[var(--ds-success)]/30",
    labelKey: "claimVerified" as const,
  },
  needs_nuance: {
    icon: AlertTriangle,
    color: "text-amber-500",
    bg: "bg-amber-500/15",
    borderColor: "border-amber-500/30",
    labelKey: "claimNuance" as const,
  },
  unsupported: {
    icon: X,
    color: "text-red-500",
    bg: "bg-red-500/15",
    borderColor: "border-red-500/30",
    labelKey: "claimUnsupported" as const,
  },
  no_full_text: {
    icon: HelpCircle,
    color: "text-[var(--ds-text-muted)]",
    bg: "bg-[var(--ds-text-muted)]/15",
    borderColor: "border-[var(--ds-text-muted)]/30",
    labelKey: "claimNoFullText" as const,
  },
  error: {
    icon: AlertTriangle,
    color: "text-orange-500",
    bg: "bg-orange-500/15",
    borderColor: "border-orange-500/30",
    labelKey: "claimError" as const,
  },
} as const;

// ── Assertion verdict config ────────────────────────────────────────────────

const assertionVerdictConfig = {
  supported: {
    color: "text-[var(--ds-success)]",
    bg: "bg-[rgba(5,150,105,0.08)]",
    labelKey: "claimAssertionSupported" as const,
  },
  contradicted: {
    color: "text-red-500",
    bg: "bg-red-500/15",
    labelKey: "claimAssertionContradicted" as const,
  },
  absent: {
    color: "text-[var(--ds-text-muted)]",
    bg: "bg-[var(--ds-text-muted)]/15",
    labelKey: "claimAssertionAbsent" as const,
  },
} as const;

// ── Expandable claim item ──────────────────────────────────────────────────

function ClaimItem({
  claim,
  t,
}: {
  claim: ClaimVerification;
  t: (key: string, values?: Record<string, string | number>) => string;
}) {
  const [expanded, setExpanded] = useState(false);
  const [assertionsOpen, setAssertionsOpen] = useState(false);
  const config =
    statusConfig[claim.status as keyof typeof statusConfig] ||
    statusConfig.no_full_text;
  const Icon = config.icon;

  // Guard reasons, model-vs-final status, and the per-assertion breakdown are all
  // optional -- absent on reports produced before the v2 prompt/guards shipped.
  const reasons = claim.machine_reasons ?? [];
  const assertions = claim.assertions ?? [];
  // Diagnostics never change `status` -- always rendered as their own neutral row,
  // never merged into `reasons` above. `unstated_details` only has content to show for a
  // `needs_nuance` claim, but the field itself is optional/absent on older reports.
  const diagnostics = claim.diagnostics ?? [];
  const unstatedDetails = claim.unstated_details ?? [];
  const evidenceQuotes = claimEvidenceQuotes(claim);

  return (
    <div
      className={`rounded-lg border ${config.borderColor} bg-[var(--ds-bg-page)] transition-colors`}
    >
      {/* Claim header (clickable) */}
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-start gap-2 px-3 py-2.5 text-left"
      >
        {expanded ? (
          <ChevronDown className="mt-0.5 size-3.5 shrink-0 text-[var(--ds-text-muted)]" />
        ) : (
          <ChevronRight className="mt-0.5 size-3.5 shrink-0 text-[var(--ds-text-muted)]" />
        )}
        <span
          className={`inline-flex shrink-0 items-center gap-1 rounded-md px-1.5 py-0.5 text-xs font-medium ${config.bg} ${config.color}`}
        >
          <Icon className="size-3" />
          {t(config.labelKey)}
        </span>
        <span
          className="line-clamp-2 text-sm text-[var(--ds-text-body)]"
          style={{ fontFamily: "var(--font-body)" }}
        >
          {claim.claim_text}
        </span>
      </button>

      {/* Expanded details */}
      {expanded && (
        <div className="space-y-2 border-t border-[var(--ds-border)] px-3 py-2.5">
          {/* Explanation */}
          <p
            className="text-sm text-[var(--ds-text-secondary)]"
            style={{ fontFamily: "var(--font-body)" }}
          >
            {unescapeVerificationText(claim.explanation, { stripWrappingQuotes: true })}
          </p>

          {/* Machine-readable guard reasons, one chip per fired guard. These are
              the only slugs that can have changed `status` -- diagnostics render in their
              own row below and can never appear here. */}
          {reasons.length > 0 && (
            <div className="flex flex-wrap gap-1" data-testid="claim-reason-chips">
              {reasons.map((slug) => {
                const labelKey = claimReasonLabelKey(slug);
                return (
                  <span
                    key={slug}
                    className="inline-flex items-center gap-1 rounded-md bg-amber-500/10 px-1.5 py-0.5 text-xs font-medium text-amber-600"
                  >
                    {labelKey ? t(labelKey) : slug}
                  </span>
                );
              })}
            </div>
          )}

          {/* Diagnostics, a separate neutral row from the reason chips above --
              these never change `status`, so they carry a neutral colour and their own
              note rather than the amber "stricter" styling. */}
          {diagnostics.length > 0 && (
            <div className="space-y-1" data-testid="claim-diagnostics-row">
              <div className="flex flex-wrap items-center gap-1">
                <span className="text-xs font-medium text-[var(--ds-text-muted)]">
                  {t("claimDiagnosticsLabel")}
                </span>
                {diagnostics.map((slug) => {
                  const labelKey = claimDiagnosticLabelKey(slug);
                  return (
                    <span
                      key={slug}
                      className="inline-flex items-center gap-1 rounded-md bg-[var(--ds-text-muted)]/10 px-1.5 py-0.5 text-xs font-medium text-[var(--ds-text-muted)]"
                    >
                      {labelKey ? t(labelKey) : slug}
                    </span>
                  );
                })}
              </div>
              <p className="text-xs text-[var(--ds-text-muted)]">{t("claimDiagnosticsNote")}</p>
            </div>
          )}

          {/* Peripheral details the v3 precedence rule marks absent for a
              `needs_nuance` claim -- shown only in the expanded card, since they explain
              why this specific claim needed nuance rather than being verified outright. */}
          {claim.status === "needs_nuance" && unstatedDetails.length > 0 && (
            <div data-testid="claim-unstated-details">
              <span
                className="text-xs font-medium text-amber-500"
                style={{ fontFamily: "var(--font-heading)" }}
              >
                {t("claimUnstatedDetails")}
              </span>
              <ul className="mt-1 list-disc space-y-0.5 pl-4">
                {unstatedDetails.map((detail, i) => (
                  <li
                    key={i}
                    className="text-xs text-[var(--ds-text-body)]"
                    style={{ fontFamily: "var(--font-body)" }}
                  >
                    {unescapeVerificationText(detail)}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Evidence quote(s). `evidence_quotes` carries every verbatim span the
              model used (multi-span composition); an older report without it falls back
              to the single `evidence_quote`. */}
          {evidenceQuotes.length > 0 && (
            <div className="space-y-1.5">
              {evidenceQuotes.map((quote, i) => (
                <div
                  key={i}
                  className="flex items-start gap-2 rounded-md bg-[var(--ds-bg-card)] px-3 py-2"
                >
                  <Quote className="mt-0.5 size-3.5 shrink-0 text-[var(--ds-primary)]" />
                  <p
                    className="text-xs italic text-[var(--ds-text-body)]"
                    style={{ fontFamily: "var(--font-body)" }}
                  >
                    {unescapeVerificationText(quote)}
                  </p>
                </div>
              ))}
            </div>
          )}

          {/* Suggested revision (guards against the literal string "null" some
              provider responses store instead of an absent value) */}
          {claim.suggested_revision && claim.suggested_revision.trim().toLowerCase() !== "null" && (
            <div className="flex items-start gap-2 rounded-md bg-amber-500/5 px-3 py-2">
              <Lightbulb className="mt-0.5 size-3.5 shrink-0 text-amber-500" />
              <div>
                <span
                  className="text-xs font-medium text-amber-500"
                  style={{ fontFamily: "var(--font-heading)" }}
                >
                  {t("claimSuggestedRevision")}
                </span>
                <p
                  className="mt-0.5 text-xs text-[var(--ds-text-body)]"
                  style={{ fontFamily: "var(--font-body)" }}
                >
                  {claim.suggested_revision}
                </p>
              </div>
            </div>
          )}

          {/* Per-assertion breakdown (steps 1-2 of the v2 prompt), collapsed by
              default because most claims have several and the header already gives the
              aggregated status. */}
          {assertions.length > 0 && (
            <div>
              <button
                type="button"
                onClick={() => setAssertionsOpen((open) => !open)}
                className="flex items-center gap-1 text-xs font-medium text-[var(--ds-text-muted)]"
              >
                {assertionsOpen ? (
                  <ChevronDown className="size-3" />
                ) : (
                  <ChevronRight className="size-3" />
                )}
                {t("claimAssertionsToggle", { count: assertions.length })}
              </button>
              {assertionsOpen && (
                <ul className="mt-1.5 space-y-1.5" data-testid="claim-assertions-list">
                  {assertions.map((assertion, i) => {
                    const verdictConfig =
                      assertionVerdictConfig[
                        assertion.verdict as keyof typeof assertionVerdictConfig
                      ];
                    return (
                      <li
                        key={i}
                        className="rounded-md bg-[var(--ds-bg-card)] px-2.5 py-1.5 text-xs"
                      >
                        <div className="flex flex-wrap items-center gap-1.5">
                          <span
                            className={`inline-flex items-center rounded-md px-1.5 py-0.5 font-medium ${
                              verdictConfig ? verdictConfig.bg : "bg-[var(--ds-text-muted)]/15"
                            } ${verdictConfig ? verdictConfig.color : "text-[var(--ds-text-muted)]"}`}
                          >
                            {verdictConfig ? t(verdictConfig.labelKey) : assertion.verdict}
                          </span>
                          <span className="text-[var(--ds-text-muted)]">{assertion.kind}</span>
                        </div>
                        <p
                          className="mt-1 text-[var(--ds-text-body)]"
                          style={{ fontFamily: "var(--font-body)" }}
                        >
                          {assertion.text}
                        </p>
                        {/* Every verbatim span composed to reach this assertion's
                            verdict; an older report without `quotes` falls back to the
                            single `quote`. */}
                        {(assertion.quotes && assertion.quotes.length > 0
                          ? assertion.quotes
                          : assertion.quote
                            ? [assertion.quote]
                            : []
                        ).map((quote, qi) => (
                          <p
                            key={qi}
                            className="mt-1 italic text-[var(--ds-text-muted)]"
                            style={{ fontFamily: "var(--font-body)" }}
                          >
                            {unescapeVerificationText(quote)}
                          </p>
                        ))}
                        {assertion.verdict === "contradicted" &&
                          (assertion.claim_value || assertion.source_value) && (
                            <p className="mt-1 text-[var(--ds-text-muted)]">
                              {assertion.claim_value &&
                                t("claimAssertionClaimValue", { value: assertion.claim_value })}
                              {assertion.claim_value && assertion.source_value && " · "}
                              {assertion.source_value &&
                                t("claimAssertionSourceValue", { value: assertion.source_value })}
                            </p>
                          )}
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Final report row ────────────────────────────────────────────────────────

/**
 * One row of the "ONE report of the final text": every claim shown here is verified by
 * construction (it survived `finalize_draft_document`/`finalize_generated_section`), so
 * there is nothing to explain about how the verdict was reached -- no status badge, no
 * guard reasons, no model-vs-final comparison, no suggested revision. Just what a reader
 * needs to spot-check the citation: the sentence, the narrower clause actually checked
 * (when a citation-link proposition narrowed it), the evidence the writer cited it for,
 * and the paper.
 */
function FinalClaimRow({
  claim,
  t,
}: {
  claim: ClaimVerification;
  t: (key: string, values?: Record<string, string | number>) => string;
}) {
  const { sentence, checkedClause } = claimTextView(claim);
  const quotes = claimEvidenceQuotes(claim);

  return (
    <div className="space-y-1.5 rounded-lg border border-[var(--ds-success)]/30 bg-[var(--ds-bg-page)] px-3 py-2.5">
      <p
        className="text-sm text-[var(--ds-text-body)]"
        style={{ fontFamily: "var(--font-body)" }}
      >
        {sentence}
      </p>
      {checkedClause && (
        <p
          className="text-xs text-[var(--ds-text-muted)]"
          style={{ fontFamily: "var(--font-body)" }}
        >
          <span className="font-medium">{t("claimCheckedClauseLabel")}</span> {checkedClause}
        </p>
      )}
      {quotes.length > 0 && (
        <div className="space-y-1.5">
          {quotes.map((quote, i) => (
            <div
              key={i}
              className="flex items-start gap-2 rounded-md bg-[var(--ds-bg-card)] px-3 py-2"
            >
              <Quote className="mt-0.5 size-3.5 shrink-0 text-[var(--ds-primary)]" />
              <p
                className="text-xs italic text-[var(--ds-text-body)]"
                style={{ fontFamily: "var(--font-body)" }}
              >
                {unescapeVerificationText(quote)}
              </p>
            </div>
          ))}
        </div>
      )}
      <p
        className="text-xs text-[var(--ds-text-muted)]"
        style={{ fontFamily: "var(--font-body)" }}
      >
        {claim.paper_title || t("claimPaperUnknown")}
      </p>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────

export function ClaimVerificationReport({
  draftId,
  projectId,
}: ClaimVerificationReportProps) {
  const t = useTranslations("search");

  const { data: report, isLoading, isError } = useClaimVerification(draftId);
  const verifyMutation = useTriggerVerifyClaims();

  // The product shows one verification report of the final text, not a before-and-after
  // of a raw pass versus a healed one. `final_report` is that final view -- present
  // whenever the standalone action (verify-and-heal) has healed this draft. An older
  // report (or one with nothing left to show) falls back to the raw, full report below.
  const finalReport = report?.final_report ?? null;

  // Provenance (model actually served, temperature, prompt version) and full-text
  // coverage are optional on older reports; render the line only when present.
  const provenanceSummary = report ? summarizeProvenance(report.provenance) : null;
  const coveragePct = report ? formatCoveragePercent(report.full_text_coverage) : null;
  // Report-level guard counts, next to the existing model/coverage line. Optional
  // on the report type -- absent on reports built before the v2 prompt/guards shipped.
  // Never shown next to the final report: a guard count describes the raw, pre-heal
  // pass, and would read as a before-and-after next to an all-verified final list.
  const hasGuardCounts =
    !finalReport &&
    (typeof report?.contradicted_count === "number" ||
      typeof report?.guarded_count === "number");
  // How many citations were found, resolved and sent to the verifier, independent of
  // how they were rendered. Absent on a report from `verify_user_edits` or one built
  // before this field existed.
  const citationCoverage = report ? citationCoverageSummary(report.citation_coverage) : null;
  const reportLineSegments = [
    provenanceSummary
      ? t("claimProvenanceLine", {
          models: provenanceSummary.models,
          temperature: provenanceSummary.temperature,
          version: provenanceSummary.promptVersion,
        })
      : null,
    coveragePct !== null ? t("claimFullTextCoverage", { pct: coveragePct }) : null,
    citationCoverage
      ? t("claimCitationCoverageLine", {
          found: citationCoverage.found,
          linked: citationCoverage.linked,
          unresolved: citationCoverage.unresolved,
        })
      : null,
    hasGuardCounts
      ? t("claimGuardCountsLine", {
          contradicted: report?.contradicted_count ?? 0,
          guarded: report?.guarded_count ?? 0,
        })
      : null,
  ].filter((segment): segment is string => Boolean(segment));

  const [isVerifying, setIsVerifying] = useState(false);
  const [isDownloadingRecord, setIsDownloadingRecord] = useState(false);
  const taskId = report?.task_id;

  const handleDownloadClaimRecord = useCallback(async () => {
    if (!taskId) return;
    setIsDownloadingRecord(true);
    try {
      const res = await fetchWithAuth(
        `${API_URL}/tasks/${taskId}/claim-record?format=csv`,
      );
      if (!res.ok) throw new Error(res.statusText);
      downloadBlob(await res.blob(), claimRecordFilename(taskId));
    } catch {
      toast.error(t("claimDownloadRecordFailed"));
    } finally {
      setIsDownloadingRecord(false);
    }
  }, [taskId, t]);

  const handleVerify = useCallback(() => {
    setIsVerifying(true);
    verifyMutation.mutate(
      { projectId, draftId },
      {
        onSuccess: () => {
          toast.info(t("claimVerifyStarted"));
          // The query will auto-refetch once the backend finishes
          setIsVerifying(false);
        },
        onError: (err) => {
          toast.error(err.message || t("claimVerifyFailed"));
          setIsVerifying(false);
        },
      },
    );
  }, [verifyMutation, projectId, draftId, t]);

  return (
    <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle
            className="flex items-center gap-2 text-base text-[var(--ds-text-heading)]"
            style={{ fontFamily: "var(--font-heading)" }}
          >
            <ShieldCheck className="size-4 text-[var(--ds-primary)]" />
            {t("claimVerificationTitle")}
          </CardTitle>
          <Button
            variant="ghost"
            size="sm"
            className="text-[var(--ds-primary)] hover:bg-[var(--ds-primary-light)] hover:text-[var(--ds-primary)]"
            onClick={handleVerify}
            disabled={isVerifying || verifyMutation.isPending}
          >
            {isVerifying || verifyMutation.isPending ? (
              <Loader2 className="mr-1 size-3.5 animate-spin" />
            ) : (
              <ShieldCheck className="mr-1 size-3.5" />
            )}
            {isVerifying
              ? t("claimVerifying")
              : t("claimVerifyButton")}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {/* Loading state */}
        {isLoading && (
          <div className="flex items-center justify-center py-6">
            <Loader2 className="size-5 animate-spin text-[var(--ds-primary)]" />
          </div>
        )}

        {/* Error / no data state */}
        {!isLoading && (isError || !report) && (
          <div className="py-6 text-center">
            <ShieldCheck className="mx-auto mb-2 size-8 text-[var(--ds-text-muted)]" />
            <p
              className="text-sm text-[var(--ds-text-secondary)]"
              style={{ fontFamily: "var(--font-body)" }}
            >
              {t("claimNoReport")}
            </p>
          </div>
        )}

        {/* Report data */}
        {!isLoading && report && (
          <div className="space-y-4">
            {/* Summary bar. Once the draft is healed, every claim shown below is
                verified by construction, so the summary states only that one count --
                a nuance/unsupported/abstract-only breakdown next to an all-verified
                list would describe the raw pass that no longer matches what is on
                screen. */}
            <div className="flex flex-wrap items-center gap-3 rounded-lg bg-[var(--ds-bg-page)] px-4 py-3">
              {finalReport ? (
                <span className="flex items-center gap-1.5 text-sm font-medium text-[var(--ds-success)]">
                  <Check className="size-3.5" />
                  {t("claimSummaryVerified", {
                    count: finalReport.verified_count,
                  })}
                </span>
              ) : (
                <>
                  <span className="flex items-center gap-1.5 text-sm font-medium text-[var(--ds-success)]">
                    <Check className="size-3.5" />
                    {t("claimSummaryVerified", {
                      count: report.verified_count,
                    })}
                  </span>
                  <span className="flex items-center gap-1.5 text-sm font-medium text-amber-500">
                    <AlertTriangle className="size-3.5" />
                    {t("claimSummaryNuance", {
                      count: report.nuance_count,
                    })}
                  </span>
                  <span className="flex items-center gap-1.5 text-sm font-medium text-red-500">
                    <X className="size-3.5" />
                    {t("claimSummaryUnsupported", {
                      count: report.unsupported_count,
                    })}
                  </span>
                  <span className="flex items-center gap-1.5 text-sm font-medium text-[var(--ds-text-muted)]">
                    <HelpCircle className="size-3.5" />
                    {t("claimSummaryAbstractOnly", {
                      count: report.abstract_only_count,
                    })}
                  </span>
                </>
              )}
            </div>

            {/* Provenance + coverage + guard-count line, with the claim record export
                next to it. Contradicted/guarded counts are optional -- absent on
                reports produced before the v2 prompt/guards shipped -- so they only add a
                segment when the backend actually sent them. */}
            {(provenanceSummary || coveragePct !== null || citationCoverage || hasGuardCounts || taskId) && (
              <div className="flex flex-wrap items-center justify-between gap-2 px-1">
                <p
                  className="text-xs text-[var(--ds-text-muted)]"
                  style={{ fontFamily: "var(--font-body)" }}
                  data-testid="claim-provenance-line"
                >
                  {reportLineSegments.join(" · ")}
                </p>
                {taskId && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleDownloadClaimRecord}
                    disabled={isDownloadingRecord}
                    className="gap-1 border-[var(--ds-border)] text-[var(--ds-text-heading)]"
                  >
                    {isDownloadingRecord ? (
                      <Loader2 className="size-3.5 animate-spin" />
                    ) : (
                      <Download className="size-3.5" />
                    )}
                    {t("claimDownloadRecord")}
                  </Button>
                )}
              </div>
            )}

            {/* Claim list: the final, healed view (sentence, checked clause, evidence,
                paper) when one is available, else the raw report. */}
            {finalReport ? (
              finalReport.verifications.length > 0 ? (
                <div className="space-y-2">
                  {finalReport.verifications.map((claim, i) => (
                    <FinalClaimRow key={`${claim.paper_id}-${i}`} claim={claim} t={t} />
                  ))}
                </div>
              ) : (
                <p
                  className="py-4 text-center text-sm text-[var(--ds-text-secondary)]"
                  style={{ fontFamily: "var(--font-body)" }}
                >
                  {t("claimFinalEmpty")}
                </p>
              )
            ) : report.verifications.length > 0 ? (
              <div className="space-y-2">
                {report.verifications.map((claim, i) => (
                  <ClaimItem
                    key={`${claim.paper_id}-${i}`}
                    claim={claim}
                    t={t}
                  />
                ))}
              </div>
            ) : (
              <p
                className="py-4 text-center text-sm text-[var(--ds-text-secondary)]"
                style={{ fontFamily: "var(--font-body)" }}
              >
                {t("claimNoClaims")}
              </p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
