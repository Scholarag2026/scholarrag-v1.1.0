"use client";

import { useTranslations } from "next-intl";

interface Prerequisite {
  label: string;
  completed: boolean;
  current?: number;
  total?: number;
  optional?: boolean;
}

interface EmptyStateProps {
  icon: string;
  title: string;
  description: string;
  prerequisites?: Prerequisite[];
  actionLabel: string;
  onAction: () => void;
  actionDisabled?: boolean;
}

export function EmptyState({
  icon,
  title,
  description,
  prerequisites,
  actionLabel,
  onAction,
  actionDisabled = false,
}: EmptyStateProps) {
  const t = useTranslations("emptyStates");
  return (
    <div className="flex flex-col items-center justify-center px-6 py-16">
      {/* Icon */}
      <span className="text-3xl" role="img" aria-label={title}>
        {icon}
      </span>

      {/* Title */}
      <h2
        className="mt-4 text-lg font-bold text-[var(--ds-text-heading)]"
        style={{ fontFamily: "var(--font-heading)" }}
      >
        {title}
      </h2>

      {/* Description */}
      <p className="mt-2 max-w-[360px] text-center text-[13px] leading-relaxed text-[var(--ds-text-muted)]">
        {description}
      </p>

      {/* Prerequisites card */}
      {prerequisites && prerequisites.length > 0 && (
        <div className="mt-6 w-full max-w-sm rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          {prerequisites.map((prereq, index) => (
            <div
              key={index}
              className={`flex items-center justify-between px-4 py-3 ${
                index !== prerequisites.length - 1
                  ? "border-b border-[var(--ds-border)]"
                  : ""
              }`}
            >
              <div className="flex items-center gap-3">
                {/* Status icon */}
                {prereq.completed ? (
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-emerald-100 text-emerald-600">
                    <svg
                      width="12"
                      height="12"
                      viewBox="0 0 12 12"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path
                        d="M2.5 6L5 8.5L9.5 3.5"
                        stroke="currentColor"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </span>
                ) : prereq.optional ? (
                  <span className="flex h-5 w-5 items-center justify-center rounded-full border border-[var(--ds-text-secondary)] text-[var(--ds-text-secondary)]">
                    <svg
                      width="8"
                      height="8"
                      viewBox="0 0 8 8"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <circle
                        cx="4"
                        cy="4"
                        r="3"
                        stroke="currentColor"
                        strokeWidth="1"
                      />
                    </svg>
                  </span>
                ) : (
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-amber-100 text-amber-600">
                    <svg
                      width="12"
                      height="12"
                      viewBox="0 0 12 12"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path
                        d="M6 3.5V6.5"
                        stroke="currentColor"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                      />
                      <circle cx="6" cy="8.5" r="0.75" fill="currentColor" />
                    </svg>
                  </span>
                )}

                {/* Label */}
                <span
                  className={`text-[13px] ${
                    prereq.completed
                      ? "text-[var(--ds-text-muted)] line-through"
                      : prereq.optional
                        ? "text-[var(--ds-text-body)]"
                        : "font-medium text-[var(--ds-text-heading)]"
                  }`}
                >
                  {prereq.label}
                </span>
              </div>

              {/* Right side: count or status text */}
              <span className="text-[12px] text-[var(--ds-text-secondary)]">
                {prereq.completed && prereq.total != null
                  ? `${prereq.total} / ${prereq.total}`
                  : !prereq.completed && !prereq.optional && prereq.total != null
                    ? `${prereq.current ?? 0} / ${prereq.total}`
                    : prereq.optional
                      ? t("prerequisites.recommended")
                      : null}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Action button */}
      <button
        onClick={onAction}
        disabled={actionDisabled}
        className={`mt-6 rounded-lg bg-[var(--ds-primary)] px-5 py-2.5 text-[13px] font-medium text-white transition-colors hover:bg-[var(--ds-primary-hover)] ${
          actionDisabled ? "cursor-not-allowed opacity-50" : ""
        }`}
      >
        {actionLabel}
      </button>
    </div>
  );
}
