"use client";

import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useTranslations } from "next-intl";

interface QualityBadgeProps {
  score: number;
  size?: "sm" | "md";
}

export function QualityBadge({ score, size = "sm" }: QualityBadgeProps) {
  const t = useTranslations("quality");

  const color =
    score > 0.6
      ? "bg-green-500/20 text-green-400 border-green-500/30"
      : score >= 0.3
        ? "bg-yellow-500/20 text-yellow-400 border-yellow-500/30"
        : "bg-red-500/20 text-[var(--ds-error)] border-[var(--ds-error)]/30";

  const label =
    score > 0.6
      ? t("high")
      : score >= 0.3
        ? t("medium")
        : t("low");

  const displayScore = Math.round(score * 100);

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger
          render={
            <span
              className={cn(
                "inline-flex items-center rounded-full border font-mono",
                color,
                size === "sm" ? "px-1.5 py-0.5 text-xs" : "px-2 py-1 text-sm",
              )}
            />
          }
        >
          {displayScore}
        </TooltipTrigger>
        <TooltipContent>
          <p>
            {t("qualityScore")}: {displayScore}/100 ({label})
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
