"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Loader2, UserPlus } from "lucide-react";
import { useTranslations } from "next-intl";

import { useAuthStore } from "@/lib/auth";
import { ApiError } from "@/lib/api";
import { PRIVACY_NOTICE_URL } from "@/lib/links";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export default function RegisterPage() {
  const router = useRouter();
  const register = useAuthStore((s) => s.register);
  const t = useTranslations("auth");
  const tCommon = useTranslations("common");

  const expertiseLevels = [
    { value: "student", label: t("expertiseStudent") },
    { value: "researcher", label: t("expertiseResearcher") },
    { value: "faculty", label: t("expertiseFaculty") },
  ];

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [expertiseLevel, setExpertiseLevel] = useState("student");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setIsSubmitting(true);

    try {
      await register(email, password, name, expertiseLevel);
      router.push("/projects");
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError(tCommon("error"));
      }
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
      <CardHeader className="text-center">
        <CardTitle
          className="text-2xl text-[var(--ds-text-heading)]"
          style={{ fontFamily: "var(--font-heading)" }}
        >
          {t("createAccount")}
        </CardTitle>
        <CardDescription
          className="text-[var(--ds-text-secondary)]"
          style={{ fontFamily: "var(--font-body)" }}
        >
          {t("joinDescription")}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          {error && (
            <div className="rounded-lg border border-[var(--ds-error)]/30 bg-[var(--ds-error)]/10 px-3 py-2 text-sm text-[var(--ds-error)]">
              {error}
            </div>
          )}

          <div className="space-y-2">
            <Label htmlFor="name" className="text-[var(--ds-text-heading)]">
              {tCommon("name")}
            </Label>
            <Input
              id="name"
              type="text"
              placeholder={t("namePlaceholder")}
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)]"
              style={{ fontFamily: "var(--font-body)" }}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="email" className="text-[var(--ds-text-heading)]">
              {tCommon("email")}
            </Label>
            <Input
              id="email"
              type="email"
              placeholder={t("emailPlaceholder")}
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)]"
              style={{ fontFamily: "var(--font-body)" }}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="password" className="text-[var(--ds-text-heading)]">
              {tCommon("password")}
            </Label>
            <Input
              id="password"
              type="password"
              placeholder={t("passwordMinLength")}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={8}
              className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)]"
              style={{ fontFamily: "var(--font-body)" }}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="expertise" className="text-[var(--ds-text-heading)]">
              {t("expertiseLevel")}
            </Label>
            <p className="text-xs text-[var(--ds-text-muted)]">{t("expertiseLevelHelp")}</p>
            <select
              id="expertise"
              value={expertiseLevel}
              onChange={(e) => setExpertiseLevel(e.target.value)}
              className="h-8 w-full rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-page)] px-2.5 py-1 text-sm text-[var(--ds-text-heading)] outline-none transition-colors focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
              style={{ fontFamily: "var(--font-body)" }}
            >
              {expertiseLevels.map((level) => (
                <option key={level.value} value={level.value}>
                  {level.label}
                </option>
              ))}
            </select>
            <div className="rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-subtle)] px-3 py-2 text-xs text-[var(--ds-text-secondary)]">
              {expertiseLevel === "student" && t("expertiseHintStudent")}
              {expertiseLevel === "researcher" && t("expertiseHintResearcher")}
              {expertiseLevel === "faculty" && t("expertiseHintFaculty")}
            </div>
          </div>

          {/* Data-processing notice (what is sent to the LLM provider and stored here) */}
          <p
            className="text-xs leading-relaxed text-[var(--ds-text-secondary)]"
            style={{ fontFamily: "var(--font-body)" }}
            data-testid="data-processing-notice"
          >
            {t.rich("dataNotice", {
              link: (chunks) => (
                <a
                  href={PRIVACY_NOTICE_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[var(--ds-primary)] underline-offset-4 hover:underline"
                >
                  {chunks}
                </a>
              ),
            })}
          </p>

          <Button
            type="submit"
            disabled={isSubmitting}
            className="w-full bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)] disabled:opacity-50"
            size="lg"
          >
            {isSubmitting ? (
              <Loader2 className="mr-2 size-4 animate-spin" />
            ) : (
              <UserPlus className="mr-2 size-4" />
            )}
            {isSubmitting ? t("creatingAccount") : t("createAccount")}
          </Button>
        </form>

        <div className="mt-4 text-center text-sm" style={{ fontFamily: "var(--font-body)" }}>
          <span className="text-[var(--ds-text-secondary)]">{t("hasAccount")} </span>
          <Link
            href="/login"
            className="text-[var(--ds-primary)] underline-offset-4 hover:underline"
          >
            {t("signIn")}
          </Link>
        </div>
      </CardContent>
    </Card>
  );
}
