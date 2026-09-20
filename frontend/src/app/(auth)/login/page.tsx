"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Loader2, LogIn } from "lucide-react";
import { useTranslations } from "next-intl";

import { useAuthStore } from "@/lib/auth";
import { ApiError } from "@/lib/api";
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

export default function LoginPage() {
  const router = useRouter();
  const login = useAuthStore((s) => s.login);
  const t = useTranslations("auth");
  const tCommon = useTranslations("common");

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setIsSubmitting(true);

    try {
      await login(email, password);
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
          {t("welcomeBack")}
        </CardTitle>
        <CardDescription
          className="text-[var(--ds-text-secondary)]"
          style={{ fontFamily: "var(--font-body)" }}
        >
          {t("signInDescription")}
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
              placeholder={t("passwordPlaceholder")}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)]"
              style={{ fontFamily: "var(--font-body)" }}
            />
          </div>

          <Button
            type="submit"
            disabled={isSubmitting}
            className="w-full bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)] disabled:opacity-50"
            size="lg"
          >
            {isSubmitting ? (
              <Loader2 className="mr-2 size-4 animate-spin" />
            ) : (
              <LogIn className="mr-2 size-4" />
            )}
            {isSubmitting ? t("signingIn") : t("signIn")}
          </Button>
        </form>

        <div className="mt-4 text-center text-sm" style={{ fontFamily: "var(--font-body)" }}>
          <span className="text-[var(--ds-text-secondary)]">{t("noAccount")} </span>
          <Link
            href="/register"
            className="text-[var(--ds-primary)] underline-offset-4 hover:underline"
          >
            {t("register")}
          </Link>
        </div>
      </CardContent>
    </Card>
  );
}
