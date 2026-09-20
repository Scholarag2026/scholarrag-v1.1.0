"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useTranslations } from "next-intl";
import { useAuthStore } from "@/lib/auth";
import { LanguageSwitcher } from "@/components/language-switcher";
import { ArrowRight } from "lucide-react";

export default function Home() {
  const router = useRouter();
  const user = useAuthStore((s) => s.user);
  const isLoading = useAuthStore((s) => s.isLoading);
  const tCommon = useTranslations("common");
  const tApp = useTranslations("app");
  const tLanding = useTranslations("landing");
  useEffect(() => {
    if (!isLoading && user) {
      router.replace("/projects");
    }
  }, [user, isLoading, router]);

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div
          className="text-[var(--ds-text-muted)]"
          style={{ fontFamily: "var(--font-body)" }}
        >
          {tCommon("loading")}
        </div>
      </div>
    );
  }

  if (user) {
    return null;
  }

  return (
    <div
      className="min-h-screen"
      style={{ background: "var(--ds-bg-page)", fontFamily: "var(--font-body)" }}
    >
      {/* ─── Navigation Bar ─── */}
      <nav
        className="sticky top-0 z-50 border-b"
        style={{
          background: "var(--ds-bg-page)",
          borderColor: "var(--ds-border)",
        }}
      >
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2.5">
            <div
              className="flex h-8 w-8 items-center justify-center rounded-md text-white font-bold text-lg"
              style={{ background: "var(--ds-primary)" }}
            >
              D
            </div>
            <span
              className="text-xl font-semibold"
              style={{
                fontFamily: "var(--font-heading)",
                color: "var(--ds-text-heading)",
              }}
            >
              ScholarRAG
            </span>
          </Link>

          {/* Right side actions */}
          <div className="flex items-center gap-3">
            <LanguageSwitcher />
            <Link
              href="/login"
              className="hidden sm:inline-flex px-3 py-1.5 text-sm font-medium transition-colors hover:opacity-80"
              style={{ color: "var(--ds-text-body)" }}
            >
              {tLanding("signIn")}
            </Link>
            <Link
              href="/register"
              className="inline-flex items-center rounded-lg px-4 py-2 text-sm font-medium text-white transition-colors"
              style={{ background: "var(--ds-primary)" }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.background = "var(--ds-primary-hover)")
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.background = "var(--ds-primary)")
              }
            >
              {tLanding("getStarted")}
            </Link>
          </div>
        </div>
      </nav>

      {/* ─── Hero Section ─── */}
      <section className="mx-auto flex min-h-[calc(100vh-4rem)] max-w-4xl flex-col items-center justify-center px-4 text-center sm:px-6">
        {/* Pill badge */}
        <span
          className="inline-block rounded-full px-4 py-1.5 text-sm font-medium mb-6"
          style={{
            background: "var(--ds-primary-light)",
            color: "var(--ds-primary)",
          }}
        >
          {tApp("tagline")}
        </span>

        {/* Heading */}
        <h1
          className="text-3xl sm:text-4xl lg:text-[44px] font-bold leading-tight mb-5"
          style={{
            fontFamily: "var(--font-heading)",
            color: "var(--ds-text-heading)",
          }}
        >
          {tLanding("heroHeading")}
        </h1>

        {/* Subheading */}
        <p
          className="mx-auto max-w-2xl text-base sm:text-lg leading-relaxed mb-10"
          style={{ color: "var(--ds-text-body)" }}
        >
          {tLanding("heroSubheading")}
        </p>

        {/* CTAs */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link
            href="/register"
            className="inline-flex items-center gap-2 rounded-lg px-6 py-3 text-base font-medium text-white transition-colors"
            style={{ background: "var(--ds-primary)" }}
            onMouseEnter={(e) =>
              (e.currentTarget.style.background = "var(--ds-primary-hover)")
            }
            onMouseLeave={(e) =>
              (e.currentTarget.style.background = "var(--ds-primary)")
            }
          >
            {tLanding("startFree")}
            <ArrowRight className="size-4" />
          </Link>
          <Link
            href="/how-it-works"
            className="inline-flex items-center rounded-lg border px-6 py-3 text-base font-medium transition-colors hover:bg-white/60"
            style={{
              borderColor: "var(--ds-border)",
              color: "var(--ds-text-body)",
              background: "var(--ds-bg-card)",
            }}
          >
            {tLanding("seeHowItWorks")}
          </Link>
        </div>
      </section>

      {/* ─── Minimal Footer ─── */}
      <footer
        className="border-t py-8 text-center text-xs"
        style={{
          borderColor: "var(--ds-border)",
          color: "var(--ds-text-muted)",
        }}
      >
        {tLanding("copyright", { year: new Date().getFullYear() })}
      </footer>
    </div>
  );
}
