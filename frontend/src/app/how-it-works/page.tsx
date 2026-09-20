"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import Link from "next/link";
import { useTranslations } from "next-intl";
import { ChevronLeft, ChevronRight, Check, Search } from "lucide-react";

const TOTAL_SLIDES = 4;
const AUTO_INTERVAL = 5000;
const RESUME_DELAY = 2000;

/* ------------------------------------------------------------------ */
/*  Mockup components (inline JSX, no images)                         */
/* ------------------------------------------------------------------ */

function FindMockup() {
  return (
    <div className="flex flex-col gap-3">
      {/* Label */}
      <span
        className="text-xs font-semibold tracking-wide uppercase"
        style={{ color: "var(--ds-primary)" }}
      >
        Smart Search
      </span>

      {/* Search bar */}
      <div className="flex gap-2">
        <div
          className="flex-1 flex items-center gap-2 rounded-lg border px-3 py-2 text-sm"
          style={{
            borderColor: "var(--ds-border)",
            background: "var(--ds-bg-page)",
            color: "var(--ds-text-muted)",
          }}
        >
          <Search className="size-4 shrink-0" style={{ color: "var(--ds-text-muted)" }} />
          <span>AI-assisted learning outcomes...</span>
        </div>
        <button
          className="rounded-lg px-4 py-2 text-sm font-medium text-white shrink-0"
          style={{ background: "var(--ds-primary)" }}
        >
          Search
        </button>
      </div>

      {/* Paper result cards */}
      {[
        {
          title: "Effects of AI Tutoring on Student Learning Outcomes",
          author: "Zhang et al.",
          year: "2024",
          journal: "Computers & Education",
          citations: 47,
        },
        {
          title: "Adaptive Learning Systems: A Systematic Review",
          author: "Garcia & Lee",
          year: "2023",
          journal: "Educational Research Review",
          citations: 82,
        },
      ].map((paper, i) => (
        <div
          key={i}
          className="rounded-lg border p-3"
          style={{
            borderColor: "var(--ds-border)",
            background: "var(--ds-bg-card)",
          }}
        >
          <p
            className="text-sm font-medium leading-snug mb-1.5"
            style={{ color: "var(--ds-text-heading)" }}
          >
            {paper.title}
          </p>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs" style={{ color: "var(--ds-text-muted)" }}>
            <span>{paper.author}</span>
            <span>{paper.year}</span>
            <span>{paper.journal}</span>
            <span>{paper.citations} citations</span>
            <span
              className="rounded px-1.5 py-0.5 text-[10px] font-bold text-white"
              style={{ background: "var(--ds-primary)" }}
            >
              SSCI
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

function ExploreMockup() {
  return (
    <div className="flex flex-col gap-3">
      {/* User message */}
      <div className="flex justify-end">
        <div
          className="max-w-[80%] rounded-2xl rounded-br-md px-4 py-2.5 text-sm text-white"
          style={{ background: "var(--ds-primary)" }}
        >
          What are the main research gaps in AI-assisted learning?
        </div>
      </div>

      {/* AI message */}
      <div className="flex justify-start">
        <div
          className="max-w-[85%] rounded-2xl rounded-bl-md px-4 py-2.5 text-sm leading-relaxed"
          style={{
            background: "var(--ds-bg-subtle)",
            color: "var(--ds-text-body)",
          }}
        >
          Based on your 24 papers, I identified three key gaps: (1) limited longitudinal studies on AI
          tutoring effectiveness, (2) lack of research in non-STEM disciplines, and (3) insufficient
          attention to equity and accessibility.
          <span className="text-xs ml-1" style={{ color: "var(--ds-text-muted)" }}>
            [Zhang 2024; Garcia 2023]
          </span>
        </div>
      </div>

      {/* Gap severity badges */}
      <div className="flex gap-2 mt-1">
        <span className="rounded-full px-3 py-1 text-xs font-semibold text-white bg-red-500">
          High
        </span>
        <span className="rounded-full px-3 py-1 text-xs font-semibold text-white bg-amber-500">
          Medium
        </span>
        <span className="rounded-full px-3 py-1 text-xs font-semibold text-white bg-amber-500">
          Medium
        </span>
      </div>
    </div>
  );
}

function CreateMockup() {
  return (
    <div className="flex flex-col gap-3">
      {/* Editor toolbar */}
      <div className="flex gap-1.5">
        {Array.from({ length: 6 }).map((_, i) => (
          <div
            key={i}
            className="h-7 w-7 rounded"
            style={{ background: "var(--ds-bg-subtle)" }}
          />
        ))}
      </div>

      {/* Heading */}
      <h4
        className="text-base font-bold"
        style={{
          fontFamily: "var(--font-heading)",
          color: "var(--ds-text-heading)",
        }}
      >
        Literature Review
      </h4>

      {/* Paragraph with highlighted phrase */}
      <p className="text-sm leading-relaxed" style={{ color: "var(--ds-text-body)" }}>
        Recent advances in artificial intelligence have significantly transformed educational
        research methodologies.{" "}
        <span
          className="px-0.5 rounded"
          style={{ background: "rgba(13, 93, 86, 0.15)", color: "var(--ds-primary)" }}
        >
          However, a critical gap remains in understanding the long-term impacts
        </span>{" "}
        of AI-assisted learning on student outcomes across diverse institutional
        contexts and disciplinary boundaries.
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main page component                                               */
/* ------------------------------------------------------------------ */

export default function HowItWorksPage() {
  const t = useTranslations("howItWorks");
  const [current, setCurrent] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const resumeRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // showArrows removed — arrows are always visible now

  /* ---------- auto-advance logic ---------- */
  const clearTimers = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (resumeRef.current) {
      clearTimeout(resumeRef.current);
      resumeRef.current = null;
    }
  }, []);

  const startAutoPlay = useCallback(() => {
    clearTimers();
    timerRef.current = setInterval(() => {
      setCurrent((prev) => {
        if (prev >= TOTAL_SLIDES - 1) {
          // Stop at last slide
          if (timerRef.current) clearInterval(timerRef.current);
          return prev;
        }
        return prev + 1;
      });
    }, AUTO_INTERVAL);
  }, [clearTimers]);

  useEffect(() => {
    if (!isPaused && current < TOTAL_SLIDES - 1) {
      startAutoPlay();
    }
    return clearTimers;
  }, [isPaused, current, startAutoPlay, clearTimers]);

  const handleMouseEnter = () => {
    setIsPaused(true);
    clearTimers();
  };

  const handleMouseLeave = () => {
    if (current < TOTAL_SLIDES - 1) {
      resumeRef.current = setTimeout(() => {
        setIsPaused(false);
      }, RESUME_DELAY);
    }
  };

  const goTo = (idx: number) => {
    setCurrent(idx);
    // If jumping backwards from last slide, resume autoplay
    if (idx < TOTAL_SLIDES - 1) {
      setIsPaused(false);
    }
  };

  const goPrev = () => {
    if (current > 0) goTo(current - 1);
  };

  const goNext = () => {
    if (current < TOTAL_SLIDES - 1) goTo(current + 1);
  };

  /* ---------- slide data ---------- */
  const slides = [
    { phase: t("slide1.phase"), title: t("slide1.title"), description: t("slide1.description"), features: [t("slide1.feature1"), t("slide1.feature2"), t("slide1.feature3"), t("slide1.feature4")], mockup: <FindMockup /> },
    { phase: t("slide2.phase"), title: t("slide2.title"), description: t("slide2.description"), features: [t("slide2.feature1"), t("slide2.feature2"), t("slide2.feature3"), t("slide2.feature4")], mockup: <ExploreMockup /> },
    { phase: t("slide3.phase"), title: t("slide3.title"), description: t("slide3.description"), features: [t("slide3.feature1"), t("slide3.feature2"), t("slide3.feature3"), t("slide3.feature4")], mockup: <CreateMockup /> },
  ];

  const progressPct = ((current + 1) / TOTAL_SLIDES) * 100;

  return (
    <div
      className="relative flex flex-col min-h-screen overflow-hidden"
      style={{ background: "var(--ds-bg-page)", fontFamily: "var(--font-body)" }}
    >
      {/* ─── Progress bar (fixed top) ─── */}
      <div className="fixed top-0 left-0 right-0 z-50 h-[3px]" style={{ background: "var(--ds-border)" }}>
        <div
          className="h-full transition-all duration-500 ease-out"
          style={{ width: `${progressPct}%`, background: "var(--ds-primary)" }}
        />
      </div>

      {/* ─── Top navigation ─── */}
      <nav className="fixed top-[3px] left-0 right-0 z-40 h-14 flex items-center justify-between px-4 sm:px-6"
        style={{ background: "var(--ds-bg-page)" }}
      >
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2">
          <div
            className="flex h-7 w-7 items-center justify-center rounded-md text-white font-bold text-sm"
            style={{ background: "var(--ds-primary)" }}
          >
            D
          </div>
          <span
            className="text-lg font-semibold"
            style={{ fontFamily: "var(--font-heading)", color: "var(--ds-text-heading)" }}
          >
            ScholarRAG
          </span>
        </Link>

        {/* Skip */}
        <Link
          href="/register"
          className="text-sm font-medium transition-opacity hover:opacity-70"
          style={{ color: "var(--ds-text-muted)" }}
        >
          {t("skip")}
        </Link>
      </nav>

      {/* ─── Slide area ─── */}
      <main
        className="flex-1 flex items-center justify-center pt-16 pb-16 px-4 sm:px-6"
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        {/* Arrow buttons (visible on hover) */}
        {current > 0 && (
          <button
            onClick={goPrev}
            className="fixed left-4 sm:left-8 top-1/2 -translate-y-1/2 z-30 flex h-10 w-10 items-center justify-center rounded-full border transition-colors duration-200 hover:border-[var(--ds-primary)] hover:text-[var(--ds-primary)]"
            style={{
              borderColor: "var(--ds-border)",
              background: "var(--ds-bg-card)",
              color: "var(--ds-text-body)",
            }}
            aria-label="Previous slide"
          >
            <ChevronLeft className="size-5" />
          </button>
        )}
        {current < TOTAL_SLIDES - 1 && (
          <button
            onClick={goNext}
            className="fixed right-4 sm:right-8 top-1/2 -translate-y-1/2 z-30 flex h-10 w-10 items-center justify-center rounded-full border transition-colors duration-200 hover:border-[var(--ds-primary)] hover:text-[var(--ds-primary)]"
            style={{
              borderColor: "var(--ds-border)",
              background: "var(--ds-bg-card)",
              color: "var(--ds-text-body)",
            }}
            aria-label="Next slide"
          >
            <ChevronRight className="size-5" />
          </button>
        )}

        {/* Slide content with crossfade */}
        <div
          key={current}
          className="w-full max-w-6xl mx-auto hiw-fade-in"
        >
          {current < 3 ? (
            /* ── Phase slides (0-2) ── */
            <div className="flex flex-col lg:flex-row items-center gap-10 lg:gap-16">
              {/* Left: text */}
              <div className="w-full lg:w-[45%] hiw-text-enter">
                {/* Phase badge */}
                <span
                  className="inline-block rounded-full px-4 py-1.5 text-xs font-semibold tracking-wide mb-5"
                  style={{
                    background: "var(--ds-primary-light)",
                    color: "var(--ds-primary)",
                  }}
                >
                  {slides[current].phase}
                </span>

                {/* Title */}
                <h1
                  className="text-2xl sm:text-3xl lg:text-4xl font-bold leading-tight mb-4"
                  style={{
                    fontFamily: "var(--font-heading)",
                    color: "var(--ds-text-heading)",
                  }}
                >
                  {slides[current].title}
                </h1>

                {/* Description */}
                <p
                  className="text-sm sm:text-base leading-relaxed mb-6"
                  style={{ color: "var(--ds-text-body)" }}
                >
                  {slides[current].description}
                </p>

                {/* Feature bullets */}
                <ul className="space-y-2.5">
                  {slides[current].features.map((feat, i) => (
                    <li key={i} className="flex items-start gap-2.5">
                      <div
                        className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full"
                        style={{ background: "var(--ds-primary-light)" }}
                      >
                        <Check className="size-3" style={{ color: "var(--ds-primary)" }} />
                      </div>
                      <span className="text-sm" style={{ color: "var(--ds-text-body)" }}>
                        {feat}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Right: mockup (hidden on sm) */}
              <div className="hidden sm:block w-full lg:w-[55%] hiw-mockup-enter">
                <div
                  className="rounded-xl border overflow-hidden"
                  style={{
                    borderColor: "var(--ds-border)",
                    background: "var(--ds-bg-card)",
                    boxShadow: "var(--ds-shadow-md)",
                  }}
                >
                  {/* Browser title bar */}
                  <div
                    className="flex items-center gap-1.5 px-4 py-2.5 border-b"
                    style={{
                      borderColor: "var(--ds-border)",
                      background: "var(--ds-bg-subtle)",
                    }}
                  >
                    <div className="h-2.5 w-2.5 rounded-full bg-red-400" />
                    <div className="h-2.5 w-2.5 rounded-full bg-amber-400" />
                    <div className="h-2.5 w-2.5 rounded-full bg-green-400" />
                    <div
                      className="ml-3 flex-1 h-5 rounded text-[10px] flex items-center px-2"
                      style={{ background: "var(--ds-bg-page)", color: "var(--ds-text-muted)" }}
                    >
                      deepresearch.app
                    </div>
                  </div>

                  {/* Mockup content */}
                  <div className="p-4 sm:p-5">
                    {slides[current].mockup}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* ── CTA slide (3) ── */
            <div className="flex flex-col items-center text-center max-w-2xl mx-auto hiw-text-enter">
              <h1
                className="text-3xl sm:text-4xl lg:text-5xl font-bold leading-tight mb-5"
                style={{
                  fontFamily: "var(--font-heading)",
                  color: "var(--ds-text-heading)",
                }}
              >
                {t("ctaTitle")}
              </h1>
              <p
                className="text-base sm:text-lg leading-relaxed mb-8"
                style={{ color: "var(--ds-text-body)" }}
              >
                {t("ctaDescription")}
              </p>
              <Link
                href="/register"
                className="inline-flex items-center rounded-lg px-8 py-3.5 text-base font-medium text-white transition-colors"
                style={{ background: "var(--ds-primary)" }}
                onMouseEnter={(e) =>
                  (e.currentTarget.style.background = "var(--ds-primary-hover)")
                }
                onMouseLeave={(e) =>
                  (e.currentTarget.style.background = "var(--ds-primary)")
                }
              >
                {t("ctaButton")}
              </Link>
            </div>
          )}
        </div>
      </main>

      {/* ─── Dot navigation (fixed bottom) ─── */}
      <div className="fixed bottom-6 left-0 right-0 z-40 flex items-center justify-center gap-2">
        {Array.from({ length: TOTAL_SLIDES }).map((_, i) => (
          <button
            key={i}
            onClick={() => goTo(i)}
            className="h-2.5 rounded-full transition-all duration-300"
            style={{
              width: i === current ? "2rem" : "0.625rem",
              background: i === current ? "var(--ds-primary)" : "var(--ds-border-strong)",
            }}
            aria-label={`Go to slide ${i + 1}`}
          />
        ))}
      </div>

      {/* ─── CSS keyframe animations ─── */}
      <style jsx>{`
        .hiw-fade-in {
          animation: hiwFadeIn 600ms ease both;
        }
        .hiw-text-enter {
          animation: hiwTextEnter 400ms ease both;
        }
        .hiw-mockup-enter {
          animation: hiwMockupEnter 600ms ease both;
        }

        @keyframes hiwFadeIn {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }

        @keyframes hiwTextEnter {
          from {
            opacity: 0;
            transform: translateX(-30px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }

        @keyframes hiwMockupEnter {
          from {
            opacity: 0;
            transform: translateX(30px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }
      `}</style>
    </div>
  );
}
