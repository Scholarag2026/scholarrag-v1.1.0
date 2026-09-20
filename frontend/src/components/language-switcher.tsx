"use client";

import { useState, useRef, useEffect, useTransition } from "react";
import { useRouter } from "next/navigation";
import { useLocale } from "next-intl";
import { ChevronDown, Globe } from "lucide-react";

const locales = [
  { code: "en", label: "English" },
  { code: "zh", label: "中文" },
] as const;

export function LanguageSwitcher() {
  const currentLocale = useLocale();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const router = useRouter();
  const [isPending, startTransition] = useTransition();

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  function switchLocale(locale: string) {
    // next-intl runs here WITHOUT i18n routing (no [locale] segment, no next-intl middleware),
    // so the locale source is this cookie and we update it ourselves. router.refresh() re-runs
    // the root server layout — which reads the cookie via getLocale()/getMessages() — while
    // keeping the React Query cache and all client state alive. A full document reload used to
    // throw the whole cache away on every toggle.
    document.cookie = `locale=${locale};path=/;max-age=31536000;samesite=lax`;
    setOpen(false);
    startTransition(() => {
      router.refresh();
    });
  }

  const current = locales.find((l) => l.code === currentLocale) || locales[0];

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        disabled={isPending}
        className="flex items-center gap-1.5 rounded-md px-2 py-1.5 text-sm text-[var(--ds-text-secondary)] transition-colors hover:bg-[var(--ds-bg-hover)] hover:text-[var(--ds-text-heading)] disabled:opacity-60"
        style={{ fontFamily: "var(--font-body)" }}
      >
        <Globe className="size-4" />
        {current.label}
        <ChevronDown className={`size-3 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div className="absolute top-full left-0 mt-1 w-32 rounded-md border border-[var(--ds-border)] bg-[var(--ds-bg-card)] py-1 shadow-lg z-50">
          {locales.map((locale) => (
            <button
              key={locale.code}
              onClick={() => switchLocale(locale.code)}
              className={`flex w-full items-center px-3 py-2 text-sm transition-colors ${
                locale.code === currentLocale
                  ? "bg-[var(--ds-primary-light)] text-[var(--ds-primary)]"
                  : "text-[var(--ds-text-secondary)] hover:bg-[var(--ds-bg-hover)] hover:text-[var(--ds-text-heading)]"
              }`}
              style={{ fontFamily: "var(--font-body)" }}
            >
              {locale.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
