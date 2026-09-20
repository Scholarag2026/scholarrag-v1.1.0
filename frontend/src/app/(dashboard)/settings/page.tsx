"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { useAuthStore } from "@/lib/auth";
import { apiFetch } from "@/lib/api";
import { toast } from "sonner";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Loader2, Save } from "lucide-react";

export default function SettingsPage() {
  const t = useTranslations("accountSettings");
  const tLang = useTranslations("language");
  const user = useAuthStore((s) => s.user);
  const fetchMe = useAuthStore((s) => s.fetchMe);

  const [name, setName] = useState(user?.name || "");
  const [expertiseLevel, setExpertiseLevel] = useState(user?.expertise_level || "student");
  const [preferredLanguage, setPreferredLanguage] = useState(user?.preferred_language || "en");
  const [saving, setSaving] = useState(false);

  if (!user) return null;

  async function handleSave() {
    setSaving(true);
    try {
      await apiFetch("/auth/me", {
        method: "PUT",
        body: JSON.stringify({
          name,
          expertise_level: expertiseLevel,
          preferred_language: preferredLanguage,
        }),
      });
      await fetchMe();
      toast.success(t("saved"));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("saveFailed"));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="p-6 md:p-8">
      <div className="mb-8">
        <h1
          className="text-2xl font-bold text-[var(--ds-text-heading)]"
          style={{ fontFamily: "var(--font-heading)" }}
        >
          {t("title")}
        </h1>
        <p
          className="mt-1 text-sm text-[var(--ds-text-secondary)]"
          style={{ fontFamily: "var(--font-body)" }}
        >
          {t("subtitle")}
        </p>
      </div>

      <div className="max-w-2xl space-y-6">
        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardHeader>
            <CardTitle
              className="text-[var(--ds-text-heading)]"
              style={{ fontFamily: "var(--font-heading)" }}
            >
              {t("profile")}
            </CardTitle>
            <CardDescription
              className="text-[var(--ds-text-secondary)]"
              style={{ fontFamily: "var(--font-body)" }}
            >
              {t("profileDesc")}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label className="text-[var(--ds-text-heading)]">{t("name")}</Label>
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)]"
                style={{ fontFamily: "var(--font-body)" }}
              />
            </div>
            <div className="space-y-2">
              <Label className="text-[var(--ds-text-heading)]">{t("email")}</Label>
              <Input
                value={user.email}
                readOnly
                className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-muted)]"
                style={{ fontFamily: "var(--font-body)" }}
              />
              <p className="text-xs text-[var(--ds-text-muted)]">{t("emailReadonly")}</p>
            </div>
          </CardContent>
        </Card>

        <Separator className="bg-[var(--ds-border)]" />

        <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
          <CardHeader>
            <CardTitle
              className="text-[var(--ds-text-heading)]"
              style={{ fontFamily: "var(--font-heading)" }}
            >
              {t("preferences")}
            </CardTitle>
            <CardDescription
              className="text-[var(--ds-text-secondary)]"
              style={{ fontFamily: "var(--font-body)" }}
            >
              {t("preferencesDesc")}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label className="text-[var(--ds-text-heading)]">{t("expertiseLevel")}</Label>
              <p className="text-xs text-[var(--ds-text-muted)]">{t("expertiseLevelHelp")}</p>
              <select
                value={expertiseLevel}
                onChange={(e) => setExpertiseLevel(e.target.value)}
                className="flex h-9 w-full rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-page)] px-3 text-sm text-[var(--ds-text-heading)] outline-none focus:border-[var(--ds-primary)] focus:ring-1 focus:ring-[var(--ds-primary)]"
                style={{ fontFamily: "var(--font-body)" }}
              >
                <option value="student">{t("student")}</option>
                <option value="researcher">{t("researcher")}</option>
                <option value="faculty">{t("faculty")}</option>
              </select>
              <div className="rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-subtle)] px-3 py-2 text-xs text-[var(--ds-text-secondary)]">
                {expertiseLevel === "student" && t("expertiseHintStudent")}
                {expertiseLevel === "researcher" && t("expertiseHintResearcher")}
                {expertiseLevel === "faculty" && t("expertiseHintFaculty")}
              </div>
            </div>
            <div className="space-y-2">
              <Label className="text-[var(--ds-text-heading)]">{t("preferredLanguage")}</Label>
              <select
                value={preferredLanguage}
                onChange={(e) => setPreferredLanguage(e.target.value)}
                className="flex h-9 w-full rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-page)] px-3 text-sm text-[var(--ds-text-heading)] outline-none focus:border-[var(--ds-primary)] focus:ring-1 focus:ring-[var(--ds-primary)]"
                style={{ fontFamily: "var(--font-body)" }}
              >
                <option value="en">{tLang("en")}</option>
                <option value="zh">{tLang("zh")}</option>
              </select>
            </div>
          </CardContent>
        </Card>

        <div className="flex justify-end">
          <Button
            onClick={handleSave}
            disabled={saving}
            className="gap-2 bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
          >
            {saving ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <Save className="size-4" />
            )}
            {t("saveChanges")}
          </Button>
        </div>
      </div>
    </div>
  );
}
