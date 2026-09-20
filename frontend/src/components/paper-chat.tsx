"use client";

import { useState, useRef, useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { Send, Loader2, BookOpen, User, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState } from "@/components/empty-state";
import { apiFetch } from "@/lib/api";
import { useProjectPapers } from "@/hooks/use-papers";

interface PaperChatProps {
  projectId: string;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export function PaperChat({ projectId }: PaperChatProps) {
  const t = useTranslations("paperChat");
  const tEmpty = useTranslations("emptyStates");
  const { data: papersData } = useProjectPapers(projectId);
  const paperCount = papersData?.total || 0;
  const queryClient = useQueryClient();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { data: history } = useQuery<ChatMessage[]>({
    queryKey: ["chat-history", projectId],
    queryFn: () => apiFetch(`/projects/${projectId}/chat/history`),
  });

  useEffect(() => {
    if (history && messages.length === 0) {
      setMessages(history.map((m) => ({ role: m.role, content: m.content })));
    }
  }, [history]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleClearHistory() {
    await apiFetch(`/projects/${projectId}/chat/history`, { method: "DELETE" });
    setMessages([]);
    queryClient.invalidateQueries({ queryKey: ["chat-history", projectId] });
  }

  async function handleSend() {
    if (!input.trim() || isLoading) return;

    const question = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setIsLoading(true);

    try {
      const response = await apiFetch<{ answer: string }>(`/projects/${projectId}/chat`, {
        method: "POST",
        body: JSON.stringify({ question }),
      });
      setMessages((prev) => [...prev, { role: "assistant", content: response.answer }]);
      queryClient.invalidateQueries({ queryKey: ["chat-history", projectId] });
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: t("chatError") },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  if (paperCount === 0) {
    return (
      <EmptyState
        icon="💬"
        title={tEmpty("paperChat.title")}
        description={tEmpty("paperChat.description")}
        prerequisites={[
          { label: tEmpty("prerequisites.addPapersFirst"), completed: false, current: 0, total: 1 },
        ]}
        actionLabel={tEmpty("prerequisites.addPapersFirst")}
        onAction={() => {}}
        actionDisabled={true}
      />
    );
  }

  return (
    <Card className="border-[var(--ds-border)] bg-[var(--ds-bg-card)]">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle
            className="flex items-center gap-2 text-base text-[var(--ds-text-heading)]"
            style={{ fontFamily: "var(--font-heading)" }}
          >
            <BookOpen className="size-4 text-[var(--ds-primary)]" />
            {t("chatTitle", { count: paperCount })}
          </CardTitle>
          {messages.length > 0 && (
            <Button
              variant="ghost"
              size="icon"
              onClick={handleClearHistory}
              className="size-8 text-[var(--ds-text-secondary)] hover:text-[var(--ds-error)] hover:bg-[var(--ds-error)]/10"
              title={t("clearHistory")}
            >
              <Trash2 className="size-4" />
            </Button>
          )}
        </div>
        <p className="text-sm text-[var(--ds-text-secondary)]" style={{ fontFamily: "var(--font-body)" }}>
          {t("chatDescription")}
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Messages */}
        <div className="h-[500px] overflow-y-auto rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-page)] p-4 space-y-4">
          {messages.length === 0 && (
            <div className="flex h-full items-center justify-center">
              <div className="text-center space-y-2">
                <BookOpen className="mx-auto size-8 text-[var(--ds-border)]" />
                <p className="text-sm text-[var(--ds-text-muted)]">
                  {t("askPlaceholder")}
                </p>
                <div className="flex flex-wrap justify-center gap-2 mt-3">
                  {[
                    t("suggestion1"),
                    t("suggestion2"),
                    t("suggestion3"),
                    t("suggestion4"),
                  ].map((suggestion) => (
                    <button
                      key={suggestion}
                      onClick={() => { setInput(suggestion); }}
                      className="rounded-full border border-[var(--ds-border)] bg-[var(--ds-bg-card)] px-3 py-1.5 text-xs text-[var(--ds-text-secondary)] hover:border-[var(--ds-primary)] hover:text-[var(--ds-text-heading)] transition-colors"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              {msg.role === "assistant" && (
                <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-[var(--ds-primary-light)]">
                  <BookOpen className="size-3.5 text-[var(--ds-primary)]" />
                </div>
              )}
              <div
                className={`max-w-[85%] rounded-lg px-4 py-3 text-sm ${
                  msg.role === "user"
                    ? "bg-[var(--ds-primary)] text-white"
                    : "bg-[var(--ds-bg-card)] text-[var(--ds-text-heading)] border border-[var(--ds-border)]"
                }`}
                style={{ fontFamily: "var(--font-body)", whiteSpace: "pre-wrap" }}
              >
                {msg.content}
              </div>
              {msg.role === "user" && (
                <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-[rgba(5,150,105,0.08)]">
                  <User className="size-3.5 text-[var(--ds-success)]" />
                </div>
              )}
            </div>
          ))}

          {isLoading && (
            <div className="flex gap-3">
              <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-[var(--ds-primary-light)]">
                <BookOpen className="size-3.5 text-[var(--ds-primary)]" />
              </div>
              <div className="rounded-lg bg-[var(--ds-bg-card)] border border-[var(--ds-border)] px-4 py-3">
                <Loader2 className="size-4 animate-spin text-[var(--ds-primary)]" />
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={t("askPlaceholder")}
            className="border-[var(--ds-border)] bg-[var(--ds-bg-page)] text-[var(--ds-text-heading)] placeholder:text-[var(--ds-text-muted)]"
            style={{ fontFamily: "var(--font-body)" }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            disabled={isLoading}
          />
          <Button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className="shrink-0 bg-[var(--ds-primary)] text-white hover:bg-[var(--ds-primary-hover)]"
          >
            {isLoading ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <Send className="size-4" />
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
