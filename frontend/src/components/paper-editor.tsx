"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useEditor, EditorContent } from "@tiptap/react";
import { BubbleMenu } from "@tiptap/react/menus";
import { Sparkles, Loader2, Check, X } from "lucide-react";
import StarterKit from "@tiptap/starter-kit";
import Underline from "@tiptap/extension-underline";
import Placeholder from "@tiptap/extension-placeholder";
import { useTranslations } from "next-intl";

import { EditorToolbar } from "@/components/editor-toolbar";
import {
  OriginParagraph,
  type CitationLinkAttr,
} from "@/components/tiptap-extensions/origin-paragraph";
import { useRefine } from "@/hooks/use-refine";
import { useTaskPolling } from "@/hooks/use-task-polling";

/** Threshold (in characters) above which an AI block is promoted to "user". */
const PROMOTE_THRESHOLD = 10;

/** Collapse runs of whitespace so a sentence lookup ignores formatting-only differences. */
function normalizeWhitespace(text: string): string {
  return text.split(/\s+/).filter(Boolean).join(" ");
}

/**
 * Drop every citation link whose ``sentence`` is no longer present in the paragraph's
 * current text: invalidation is per sentence and deterministic, since a stored index has
 * no way to notice an insertion above it. This is housekeeping only -- the backend
 * repeats the same check (authoritative) when the draft is verified, so a draft edited
 * through the API behaves identically even if this never ran.
 */
function pruneStaleCitationLinks(
  links: CitationLinkAttr[],
  currentText: string,
): CitationLinkAttr[] {
  const normalizedText = normalizeWhitespace(currentText);
  return links.filter((link) =>
    normalizedText.includes(normalizeWhitespace(link.sentence)),
  );
}

interface PaperEditorProps {
  content: Record<string, unknown> | null;
  onSave: (content: Record<string, unknown>) => void;
  saving?: boolean;
  draftId?: string;
}

export function PaperEditor({ content, onSave, saving, draftId }: PaperEditorProps) {
  const t = useTranslations("editor");
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // --- Selection-level refine state ---
  const refine = useRefine();
  const [refineTaskId, setRefineTaskId] = useState<string | null>(null);
  const [refineResult, setRefineResult] = useState<{ original: string; refined: string } | null>(null);
  const refineRangeRef = useRef<{ from: number; to: number } | null>(null);

  // Poll for refine task result — bounded, backed off.
  const { data: refineTaskData } = useTaskPolling<{
    status: string;
    result: { original: string; refined: string } | null;
  }>({ taskId: refineTaskId, kind: "refine" });

  // Handle refine completion
  useEffect(() => {
    if (!refineTaskData || !refineTaskId) return;
    if (refineTaskData.status === "completed" && refineTaskData.result) {
      setRefineResult(refineTaskData.result);
      setRefineTaskId(null);
    } else if (refineTaskData.status === "failed") {
      setRefineTaskId(null);
    }
  }, [refineTaskData, refineTaskId]);

  /** Saved text per blockId – used for auto-promotion comparison. */
  const blockTextRef = useRef<Map<string, string>>(new Map());
  /** Set of blockIds we've already seen – used to detect new paragraphs. */
  const knownBlocksRef = useRef<Set<string>>(new Set());

  /**
   * Walk the document and:
   * 1. Auto-promote AI paragraphs whose text changed by > PROMOTE_THRESHOLD chars.
   * 2. Tag newly created paragraphs (unseen blockId with content) as "user".
   */
  const handleOriginTracking = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (editor: any) => {
      const { doc, tr } = editor.state;
      let dispatched = false;

      doc.descendants((node: { type: { name: string }; attrs: Record<string, unknown>; textContent: string }, pos: number) => {
        if (node.type.name !== "paragraph") return;

        const blockId = node.attrs.blockId as string | null;
        if (!blockId) return;

        const currentText = node.textContent;
        const origin = (node.attrs.origin as string) || "ai";
        let nextAttrs: Record<string, unknown> | null = null;

        // --- New paragraph tagging ---
        if (!knownBlocksRef.current.has(blockId)) {
          knownBlocksRef.current.add(blockId);
          // If the paragraph already has content, the user is typing in a new block
          if (currentText.length > 0 && origin === "ai") {
            nextAttrs = { ...node.attrs, origin: "user" };
          }
          blockTextRef.current.set(blockId, currentText);
        } else {
          // --- Auto-promotion of AI blocks ---
          if (origin === "ai") {
            const savedText = blockTextRef.current.get(blockId) ?? "";
            const diff = Math.abs(currentText.length - savedText.length);
            if (diff > PROMOTE_THRESHOLD) {
              nextAttrs = { ...node.attrs, origin: "user" };
            }
          }
          // Always keep the snapshot up to date
          blockTextRef.current.set(blockId, currentText);
        }

        // --- Prune citation links whose sentence no longer appears in the text ---
        const existingLinks = (node.attrs.citationLinks as CitationLinkAttr[] | null) || [];
        if (existingLinks.length > 0) {
          const pruned = pruneStaleCitationLinks(existingLinks, currentText);
          if (pruned.length !== existingLinks.length) {
            nextAttrs = { ...(nextAttrs || node.attrs), citationLinks: pruned };
          }
        }

        if (nextAttrs) {
          tr.setNodeMarkup(pos, undefined, nextAttrs);
          dispatched = true;
        }
      });

      if (dispatched) {
        editor.view.dispatch(tr);
      }
    },
    [],
  );

  /** Seed the tracking refs from the current document so existing blocks
   *  are not mistakenly promoted or tagged as new. */
  const seedTrackingRefs = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (doc: any) => {
      blockTextRef.current.clear();
      knownBlocksRef.current.clear();
      doc.descendants((node: { type: { name: string }; attrs: Record<string, unknown>; textContent: string }) => {
        if (node.type.name !== "paragraph") return;
        const blockId = node.attrs.blockId as string | null;
        if (!blockId) return;
        knownBlocksRef.current.add(blockId);
        blockTextRef.current.set(blockId, node.textContent);
      });
    },
    [],
  );

  const editor = useEditor({
    immediatelyRender: false,
    extensions: [
      StarterKit.configure({
        heading: { levels: [1, 2, 3, 4] },
        paragraph: false,
      }),
      OriginParagraph,
      Underline,
      Placeholder.configure({
        placeholder: t("placeholder"),
      }),
    ],
    content: content || undefined,
    editorProps: {
      attributes: {
        class:
          "prose max-w-none min-h-[500px] px-8 py-6 outline-none focus:outline-none",
      },
    },
    onCreate: ({ editor }) => {
      // Seed tracking refs with the initial document state
      seedTrackingRefs(editor.state.doc);
    },
    onUpdate: ({ editor }) => {
      // Origin-tracking logic
      handleOriginTracking(editor);

      // Debounced save
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
      saveTimerRef.current = setTimeout(() => {
        onSave(editor.getJSON());
      }, 2000);
    },
  });

  useEffect(() => {
    return () => {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    };
  }, []);

  useEffect(() => {
    if (editor && content) {
      const currentJSON = JSON.stringify(editor.getJSON());
      const newJSON = JSON.stringify(content);
      if (currentJSON !== newJSON) {
        editor.commands.setContent(content);
        // Re-seed tracking refs after replacing content
        seedTrackingRefs(editor.state.doc);
      }
    }
  }, [editor, content, seedTrackingRefs]);

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <EditorToolbar editor={editor} />
        {saving && (
          <span
            className="text-xs text-[var(--ds-text-secondary)]"
            style={{ fontFamily: "var(--font-body)" }}
          >
            {t("saving")}
          </span>
        )}
      </div>
      <div className="rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-page)]">
        <EditorContent editor={editor} />
        {editor && (
          <BubbleMenu editor={editor}>
            <div className="flex items-center gap-1 rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-card)] p-1 shadow-lg">
              <button
                className="flex items-center gap-1 rounded px-2 py-1 text-xs text-[var(--ds-text-secondary)] hover:bg-[var(--ds-bg-hover)] hover:text-[var(--ds-text-heading)] disabled:opacity-50"
                onClick={() => {
                  if (!draftId) return;
                  const { from, to } = editor.state.selection;
                  const selectedText = editor.state.doc.textBetween(from, to, " ");
                  if (!selectedText.trim()) return;
                  refineRangeRef.current = { from, to };
                  refine.mutate(
                    { draftId, text: selectedText, scope: "selection" },
                    {
                      onSuccess: (data) => setRefineTaskId(data.task_id),
                    },
                  );
                }}
                disabled={refine.isPending || !!refineTaskId || !draftId}
              >
                {refine.isPending || !!refineTaskId ? (
                  <Loader2 className="size-3 animate-spin" />
                ) : (
                  <Sparkles className="size-3" />
                )}
                Refine
              </button>
            </div>
          </BubbleMenu>
        )}
      </div>

      {refineResult && (
        <div className="mt-3 rounded-lg border border-[var(--ds-border)] bg-[var(--ds-bg-card)] p-4">
          <h4 className="mb-3 text-sm font-medium text-[var(--ds-text-heading)]">Refined Text</h4>
          <div className="mb-2 rounded bg-red-500/10 p-3 text-sm text-red-300 line-through">
            {refineResult.original}
          </div>
          <div className="mb-3 rounded bg-green-500/10 p-3 text-sm text-green-300">
            {refineResult.refined}
          </div>
          <div className="flex justify-end gap-2">
            <button
              className="flex items-center gap-1 rounded px-3 py-1.5 text-sm text-[var(--ds-text-secondary)] hover:bg-[var(--ds-bg-hover)]"
              onClick={() => { setRefineResult(null); refineRangeRef.current = null; }}
            >
              <X className="size-3.5" />
              Reject
            </button>
            <button
              className="flex items-center gap-1 rounded bg-[var(--ds-primary)] px-3 py-1.5 text-sm text-white hover:bg-[var(--ds-primary-hover)]"
              onClick={() => {
                const range = refineRangeRef.current || editor!.state.selection;
                editor?.chain().focus().deleteRange({ from: range.from, to: range.to }).insertContent(refineResult.refined).run();
                setRefineResult(null);
                refineRangeRef.current = null;
              }}
            >
              <Check className="size-3.5" />
              Accept
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
