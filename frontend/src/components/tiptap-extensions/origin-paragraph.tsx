import Paragraph from "@tiptap/extension-paragraph";
import {
  NodeViewContent,
  NodeViewWrapper,
  ReactNodeViewRenderer,
} from "@tiptap/react";
import type { NodeViewProps } from "@tiptap/core";

/**
 * One sentence-to-reference entry the writer's citation-link map contributes to a
 * paragraph. Stored on the paragraph node itself, beside ``origin`` and ``blockId``, so
 * it travels with the paragraph through reordering, section replacement, draft versions
 * and the existing ``PUT /drafts/{id}`` contract -- a sibling JSON field keyed by
 * paragraph index would rot the moment a section is replaced.
 */
export interface CitationLinkAttr {
  sentence: string;
  keys: string[];
  citation_text: string;
}

function OriginParagraphView({ node, updateAttributes }: NodeViewProps) {
  const origin = (node.attrs.origin as string) || "ai";
  return (
    <NodeViewWrapper
      as="div"
      className={`origin-block origin-${origin}`}
      data-origin={origin}
      data-block-id={node.attrs.blockId}
    >
      <button
        className="origin-indicator"
        contentEditable={false}
        onClick={() =>
          updateAttributes({ origin: origin === "ai" ? "user" : "ai" })
        }
        title={
          origin === "ai"
            ? "AI generated (click to mark as yours)"
            : "User written (click to mark as AI)"
        }
      >
        {origin === "ai" ? "AI" : "You"}
      </button>
      <NodeViewContent<"p"> as="p" />
    </NodeViewWrapper>
  );
}

export const OriginParagraph = Paragraph.extend({
  addAttributes() {
    return {
      ...this.parent?.(),
      origin: {
        default: "ai",
        parseHTML: (element: HTMLElement) =>
          element.getAttribute("data-origin") || "ai",
        renderHTML: (attributes: Record<string, unknown>) => ({
          "data-origin": attributes.origin as string,
        }),
      },
      blockId: {
        default: null,
        parseHTML: (element: HTMLElement) =>
          element.getAttribute("data-block-id"),
        renderHTML: (attributes: Record<string, unknown>) => {
          const id = attributes.blockId || crypto.randomUUID();
          return { "data-block-id": id };
        },
      },
      citationLinks: {
        default: [],
        parseHTML: (element: HTMLElement) => {
          const raw = element.getAttribute("data-citation-links");
          if (!raw) return [];
          try {
            const parsed = JSON.parse(raw);
            return Array.isArray(parsed) ? parsed : [];
          } catch {
            return [];
          }
        },
        renderHTML: (attributes: Record<string, unknown>) => {
          const links = Array.isArray(attributes.citationLinks)
            ? (attributes.citationLinks as CitationLinkAttr[])
            : [];
          if (links.length === 0) return {};
          return { "data-citation-links": JSON.stringify(links) };
        },
      },
    };
  },

  addNodeView() {
    return ReactNodeViewRenderer(OriginParagraphView);
  },
});
