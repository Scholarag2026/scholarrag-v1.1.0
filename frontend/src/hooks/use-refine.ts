import { useMutation } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";

export function useRefine() {
  return useMutation<
    { task_id: string },
    Error,
    {
      draftId: string;
      text: string;
      context?: string;
      scope: "selection" | "section";
      sectionKey?: string;
    }
  >({
    mutationFn: ({ draftId, ...body }) =>
      apiFetch(`/drafts/${draftId}/refine`, {
        method: "POST",
        body: JSON.stringify({
          text: body.text,
          context: body.context,
          scope: body.scope,
          section_key: body.sectionKey,
        }),
      }),
  });
}
