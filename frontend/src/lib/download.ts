/**
 * Browser download helpers. The DOM and URL objects are injectable so the logic can be
 * unit-tested outside a browser.
 */

interface AnchorLike {
  href: string;
  download: string;
  click: () => void;
  remove: () => void;
}

interface DocumentLike {
  createElement: (tag: "a") => AnchorLike;
  body: { appendChild: (el: AnchorLike) => unknown };
}

interface UrlLike {
  createObjectURL: (blob: Blob) => string;
  revokeObjectURL: (url: string) => void;
}

export interface DownloadDeps {
  doc?: DocumentLike;
  url?: UrlLike;
}

export function downloadBlob(blob: Blob, filename: string, deps: DownloadDeps = {}): void {
  const doc = deps.doc ?? (document as unknown as DocumentLike);
  const url = deps.url ?? (URL as unknown as UrlLike);

  const objectUrl = url.createObjectURL(blob);
  const anchor = doc.createElement("a");
  anchor.href = objectUrl;
  anchor.download = filename;
  doc.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  url.revokeObjectURL(objectUrl);
}

export function screeningRecordFilename(taskId: string): string {
  return `screening-record-${taskId}.csv`;
}

export function claimRecordFilename(taskId: string): string {
  return `claim-record-${taskId}.csv`;
}
