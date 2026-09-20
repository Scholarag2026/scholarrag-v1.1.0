const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

export class ApiError extends Error {
  status: number;
  code?: string;

  constructor(status: number, message: string, code?: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

let isRefreshing = false;
let refreshPromise: Promise<boolean> | null = null;

export async function tryRefreshToken(): Promise<boolean> {
  const refreshToken = localStorage.getItem("refresh_token");
  if (!refreshToken) return false;

  try {
    const res = await fetch(`${API_URL}/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
    if (!res.ok) return false;
    const data = await res.json();
    localStorage.setItem("access_token", data.access_token);
    return true;
  } catch {
    return false;
  }
}

export async function apiFetch<T = unknown>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const token =
    typeof window !== "undefined"
      ? localStorage.getItem("access_token")
      : null;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...((options.headers as Record<string, string>) || {}),
  };

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_URL}${path}`, { ...options, headers });

  if (res.status === 401 && typeof window !== "undefined") {
    const hadToken = !!token;

    if (hadToken) {
      // Try to refresh the token (deduplicate concurrent refresh attempts)
      if (!isRefreshing) {
        isRefreshing = true;
        refreshPromise = tryRefreshToken().finally(() => {
          isRefreshing = false;
          refreshPromise = null;
        });
      }

      const refreshed = await (refreshPromise || tryRefreshToken());

      if (refreshed) {
        // Retry the original request with the new token
        const newToken = localStorage.getItem("access_token");
        headers["Authorization"] = `Bearer ${newToken}`;
        const retryRes = await fetch(`${API_URL}${path}`, { ...options, headers });

        if (retryRes.ok) {
          return retryRes.json();
        }
      }

      // Refresh failed — clear tokens and redirect
      localStorage.removeItem("access_token");
      localStorage.removeItem("refresh_token");
      window.location.href = "/login";
    }

    throw new ApiError(401, "Unauthorized");
  }

  if (!res.ok) {
    const error = await res
      .json()
      .catch(() => ({ detail: res.statusText }));
    let detail = error.detail ?? error.message ?? res.statusText;
    if (Array.isArray(detail)) {
      detail = detail.map((d: { msg?: string }) => d.msg || JSON.stringify(d)).join("; ");
    }
    throw new ApiError(res.status, detail, error.code);
  }

  return res.json();
}

export async function fetchWithAuth(
  url: string,
  options: RequestInit = {},
): Promise<Response> {
  const token =
    typeof window !== "undefined"
      ? localStorage.getItem("access_token")
      : null;

  const headers: Record<string, string> = {
    ...((options.headers as Record<string, string>) || {}),
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(url, { ...options, headers });

  if (res.status === 401 && typeof window !== "undefined" && token) {
    const refreshed = await tryRefreshToken();
    if (refreshed) {
      const newToken = localStorage.getItem("access_token");
      headers["Authorization"] = `Bearer ${newToken}`;
      return fetch(url, { ...options, headers });
    }
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    window.location.href = "/login";
  }

  return res;
}

/**
 * Cancel a running background job (DELETE /api/v1/tasks/{task_id}).
 *
 * Status-code based on purpose so it is robust to the endpoint returning either a JSON body or
 * 204 No Content: any 2xx means "cancelled", 404 means the job no longer exists and 409 means it
 * already reached a terminal state — in all three cases there is nothing left to cancel.
 */
export async function cancelTask(taskId: string): Promise<void> {
  const res = await fetchWithAuth(`${API_URL}/tasks/${taskId}`, { method: "DELETE" });

  if (res.ok || res.status === 404 || res.status === 409) return;

  const body = await res
    .json()
    .catch(() => ({ detail: res.statusText } as { detail?: string; code?: string }));
  throw new ApiError(res.status, body.detail ?? res.statusText, body.code);
}
