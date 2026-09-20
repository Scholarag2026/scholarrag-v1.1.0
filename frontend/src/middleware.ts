import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

export function middleware(request: NextRequest) {
  // This app defines ZERO Server Actions ("use server" has 0 hits in src/ and in git history),
  // so any POST carrying a Next-Action header is scanner traffic. Next's action handler throws
  // "Failed to find Server Action" for these before app code runs, which buries >99% of the
  // frontend log stream. Reject them at the edge instead.
  if (request.method === "POST" && request.headers.has("next-action")) {
    return new NextResponse(null, { status: 404 });
  }

  const response = NextResponse.next();

  // Set the locale cookie from Accept-Language if not already set
  if (!request.cookies.get("locale")) {
    const acceptLang = request.headers.get("accept-language") || "";
    const locale = acceptLang.startsWith("zh") ? "zh" : "en";
    response.cookies.set("locale", locale, { path: "/", sameSite: "lax" });
  }

  return response;
}

export const config = {
  matcher: "/((?!api|trpc|_next|_vercel|.*\\..*).*)",
};
