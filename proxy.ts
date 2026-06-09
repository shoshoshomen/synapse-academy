import createMiddleware from "next-intl/middleware";
import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { routing } from "./i18n/routing";
import { GATE_COOKIE, gateToken, isGateEnabled, isLessonPath } from "./lib/gate";

const intlProxy = createMiddleware(routing);

export default async function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Gate lesson bodies behind the shared password; everything else (home,
  // curriculum, phase index) stays public so the school is discoverable.
  if (isGateEnabled() && isLessonPath(pathname)) {
    const presented = request.cookies.get(GATE_COOKIE)?.value;
    const expected = await gateToken();
    if (!expected || presented !== expected) {
      const locale =
        pathname.split("/").filter(Boolean)[0] || routing.defaultLocale;
      const url = request.nextUrl.clone();
      url.pathname = `/${locale}/unlock`;
      url.search = "";
      url.searchParams.set("from", pathname);
      return NextResponse.redirect(url);
    }
  }

  return intlProxy(request);
}

export const config = {
  // Match all pathnames except for
  // - API routes
  // - Next.js internals (_next, _vercel)
  // - static files (with a dot, e.g. favicon.ico)
  matcher: ["/((?!api|_next|_vercel|.*\\..*).*)"],
};
