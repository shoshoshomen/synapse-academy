import { defineRouting } from "next-intl/routing";

export const routing = defineRouting({
  locales: ["en", "ja"],
  defaultLocale: "en",
  // Prefix the default locale too, so URLs are always /en/... or /ja/...
  localePrefix: "always",
});

export type Locale = (typeof routing.locales)[number];
