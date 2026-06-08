"use client";

import { useLocale } from "next-intl";
import { usePathname, useRouter } from "@/i18n/navigation";
import { useTransition } from "react";
import { cn } from "@/lib/utils";

const locales = [
  { code: "en", label: "EN" },
  { code: "ja", label: "日本語" },
] as const;

export function LanguageSwitcher() {
  const locale = useLocale();
  const pathname = usePathname();
  const router = useRouter();
  const [pending, startTransition] = useTransition();

  return (
    <div className="inline-flex items-center rounded-full border border-border bg-card p-0.5 text-xs font-medium">
      {locales.map((l) => (
        <button
          key={l.code}
          type="button"
          disabled={pending}
          onClick={() =>
            startTransition(() =>
              router.replace(pathname, { locale: l.code }),
            )
          }
          aria-current={locale === l.code}
          className={cn(
            "rounded-full px-2.5 py-1 transition-colors",
            locale === l.code
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          {l.label}
        </button>
      ))}
    </div>
  );
}
