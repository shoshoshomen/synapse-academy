import { getTranslations } from "next-intl/server";
import { Link } from "@/i18n/navigation";
import { siteConfig } from "@/config/site";
import { LanguageSwitcher } from "./language-switcher";
import { buttonVariants } from "./ui/button";
import { cn } from "@/lib/utils";

function Logo() {
  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
      className="text-accent"
    >
      <path
        d="M6.6 7.2 17 11M6.6 16.8 17 13"
        stroke="currentColor"
        strokeWidth="1.4"
      />
      <circle cx="5" cy="6.4" r="2.1" fill="currentColor" />
      <circle cx="5" cy="17.6" r="2.1" fill="currentColor" />
      <circle cx="18.6" cy="12" r="2.6" fill="currentColor" />
    </svg>
  );
}

export async function SiteHeader() {
  const t = await getTranslations("nav");

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-background/85 backdrop-blur supports-[backdrop-filter]:bg-background/70">
      <div className="mx-auto flex h-16 max-w-6xl items-center gap-6 px-4 sm:px-6">
        <Link href="/" className="flex items-center gap-2.5">
          <Logo />
          <span className="font-serif text-lg font-semibold tracking-tight text-primary">
            {siteConfig.name}
          </span>
        </Link>

        <nav className="hidden items-center gap-6 text-sm md:flex">
          <Link
            href="/curriculum"
            className="text-muted-foreground transition-colors hover:text-foreground"
          >
            {t("curriculum")}
          </Link>
          <Link
            href="/dashboard"
            className="text-muted-foreground transition-colors hover:text-foreground"
          >
            {t("dashboard")}
          </Link>
        </nav>

        <div className="ml-auto flex items-center gap-3">
          <LanguageSwitcher />
          <Link
            href="/sign-in"
            className={cn(
              buttonVariants({ variant: "outline", size: "sm" }),
              "hidden sm:inline-flex",
            )}
          >
            {t("signIn")}
          </Link>
        </div>
      </div>
    </header>
  );
}
