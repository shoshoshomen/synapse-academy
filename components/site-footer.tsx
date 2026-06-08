import { getTranslations } from "next-intl/server";
import { Link } from "@/i18n/navigation";
import { siteConfig } from "@/config/site";

export async function SiteFooter() {
  const t = await getTranslations("footer");

  return (
    <footer className="border-t border-border bg-card">
      <div className="mx-auto grid max-w-6xl gap-8 px-4 py-12 sm:px-6 md:grid-cols-[2fr_1fr]">
        <div>
          <div className="font-serif text-lg font-semibold text-primary">
            {siteConfig.name}
          </div>
          <p className="mt-2 max-w-xs text-sm text-muted-foreground">
            {t("tagline")} {t("rights")}
          </p>
        </div>

        <div className="text-sm">
          <div className="mb-3 font-medium text-foreground">
            {t("curriculum")}
          </div>
          <ul className="space-y-2 text-muted-foreground">
            <li>
              <Link href="/curriculum" className="hover:text-foreground">
                {t("curriculum")}
              </Link>
            </li>
            <li>
              <Link href="/dashboard" className="hover:text-foreground">
                {t("progress")}
              </Link>
            </li>
          </ul>
        </div>
      </div>

      <div className="border-t border-border">
        <div className="mx-auto max-w-6xl px-4 py-5 text-xs text-muted-foreground sm:px-6">
          <span>{t("builtWith")}</span>
        </div>
      </div>
    </footer>
  );
}
