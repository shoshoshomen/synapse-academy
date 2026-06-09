import { getTranslations, setRequestLocale } from "next-intl/server";
import { Info } from "lucide-react";
import { Link } from "@/i18n/navigation";
import { buttonVariants } from "@/components/ui/button";
import { hasDatabase } from "@/lib/db";
import { SignInForm } from "@/components/auth/sign-in-form";
import { cn } from "@/lib/utils";

export default async function SignInPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  const t = await getTranslations("auth");

  return (
    <div className="mx-auto max-w-md px-4 py-20 sm:px-6">
      <div className="rounded-xl border border-border bg-card p-8 shadow-[var(--shadow-card)]">
        <h1 className="text-center font-serif text-2xl font-semibold text-primary">
          {t("signInTitle")}
        </h1>
        <p className="mt-2 text-center text-sm text-muted-foreground">
          {t("signInSubtitle")}
        </p>

        {hasDatabase ? (
          <SignInForm googleEnabled={Boolean(process.env.AUTH_GOOGLE_ID)} />
        ) : (
          <>
            <div className="mt-6 flex items-start gap-3 rounded-lg bg-subtle p-4 text-sm">
              <Info className="mt-0.5 size-4 shrink-0 text-accent" />
              <p className="text-muted-foreground">{t("dbDisabledNotice")}</p>
            </div>
            <Link
              href="/curriculum"
              className={cn(buttonVariants(), "mt-6 w-full")}
            >
              {t("continueAsGuest")}
            </Link>
            <p className="mt-3 text-center text-xs text-muted-foreground">
              {t("guestExplain")}
            </p>
          </>
        )}
      </div>
    </div>
  );
}
