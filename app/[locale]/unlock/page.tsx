import { getTranslations, setRequestLocale } from "next-intl/server";
import { Lock } from "lucide-react";
import { UnlockForm } from "@/components/auth/unlock-form";

export default async function UnlockPage({
  params,
  searchParams,
}: {
  params: Promise<{ locale: string }>;
  searchParams: Promise<{ from?: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  const { from } = await searchParams;
  const t = await getTranslations("unlock");

  return (
    <div className="mx-auto max-w-md px-4 py-20 sm:px-6">
      <div className="rounded-xl border border-border bg-card p-8 shadow-[var(--shadow-card)]">
        <div className="mx-auto flex size-12 items-center justify-center rounded-full bg-accent-soft text-accent">
          <Lock className="size-6" />
        </div>
        <h1 className="mt-5 text-center font-serif text-2xl font-semibold text-primary">
          {t("title")}
        </h1>
        <p className="mt-2 text-center text-sm text-muted-foreground">
          {t("subtitle")}
        </p>
        <UnlockForm from={from ?? ""} />
      </div>
    </div>
  );
}
