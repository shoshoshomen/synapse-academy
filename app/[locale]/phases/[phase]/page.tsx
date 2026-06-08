import { notFound } from "next/navigation";
import { getTranslations, setRequestLocale } from "next-intl/server";
import { ArrowRight, ChevronLeft, Lock } from "lucide-react";
import { Link } from "@/i18n/navigation";
import { getPhase, getPhases } from "@/lib/content";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export async function generateStaticParams() {
  const phases = await getPhases();
  return phases.map((p) => ({ phase: p.slug }));
}

export default async function PhasePage({
  params,
}: {
  params: Promise<{ locale: string; phase: string }>;
}) {
  const { locale, phase: phaseSlug } = await params;
  setRequestLocale(locale);

  const phase = await getPhase(phaseSlug, locale);
  if (!phase) notFound();

  const t = await getTranslations("phase");
  const tc = await getTranslations("common");

  return (
    <div className="mx-auto max-w-3xl px-4 py-12 sm:px-6 sm:py-14">
      <Link
        href="/curriculum"
        className="inline-flex items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground"
      >
        <ChevronLeft className="size-4" />
        {t("backToCurriculum")}
      </Link>

      <header className="mt-6 border-b border-border pb-8">
        <div className="text-sm font-medium uppercase tracking-wider text-accent">
          {tc("phase")} {String(phase.phaseNumber).padStart(2, "0")}
        </div>
        <h1 className="mt-2 font-serif text-4xl font-semibold text-primary">
          {phase.title}
        </h1>
        <p className="mt-3 text-lg text-muted-foreground">
          {phase.description}
        </p>
        <div className="mt-4 text-sm text-muted-foreground">
          {phase.lessonCount} {tc("lessons")}
        </div>
      </header>

      <h2 className="mt-8 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
        {t("lessonsInPhase")}
      </h2>

      <ol className="mt-4 overflow-hidden rounded-xl border border-border">
        {phase.lessons.map((l, i) => {
          const numberBadge = (
            <span className="flex size-8 shrink-0 items-center justify-center rounded-full border border-border bg-subtle text-xs font-semibold text-muted-foreground tabular">
              {l.lessonNumber}
            </span>
          );
          const base = cn(
            "flex items-center gap-4 px-4 py-3.5 sm:px-5",
            i !== 0 && "border-t border-border",
          );

          if (l.hasContent) {
            return (
              <li key={l.id}>
                <Link
                  href={`/phases/${phase.slug}/${l.slug}`}
                  className={cn(base, "group bg-card transition-colors hover:bg-subtle")}
                >
                  {numberBadge}
                  <span className="min-w-0 flex-1 font-medium text-foreground">
                    {l.title}
                  </span>
                  <Badge variant="success" className="hidden sm:inline-flex">
                    {t("previewAvailable")}
                  </Badge>
                  <ArrowRight className="size-4 shrink-0 text-muted-foreground/50 transition-transform group-hover:translate-x-0.5 group-hover:text-accent" />
                </Link>
              </li>
            );
          }

          return (
            <li key={l.id}>
              <div className={cn(base, "bg-card/60")}>
                {numberBadge}
                <span className="min-w-0 flex-1 text-muted-foreground">
                  {l.title}
                </span>
                <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground/70">
                  <Lock className="size-3" />
                  <span className="hidden sm:inline">{t("comingSoon")}</span>
                </span>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
