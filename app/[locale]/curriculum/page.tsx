import { getTranslations, setRequestLocale } from "next-intl/server";
import { ArrowRight, BookOpen, Compass } from "lucide-react";
import { Link } from "@/i18n/navigation";
import { getCurriculum, getTotalLessons } from "@/lib/content";
import { Badge } from "@/components/ui/badge";

export default async function CurriculumPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);

  const t = await getTranslations("curriculum");
  const tc = await getTranslations("common");
  const curriculum = await getCurriculum();
  const total = await getTotalLessons();

  return (
    <div className="mx-auto max-w-4xl px-4 py-14 sm:px-6 sm:py-16">
      <header className="max-w-2xl">
        <h1 className="font-serif text-4xl font-semibold text-primary">
          {t("title")}
        </h1>
        <p className="mt-3 text-lg text-muted-foreground">
          {t("subtitle", { count: total.toLocaleString() })}
        </p>
      </header>

      {/* Level guide */}
      <div className="mt-8 flex gap-4 rounded-xl border border-border bg-accent-soft/60 p-5">
        <Compass className="size-5 shrink-0 text-accent" />
        <div>
          <div className="font-medium text-primary">{t("yourLevel")}</div>
          <p className="mt-1 text-sm text-muted-foreground">
            {t("yourLevelBody")}
          </p>
        </div>
      </div>

      {/* Phases */}
      <div className="mt-10 space-y-4">
        {curriculum.phases.map((p) => (
          <Link
            key={p.id}
            href={`/phases/${p.slug}`}
            className="group block rounded-xl border border-border bg-card p-5 transition-all hover:border-border-strong hover:shadow-[var(--shadow-card)] sm:p-6"
          >
            <div className="flex items-start gap-5">
              <div className="flex size-12 shrink-0 items-center justify-center rounded-lg bg-primary font-serif text-lg font-semibold text-primary-foreground tabular">
                {String(p.phaseNumber).padStart(2, "0")}
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
                  <h2 className="text-lg font-semibold text-primary">
                    {p.title}
                  </h2>
                  <Badge variant="success">{t("preview")}</Badge>
                </div>
                <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">
                  {p.description}
                </p>
                <div className="mt-3 inline-flex items-center gap-1.5 text-xs text-muted-foreground">
                  <BookOpen className="size-3.5" />
                  {p.lessonCount} {tc("lessons")}
                </div>
              </div>
              <ArrowRight className="mt-1 size-5 shrink-0 text-muted-foreground/40 transition-all group-hover:translate-x-0.5 group-hover:text-accent" />
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
