import { getTranslations, setRequestLocale } from "next-intl/server";
import {
  ArrowRight,
  Hammer,
  Package,
  GraduationCap,
  Languages,
  Sparkles,
} from "lucide-react";
import { Link } from "@/i18n/navigation";
import { getCurriculum, getTotalLessons } from "@/lib/content";
import { siteConfig } from "@/config/site";
import { buttonVariants } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

const START_HREF = "/phases/setup-and-tooling/dev-environment";

export default async function HomePage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);

  const t = await getTranslations("home");
  const tc = await getTranslations("common");
  const curriculum = await getCurriculum();
  const total = await getTotalLessons();

  const title = t("title");
  const accent = t("titleAccent");
  const [beforeAccent, afterAccent] = title.split(accent);

  const stats = [
    { value: total.toLocaleString(), label: t("statLessons") },
    { value: String(curriculum.phases.length), label: t("statPhases") },
    { value: `${siteConfig.stats.hours}+`, label: t("statHours") },
    { value: String(siteConfig.stats.languages), label: t("statLanguages") },
  ];

  const features = [
    { icon: Hammer, title: t("feature1Title"), body: t("feature1Body") },
    { icon: Package, title: t("feature2Title"), body: t("feature2Body") },
    {
      icon: GraduationCap,
      title: t("feature3Title"),
      body: t("feature3Body"),
    },
    { icon: Languages, title: t("feature4Title"), body: t("feature4Body") },
  ];

  return (
    <>
      {/* Hero */}
      <section className="relative overflow-hidden border-b border-border">
        <div className="absolute inset-0 bg-grid" aria-hidden />
        <div className="relative mx-auto max-w-5xl px-4 py-24 text-center sm:px-6 sm:py-32">
          <Badge variant="accent" className="mb-6">
            <Sparkles className="size-3.5" />
            {t("badge")}
          </Badge>
          <h1 className="mx-auto max-w-3xl font-serif text-4xl font-semibold leading-[1.08] text-primary sm:text-6xl">
            {beforeAccent}
            <span className="italic text-accent">{accent}</span>
            {afterAccent}
          </h1>
          <p className="mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-muted-foreground">
            {t("subtitle")}
          </p>
          <div className="mt-9 flex flex-col items-center justify-center gap-3 sm:flex-row">
            <Link
              href={START_HREF}
              className={cn(buttonVariants({ size: "lg" }), "group")}
            >
              {t("ctaStart")}
              <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" />
            </Link>
            <Link
              href="/curriculum"
              className={buttonVariants({ variant: "outline", size: "lg" })}
            >
              {t("ctaCurriculum")}
            </Link>
          </div>
        </div>
      </section>

      {/* Stats */}
      <section className="border-b border-border bg-card">
        <div className="mx-auto grid max-w-5xl grid-cols-2 divide-x divide-y divide-border sm:grid-cols-4 sm:divide-y-0">
          {stats.map((s) => (
            <div key={s.label} className="px-6 py-8 text-center">
              <div className="font-serif text-3xl font-semibold text-primary tabular">
                {s.value}
              </div>
              <div className="mt-1 text-xs uppercase tracking-wider text-muted-foreground">
                {s.label}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="mx-auto max-w-5xl px-4 py-20 sm:px-6 sm:py-24">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="font-serif text-3xl font-semibold text-primary sm:text-4xl">
            {t("featuresTitle")}
          </h2>
          <p className="mt-4 text-muted-foreground">{t("featuresSubtitle")}</p>
        </div>
        <div className="mt-14 grid gap-6 sm:grid-cols-2">
          {features.map((f) => (
            <div
              key={f.title}
              className="rounded-xl border border-border bg-card p-7 shadow-[var(--shadow-card)]"
            >
              <div className="flex size-11 items-center justify-center rounded-lg bg-accent-soft text-accent">
                <f.icon className="size-5" />
              </div>
              <h3 className="mt-5 text-lg font-semibold text-primary">
                {f.title}
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                {f.body}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Path / phases */}
      <section className="border-t border-border bg-subtle/50">
        <div className="mx-auto max-w-5xl px-4 py-20 sm:px-6 sm:py-24">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="font-serif text-3xl font-semibold text-primary sm:text-4xl">
              {t("pathTitle")}
            </h2>
            <p className="mt-4 text-muted-foreground">{t("pathSubtitle")}</p>
          </div>
          <ol className="mt-12 grid gap-px overflow-hidden rounded-xl border border-border bg-border sm:grid-cols-2">
            {curriculum.phases.map((p) => (
              <li key={p.id}>
                <Link
                  href={`/phases/${p.slug}`}
                  className="group flex h-full items-center gap-4 bg-card p-5 transition-colors hover:bg-subtle"
                >
                  <span className="font-serif text-2xl font-semibold text-border-strong tabular">
                    {String(p.phaseNumber).padStart(2, "0")}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate font-medium text-foreground">
                      {p.title}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {p.lessonCount} {tc("lessons")}
                    </span>
                  </span>
                  <ArrowRight className="size-4 shrink-0 text-muted-foreground/50 transition-transform group-hover:translate-x-0.5 group-hover:text-accent" />
                </Link>
              </li>
            ))}
          </ol>
        </div>
      </section>

      {/* CTA */}
      <section className="border-t border-border bg-primary">
        <div className="mx-auto max-w-4xl px-4 py-20 text-center sm:px-6">
          <h2 className="font-serif text-3xl font-semibold !text-primary-foreground sm:text-4xl">
            {t("ctaSectionTitle")}
          </h2>
          <p className="mx-auto mt-4 max-w-xl text-primary-foreground/75">
            {t("ctaSectionBody")}
          </p>
          <Link
            href={START_HREF}
            className={cn(buttonVariants({ variant: "accent", size: "lg" }), "mt-8 group")}
          >
            {t("ctaSectionButton")}
            <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" />
          </Link>
        </div>
      </section>
    </>
  );
}
