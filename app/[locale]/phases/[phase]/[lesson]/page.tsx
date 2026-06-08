import { notFound } from "next/navigation";
import { getTranslations, setRequestLocale } from "next-intl/server";
import { ChevronLeft, Clock, Code2, Target, Construction } from "lucide-react";
import { Link } from "@/i18n/navigation";
import {
  getLessonContent,
  getPhase,
  getPhases,
  lessonKey,
} from "@/lib/content";
import { Badge } from "@/components/ui/badge";
import { LessonMarkdown } from "@/components/lesson/markdown";
import { Toc } from "@/components/lesson/toc";
import { Quiz } from "@/components/lesson/quiz";
import { LessonActions } from "@/components/lesson/lesson-actions";

export async function generateStaticParams() {
  const phases = await getPhases();
  const params: { phase: string; lesson: string }[] = [];
  for (const p of phases) {
    for (const l of p.lessons) {
      if (l.hasContent) params.push({ phase: p.slug, lesson: l.slug });
    }
  }
  return params;
}

export default async function LessonPage({
  params,
}: {
  params: Promise<{ locale: string; phase: string; lesson: string }>;
}) {
  const { locale, phase: phaseSlug, lesson: lessonSlug } = await params;
  setRequestLocale(locale);

  const content = await getLessonContent(phaseSlug, lessonSlug);
  const t = await getTranslations("lesson");
  const tc = await getTranslations("common");
  const tq = await getTranslations("quiz");

  // Lesson has no written content yet → "coming soon" screen.
  if (!content) {
    const phase = await getPhase(phaseSlug);
    if (!phase) notFound();
    const tp = await getTranslations("phase");
    return (
      <div className="mx-auto max-w-2xl px-4 py-24 text-center sm:px-6">
        <div className="mx-auto flex size-14 items-center justify-center rounded-full bg-accent-soft text-accent">
          <Construction className="size-7" />
        </div>
        <h1 className="mt-6 font-serif text-3xl font-semibold text-primary">
          {tp("comingSoon")}
        </h1>
        <p className="mt-3 text-muted-foreground">{tp("lessonLocked")}</p>
        <Link
          href={`/phases/${phase.slug}`}
          className="mt-8 inline-flex items-center gap-1.5 text-sm font-medium text-accent hover:underline"
        >
          <ChevronLeft className="size-4" />
          {t("backToPhase")}
        </Link>
      </div>
    );
  }

  const { phase, lesson, markdown, meta, quiz, headings } = content;

  const idx = phase.lessons.findIndex((l) => l.id === lesson.id);
  const prevRef = idx > 0 ? phase.lessons[idx - 1] : null;
  const nextRef =
    idx < phase.lessons.length - 1 ? phase.lessons[idx + 1] : null;
  const prev = prevRef
    ? { href: `/phases/${phase.slug}/${prevRef.slug}`, title: prevRef.title }
    : null;
  const next = nextRef
    ? { href: `/phases/${phase.slug}/${nextRef.slug}`, title: nextRef.title }
    : null;

  const key = lessonKey(phase.slug, lesson.slug);

  return (
    <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6">
      {/* Breadcrumb */}
      <nav className="flex flex-wrap items-center gap-1.5 text-sm text-muted-foreground">
        <Link href="/curriculum" className="hover:text-foreground">
          {t("breadcrumbCurriculum")}
        </Link>
        <span aria-hidden>/</span>
        <Link href={`/phases/${phase.slug}`} className="hover:text-foreground">
          {phase.title}
        </Link>
      </nav>

      <div className="mt-6 lg:grid lg:grid-cols-[minmax(0,1fr)_15rem] lg:gap-12">
        <article className="min-w-0">
          {/* Header */}
          <header className="border-b border-border pb-7">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="navy">
                {tc("phase")} {String(phase.phaseNumber).padStart(2, "0")}
              </Badge>
              {meta.type && <Badge variant="accent">{meta.type}</Badge>}
            </div>
            <h1 className="mt-3 font-serif text-4xl font-semibold leading-tight text-primary">
              {meta.title || lesson.title}
            </h1>

            <div className="mt-4 flex flex-wrap gap-x-6 gap-y-2 text-sm text-muted-foreground">
              {meta.timeEstimate && (
                <span className="inline-flex items-center gap-1.5">
                  <Clock className="size-4" />
                  {meta.timeEstimate}
                </span>
              )}
              {meta.languages?.length > 0 && (
                <span className="inline-flex items-center gap-1.5">
                  <Code2 className="size-4" />
                  {meta.languages.join(" · ")}
                </span>
              )}
            </div>

            {meta.objectives?.length > 0 && (
              <div className="mt-5 rounded-lg border border-border bg-subtle/60 p-4">
                <div className="flex items-center gap-2 text-sm font-medium text-primary">
                  <Target className="size-4" />
                  {t("objectives")}
                </div>
                <ul className="mt-2.5 space-y-1.5 text-sm text-muted-foreground">
                  {meta.objectives.map((o, i) => (
                    <li key={i} className="flex gap-2">
                      <span className="mt-0.5 text-accent">›</span>
                      <span>{o}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </header>

          {/* Lesson body */}
          <div className="prose-lesson mt-8">
            <LessonMarkdown>{markdown}</LessonMarkdown>
          </div>

          {/* Quiz */}
          {quiz && quiz.questions.length > 0 && (
            <section className="mt-14">
              <div className="mb-4">
                <h2 className="font-serif text-2xl font-semibold text-primary">
                  {tq("title")}
                </h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  {tq("subtitle")}
                </p>
              </div>
              <Quiz questions={quiz.questions} lessonKey={key} />
            </section>
          )}

          <LessonActions lessonKey={key} prev={prev} next={next} />
        </article>

        {/* TOC */}
        <aside className="hidden lg:block">
          <div className="sticky top-24">
            <Toc headings={headings} />
          </div>
        </aside>
      </div>
    </div>
  );
}
