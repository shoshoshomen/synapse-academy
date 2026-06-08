"use client";

import { useTranslations } from "next-intl";
import { Check, Circle, ArrowRight, Trophy, Info } from "lucide-react";
import { Link } from "@/i18n/navigation";
import { useProgress } from "@/lib/progress";
import { ProgressBar } from "@/components/ui/progress";
import { cn, pct } from "@/lib/utils";

export interface DashboardLesson {
  key: string;
  href: string;
  phaseNumber: number;
  phaseTitle: string;
  lessonTitle: string;
}

export function DashboardClient({ lessons }: { lessons: DashboardLesson[] }) {
  const t = useTranslations("dashboard");
  const { state, isComplete, reset } = useProgress();

  const total = lessons.length;
  const completed = lessons.filter((l) => isComplete(l.key)).length;
  const overall = pct(completed / Math.max(1, total));

  const quizValues = Object.values(state.quizzes);
  const passed = quizValues.filter((q) => q.passed).length;
  const avg = quizValues.length
    ? Math.round(
        (quizValues.reduce((s, q) => s + q.correct / q.total, 0) /
          quizValues.length) *
          100,
      )
    : 0;

  const nextLesson = lessons.find((l) => !isComplete(l.key)) ?? null;

  return (
    <div className="mx-auto max-w-3xl px-4 py-14 sm:px-6">
      <h1 className="font-serif text-4xl font-semibold text-primary">
        {t("title")}
      </h1>

      {/* Guest notice */}
      <div className="mt-6 flex items-start gap-3 rounded-lg border border-border bg-accent-soft/50 p-4 text-sm">
        <Info className="mt-0.5 size-4 shrink-0 text-accent" />
        <div>
          <p className="text-muted-foreground">{t("guestNotice")}</p>
          <Link
            href="/sign-in"
            className="mt-1 inline-block font-medium text-accent hover:underline"
          >
            {t("guestCta")} →
          </Link>
        </div>
      </div>

      {/* Overall progress */}
      <div className="mt-8 rounded-xl border border-border bg-card p-6">
        <div className="flex items-baseline justify-between">
          <span className="text-sm font-medium text-muted-foreground">
            {t("overall")}
          </span>
          <span className="font-serif text-2xl font-semibold text-primary tabular">
            {overall}%
          </span>
        </div>
        <ProgressBar value={overall} className="mt-3" />
        <p className="mt-2 text-xs text-muted-foreground">
          {t("lessonsCompleted", { completed, total })}
        </p>
      </div>

      {/* Stat cards */}
      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <div className="rounded-xl border border-border bg-card p-5">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Trophy className="size-4 text-accent" />
            {t("quizzesPassed")}
          </div>
          <div className="mt-2 font-serif text-3xl font-semibold text-primary tabular">
            {passed}
          </div>
        </div>
        <div className="rounded-xl border border-border bg-card p-5">
          <div className="text-sm text-muted-foreground">
            {t("averageScore")}
          </div>
          <div className="mt-2 font-serif text-3xl font-semibold text-primary tabular">
            {avg}%
          </div>
        </div>
      </div>

      {/* Continue */}
      {nextLesson && (
        <Link
          href={nextLesson.href}
          className="group mt-6 flex items-center gap-4 rounded-xl border border-primary/20 bg-primary p-5 text-primary-foreground transition-colors hover:bg-primary-hover"
        >
          <div className="min-w-0 flex-1">
            <div className="text-xs uppercase tracking-wider text-primary-foreground/60">
              {t("continue")}
            </div>
            <div className="mt-1 truncate font-medium">
              {nextLesson.lessonTitle}
            </div>
          </div>
          <ArrowRight className="size-5 shrink-0 transition-transform group-hover:translate-x-0.5" />
        </Link>
      )}

      {/* Phase-by-phase */}
      <h2 className="mt-10 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
        {t("phaseProgress")}
      </h2>
      <ul className="mt-4 overflow-hidden rounded-xl border border-border">
        {lessons.map((l, i) => {
          const done = isComplete(l.key);
          const quiz = state.quizzes[l.key];
          return (
            <li key={l.key}>
              <Link
                href={l.href}
                className={cn(
                  "flex items-center gap-3 bg-card px-4 py-3 transition-colors hover:bg-subtle sm:px-5",
                  i !== 0 && "border-t border-border",
                )}
              >
                {done ? (
                  <Check className="size-4 shrink-0 text-success" />
                ) : (
                  <Circle className="size-4 shrink-0 text-muted-foreground/40" />
                )}
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm font-medium text-foreground">
                    {l.lessonTitle}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {String(l.phaseNumber).padStart(2, "0")} · {l.phaseTitle}
                  </span>
                </span>
                {quiz && (
                  <span
                    className={cn(
                      "shrink-0 rounded-full px-2 py-0.5 text-xs font-medium tabular",
                      quiz.passed
                        ? "bg-success-soft text-success"
                        : "bg-danger-soft text-danger",
                    )}
                  >
                    {pct(quiz.correct / quiz.total)}%
                  </span>
                )}
              </Link>
            </li>
          );
        })}
      </ul>

      {(completed > 0 || quizValues.length > 0) && (
        <div className="mt-6 text-center">
          <button
            type="button"
            onClick={reset}
            className="text-xs text-muted-foreground underline-offset-2 hover:text-danger hover:underline"
          >
            {t("resetProgress")}
          </button>
        </div>
      )}
    </div>
  );
}
