"use client";

import { useTranslations } from "next-intl";
import { Check, Circle, ChevronLeft, ChevronRight } from "lucide-react";
import { Link } from "@/i18n/navigation";
import { useProgress } from "@/lib/progress";
import { cn } from "@/lib/utils";

interface NavLink {
  href: string;
  title: string;
}

export function LessonActions({
  lessonKey,
  prev,
  next,
}: {
  lessonKey: string;
  prev: NavLink | null;
  next: NavLink | null;
}) {
  const t = useTranslations("lesson");
  const { isComplete, toggleComplete } = useProgress();
  const done = isComplete(lessonKey);

  return (
    <div className="mt-12 border-t border-border pt-8">
      <div className="flex justify-center">
        <button
          type="button"
          onClick={() => toggleComplete(lessonKey)}
          className={cn(
            "inline-flex h-11 items-center gap-2 rounded-md border px-5 text-sm font-medium transition-colors",
            done
              ? "border-success bg-success-soft text-success"
              : "border-border-strong bg-card text-foreground hover:bg-subtle",
          )}
          aria-pressed={done}
        >
          {done ? <Check className="size-4" /> : <Circle className="size-4" />}
          {done ? t("completed") : t("markComplete")}
        </button>
      </div>

      {(prev || next) && (
        <div className="mt-8 grid gap-3 sm:grid-cols-2">
          {prev ? (
            <Link
              href={prev.href}
              className="group flex items-center gap-3 rounded-lg border border-border bg-card p-4 transition-colors hover:bg-subtle"
            >
              <ChevronLeft className="size-5 shrink-0 text-muted-foreground" />
              <span className="min-w-0">
                <span className="block text-xs text-muted-foreground">
                  {t("prevLesson")}
                </span>
                <span className="block truncate font-medium text-foreground">
                  {prev.title}
                </span>
              </span>
            </Link>
          ) : (
            <span />
          )}
          {next && (
            <Link
              href={next.href}
              className="group flex items-center gap-3 rounded-lg border border-border bg-card p-4 text-right transition-colors hover:bg-subtle sm:justify-end"
            >
              <span className="min-w-0">
                <span className="block text-xs text-muted-foreground">
                  {t("nextLesson")}
                </span>
                <span className="block truncate font-medium text-foreground">
                  {next.title}
                </span>
              </span>
              <ChevronRight className="size-5 shrink-0 text-accent" />
            </Link>
          )}
        </div>
      )}
    </div>
  );
}
