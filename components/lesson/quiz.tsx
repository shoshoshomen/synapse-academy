"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { Check, X, RotateCcw, Award, ChevronRight } from "lucide-react";
import { useProgress } from "@/lib/progress";
import { siteConfig } from "@/config/site";
import { cn, pct } from "@/lib/utils";
import type { QuizQuestion } from "@/types/content";

export function Quiz({
  questions,
  lessonKey,
}: {
  questions: QuizQuestion[];
  lessonKey: string;
}) {
  const t = useTranslations("quiz");
  const { saveQuiz } = useProgress();
  const threshold = siteConfig.passThreshold;

  const [index, setIndex] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);
  const [revealed, setRevealed] = useState(false);
  const [correct, setCorrect] = useState(0);
  const [finished, setFinished] = useState(false);

  const total = questions.length;
  const q = questions[index];
  const isLast = index === total - 1;

  function check() {
    if (selected === null || revealed) return;
    setRevealed(true);
    if (selected === q.correct) setCorrect((c) => c + 1);
  }

  function advance() {
    if (isLast) {
      const passed = correct / total >= threshold;
      saveQuiz(lessonKey, { correct, total, passed, at: Date.now() });
      setFinished(true);
    } else {
      setIndex((i) => i + 1);
      setSelected(null);
      setRevealed(false);
    }
  }

  function retry() {
    setIndex(0);
    setSelected(null);
    setRevealed(false);
    setCorrect(0);
    setFinished(false);
  }

  if (finished) {
    const passed = correct / total >= threshold;
    const scorePct = pct(correct / total);
    return (
      <div
        id="quiz"
        className="rounded-xl border border-border bg-card p-8 text-center"
      >
        <div
          className={cn(
            "mx-auto flex size-14 items-center justify-center rounded-full",
            passed
              ? "bg-success-soft text-success"
              : "bg-danger-soft text-danger",
          )}
        >
          {passed ? (
            <Award className="size-7" />
          ) : (
            <RotateCcw className="size-7" />
          )}
        </div>
        <div className="mt-4 font-serif text-4xl font-semibold text-primary tabular">
          {scorePct}%
        </div>
        <p className="mt-1 text-sm text-muted-foreground">
          {t("scoreLine", { correct, total })}
        </p>
        <p
          className={cn(
            "mt-3 font-medium",
            passed ? "text-success" : "text-danger",
          )}
        >
          {passed ? t("passed") : t("failed")}
        </p>
        <button
          type="button"
          onClick={retry}
          className="mt-6 inline-flex h-10 items-center gap-2 rounded-md border border-border-strong bg-card px-4 text-sm font-medium text-foreground transition-colors hover:bg-subtle"
        >
          <RotateCcw className="size-4" />
          {t("retry")}
        </button>
      </div>
    );
  }

  return (
    <div id="quiz" className="rounded-xl border border-border bg-card p-6 sm:p-8">
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium text-primary">{t("title")}</span>
        <span className="tabular text-muted-foreground">
          {t("question")} {index + 1} {t("of")} {total}
        </span>
      </div>
      <div className="mt-3 h-1 w-full overflow-hidden rounded-full bg-muted">
        <div
          className="h-full bg-accent transition-all duration-300"
          style={{ width: `${pct((index + (revealed ? 1 : 0)) / total)}%` }}
        />
      </div>

      <h3 className="mt-5 text-lg font-medium text-foreground">{q.question}</h3>

      <div className="mt-4 space-y-2.5">
        {q.options.map((opt, i) => {
          const isCorrect = i === q.correct;
          const isSelected = i === selected;
          let state: "idle" | "selected" | "correct" | "wrong" = "idle";
          if (revealed) {
            if (isCorrect) state = "correct";
            else if (isSelected) state = "wrong";
          } else if (isSelected) {
            state = "selected";
          }
          return (
            <button
              key={i}
              type="button"
              disabled={revealed}
              onClick={() => setSelected(i)}
              className={cn(
                "flex w-full items-center gap-3 rounded-lg border px-4 py-3 text-left text-sm transition-colors",
                state === "idle" &&
                  "border-border bg-background hover:border-border-strong",
                state === "selected" && "border-primary bg-primary/5",
                state === "correct" && "border-success bg-success-soft",
                state === "wrong" && "border-danger bg-danger-soft",
              )}
            >
              <span
                className={cn(
                  "flex size-5 shrink-0 items-center justify-center rounded-full border text-xs font-semibold",
                  state === "correct" && "border-success bg-success text-white",
                  state === "wrong" && "border-danger bg-danger text-white",
                  (state === "idle" || state === "selected") &&
                    "border-border-strong text-muted-foreground",
                )}
              >
                {revealed && isCorrect ? (
                  <Check className="size-3" />
                ) : revealed && isSelected ? (
                  <X className="size-3" />
                ) : (
                  String.fromCharCode(65 + i)
                )}
              </span>
              <span
                className={cn(
                  state === "correct" && "text-success",
                  state === "wrong" && "text-danger",
                )}
              >
                {opt}
              </span>
            </button>
          );
        })}
      </div>

      {revealed && (
        <div className="mt-4 rounded-lg bg-subtle p-4 text-sm">
          <div className="mb-1 font-medium text-primary">
            {t("explanation")}
          </div>
          <p className="leading-relaxed text-muted-foreground">
            {q.explanation}
          </p>
        </div>
      )}

      <div className="mt-6 flex justify-end">
        {!revealed ? (
          <button
            type="button"
            onClick={check}
            disabled={selected === null}
            className="inline-flex h-11 items-center gap-2 rounded-md bg-primary px-5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary-hover disabled:opacity-50"
          >
            {t("submit")}
          </button>
        ) : (
          <button
            type="button"
            onClick={advance}
            className="inline-flex h-11 items-center gap-2 rounded-md bg-primary px-5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary-hover"
          >
            {isLast ? t("finish") : t("next")}
            <ChevronRight className="size-4" />
          </button>
        )}
      </div>
    </div>
  );
}
