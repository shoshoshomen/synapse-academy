import "server-only";
import { promises as fs } from "fs";
import path from "path";
import GithubSlugger from "github-slugger";
import type {
  Curriculum,
  Phase,
  LessonRef,
  LessonMeta,
  Quiz,
  Heading,
  LessonContent,
} from "@/types/content";

const CONTENT_DIR = path.join(process.cwd(), "content");

let curriculumCache: Curriculum | null = null;

export async function getCurriculum(): Promise<Curriculum> {
  if (curriculumCache) return curriculumCache;
  const raw = await fs.readFile(
    path.join(CONTENT_DIR, "curriculum.json"),
    "utf8",
  );
  curriculumCache = JSON.parse(raw) as Curriculum;
  return curriculumCache;
}

export async function getPhases(): Promise<Phase[]> {
  return (await getCurriculum()).phases;
}

export async function getPhase(
  idOrSlug: string,
): Promise<Phase | undefined> {
  const phases = await getPhases();
  return phases.find(
    (p) =>
      p.slug === idOrSlug ||
      p.id === idOrSlug ||
      String(p.phaseNumber) === idOrSlug ||
      String(p.phaseNumber).padStart(2, "0") === idOrSlug,
  );
}

export function findLesson(
  phase: Phase,
  lessonSlugOrId: string,
): LessonRef | undefined {
  return phase.lessons.find(
    (l) => l.slug === lessonSlugOrId || l.id === lessonSlugOrId,
  );
}

/** Total number of lessons across all phases. */
export async function getTotalLessons(): Promise<number> {
  const phases = await getPhases();
  return phases.reduce((sum, p) => sum + p.lessonCount, 0);
}

/**
 * The set of lessons that actually have written content right now
 * (one preview lesson per phase). Used for progress denominators and
 * "available now" badges.
 */
export async function getPreviewLessons(): Promise<
  { phase: Phase; lesson: LessonRef }[]
> {
  const phases = await getPhases();
  const out: { phase: Phase; lesson: LessonRef }[] = [];
  for (const phase of phases) {
    const lesson = phase.lessons.find((l) => l.hasContent);
    if (lesson) out.push({ phase, lesson });
  }
  return out;
}

/** Stable key used for progress storage + URLs: "<phaseSlug>/<lessonSlug>". */
export function lessonKey(phaseSlug: string, lessonSlug: string): string {
  return `${phaseSlug}/${lessonSlug}`;
}

function extractHeadings(markdown: string): Heading[] {
  const slugger = new GithubSlugger();
  const headings: Heading[] = [];
  const lines = markdown.split("\n");
  let inFence = false;
  for (const line of lines) {
    if (/^\s*```/.test(line)) {
      inFence = !inFence;
      continue;
    }
    if (inFence) continue;
    const m = /^(#{2,3})\s+(.+?)\s*#*\s*$/.exec(line);
    if (m) {
      const level = m[1].length as 2 | 3;
      const text = m[2].replace(/[*_`]/g, "").trim();
      headings.push({ id: slugger.slug(text), text, level });
    }
  }
  return headings;
}

/**
 * Load the full content for a lesson. Returns null if the lesson has no
 * written material yet (i.e. "coming soon").
 */
export async function getLessonContent(
  phaseSlugOrId: string,
  lessonSlugOrId: string,
): Promise<LessonContent | null> {
  const phase = await getPhase(phaseSlugOrId);
  if (!phase) return null;
  const lesson = findLesson(phase, lessonSlugOrId);
  if (!lesson) return null;

  const dir = path.join(CONTENT_DIR, "phases", phase.id, lesson.id);

  let markdown: string;
  let meta: LessonMeta;
  try {
    markdown = await fs.readFile(path.join(dir, "lesson.md"), "utf8");
    const metaRaw = await fs.readFile(path.join(dir, "meta.json"), "utf8");
    meta = JSON.parse(metaRaw) as LessonMeta;
  } catch {
    return null; // no content yet
  }

  // quiz.json comes in two shapes upstream: a top-level array of questions,
  // or an object { questions: [...] }. Normalize both, drop malformed items.
  let quiz: Quiz | null = null;
  try {
    const quizRaw = await fs.readFile(path.join(dir, "quiz.json"), "utf8");
    const parsed: unknown = JSON.parse(quizRaw);
    const rawQuestions: unknown[] = Array.isArray(parsed)
      ? parsed
      : Array.isArray((parsed as { questions?: unknown[] })?.questions)
        ? (parsed as { questions: unknown[] }).questions
        : [];
    const questions = rawQuestions.filter(
      (q): q is import("@/types/content").QuizQuestion =>
        !!q &&
        typeof (q as { question?: unknown }).question === "string" &&
        Array.isArray((q as { options?: unknown }).options) &&
        typeof (q as { correct?: unknown }).correct === "number",
    );
    quiz = questions.length ? { questions } : null;
  } catch {
    quiz = null;
  }

  return {
    phase,
    lesson,
    markdown,
    meta,
    quiz,
    headings: extractHeadings(markdown),
  };
}

/** Recommended entry phase by learner level (used on the curriculum page). */
export const levelEntryPoints = [
  { phaseNumber: 0, key: "beginner" },
  { phaseNumber: 1, key: "python" },
  { phaseNumber: 3, key: "ml" },
  { phaseNumber: 10, key: "llm" },
  { phaseNumber: 14, key: "agents" },
] as const;
