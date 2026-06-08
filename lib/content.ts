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

export type Locale = "en" | "ja";

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

// ─── Japanese sidecar (per-phase structure translation) ──────────────────────
interface PhaseJa {
  title?: string;
  description?: string;
  lessons?: Record<string, string>; // lessonSlug -> JA title
}

const phaseJaCache = new Map<string, PhaseJa | null>();

async function readPhaseJa(phaseId: string): Promise<PhaseJa | null> {
  if (phaseJaCache.has(phaseId)) return phaseJaCache.get(phaseId)!;
  let value: PhaseJa | null = null;
  try {
    const raw = await fs.readFile(
      path.join(CONTENT_DIR, "phases", phaseId, "phase.ja.json"),
      "utf8",
    );
    value = JSON.parse(raw) as PhaseJa;
  } catch {
    value = null;
  }
  phaseJaCache.set(phaseId, value);
  return value;
}

function localizePhase(phase: Phase, ja: PhaseJa | null): Phase {
  if (!ja) return phase;
  return {
    ...phase,
    title: ja.title || phase.title,
    description: ja.description || phase.description,
    lessons: phase.lessons.map((l) => ({
      ...l,
      title: ja.lessons?.[l.slug] || l.title,
    })),
  };
}

// ─── Public API (locale-aware) ───────────────────────────────────────────────

export async function getPhases(locale: string = "en"): Promise<Phase[]> {
  const phases = (await getCurriculum()).phases;
  if (locale !== "ja") return phases;
  return Promise.all(
    phases.map(async (p) => localizePhase(p, await readPhaseJa(p.id))),
  );
}

export async function getPhase(
  idOrSlug: string,
  locale: string = "en",
): Promise<Phase | undefined> {
  const phases = (await getCurriculum()).phases;
  const phase = phases.find(
    (p) =>
      p.slug === idOrSlug ||
      p.id === idOrSlug ||
      String(p.phaseNumber) === idOrSlug ||
      String(p.phaseNumber).padStart(2, "0") === idOrSlug,
  );
  if (!phase) return undefined;
  if (locale !== "ja") return phase;
  return localizePhase(phase, await readPhaseJa(phase.id));
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
  const phases = (await getCurriculum()).phases;
  return phases.reduce((sum, p) => sum + p.lessonCount, 0);
}

/** One preview lesson per phase (lessons that have written content). */
export async function getPreviewLessons(
  locale: string = "en",
): Promise<{ phase: Phase; lesson: LessonRef }[]> {
  const phases = await getPhases(locale);
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

async function readFirst(paths: string[]): Promise<string | null> {
  for (const p of paths) {
    try {
      return await fs.readFile(p, "utf8");
    } catch {
      /* try next */
    }
  }
  return null;
}

/**
 * Load the full content for a lesson. When locale === "ja" the Japanese
 * sidecar files (lesson.ja.md / meta.ja.json / quiz.ja.json) are used when
 * present, falling back to English otherwise. Returns null if no material.
 */
export async function getLessonContent(
  phaseSlugOrId: string,
  lessonSlugOrId: string,
  locale: string = "en",
): Promise<LessonContent | null> {
  const phase = await getPhase(phaseSlugOrId, locale);
  if (!phase) return null;
  const lesson = findLesson(phase, lessonSlugOrId);
  if (!lesson) return null;

  const dir = path.join(CONTENT_DIR, "phases", phase.id, lesson.id);
  const ja = locale === "ja";

  const markdown = await readFirst([
    ...(ja ? [path.join(dir, "lesson.ja.md")] : []),
    path.join(dir, "lesson.md"),
  ]);
  const metaRaw = await readFirst([
    ...(ja ? [path.join(dir, "meta.ja.json")] : []),
    path.join(dir, "meta.json"),
  ]);
  if (markdown === null || metaRaw === null) return null;

  let meta: LessonMeta;
  try {
    meta = JSON.parse(metaRaw) as LessonMeta;
  } catch {
    return null;
  }

  // quiz.json comes in two shapes upstream: a top-level array of questions,
  // or an object { questions: [...] }. Normalize both, drop malformed items.
  let quiz: Quiz | null = null;
  const quizRaw = await readFirst([
    ...(ja ? [path.join(dir, "quiz.ja.json")] : []),
    path.join(dir, "quiz.json"),
  ]);
  if (quizRaw) {
    try {
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
