export interface ContentSource {
  repo: string;
  url: string;
  license: string;
  author: string;
}

export interface LessonRef {
  id: string; // e.g. "01-dev-environment"
  lessonNumber: number;
  slug: string; // e.g. "dev-environment"
  title: string;
  hasContent: boolean;
}

export interface Phase {
  id: string; // e.g. "00-setup-and-tooling"
  phaseNumber: number;
  slug: string; // e.g. "setup-and-tooling"
  title: string;
  description: string;
  lessonCount: number;
  lessons: LessonRef[];
}

export interface Curriculum {
  source: ContentSource;
  phases: Phase[];
}

export interface LessonMeta {
  title: string;
  type: string; // "Build" | "Learn"
  languages: string[];
  prerequisites: string;
  timeEstimate: string;
  objectives: string[];
}

export type QuizStage = "pre" | "post";

export interface QuizQuestion {
  stage: QuizStage;
  question: string;
  options: string[];
  correct: number; // 0-based index into options
  explanation: string;
}

export interface Quiz {
  questions: QuizQuestion[];
}

export interface Heading {
  id: string;
  text: string;
  level: 2 | 3;
}

export interface LessonContent {
  phase: Phase;
  lesson: LessonRef;
  markdown: string;
  meta: LessonMeta;
  quiz: Quiz | null;
  headings: Heading[];
}
