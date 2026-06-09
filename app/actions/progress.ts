"use server";

import { eq } from "drizzle-orm";
import { auth } from "@/auth";
import { getDb, hasDatabase } from "@/lib/db";
import { lessonProgress, quizResults } from "@/lib/db/schema";

export interface ServerQuiz {
  correct: number;
  total: number;
  passed: boolean;
  at: number;
}

export interface ServerProgress {
  completed: string[];
  quizzes: Record<string, ServerQuiz>;
}

async function currentEmail(): Promise<string | null> {
  const session = await auth();
  const email = session?.user?.email;
  return email ? email.toLowerCase() : null;
}

/** Reads the signed-in user's progress, or null when not signed in / no DB. */
export async function getServerProgress(): Promise<ServerProgress | null> {
  if (!hasDatabase) return null;
  const email = await currentEmail();
  if (!email) return null;
  const db = getDb();
  if (!db) return null;

  const [lessons, quizzes] = await Promise.all([
    db.select().from(lessonProgress).where(eq(lessonProgress.userEmail, email)),
    db.select().from(quizResults).where(eq(quizResults.userEmail, email)),
  ]);

  return {
    completed: lessons.filter((l) => l.completed).map((l) => l.lessonKey),
    quizzes: Object.fromEntries(
      quizzes.map((q) => [
        q.lessonKey,
        {
          correct: q.correct,
          total: q.total,
          passed: q.passed,
          at: q.at.getTime(),
        },
      ]),
    ),
  };
}

export async function setServerComplete(lessonKey: string, completed: boolean) {
  if (!hasDatabase) return;
  const email = await currentEmail();
  if (!email) return;
  const db = getDb();
  if (!db) return;

  await db
    .insert(lessonProgress)
    .values({ userEmail: email, lessonKey, completed })
    .onConflictDoUpdate({
      target: [lessonProgress.userEmail, lessonProgress.lessonKey],
      set: { completed, updatedAt: new Date() },
    });
}

export async function setServerQuiz(
  lessonKey: string,
  result: { correct: number; total: number; passed: boolean },
) {
  if (!hasDatabase) return;
  const email = await currentEmail();
  if (!email) return;
  const db = getDb();
  if (!db) return;

  await db
    .insert(quizResults)
    .values({ userEmail: email, lessonKey, ...result })
    .onConflictDoUpdate({
      target: [quizResults.userEmail, quizResults.lessonKey],
      set: { ...result, at: new Date() },
    });
}
