import {
  pgTable,
  text,
  timestamp,
  integer,
  boolean,
  uuid,
  primaryKey,
} from "drizzle-orm/pg-core";

export const users = pgTable("users", {
  id: uuid("id").defaultRandom().primaryKey(),
  name: text("name"),
  email: text("email").notNull().unique(),
  passwordHash: text("password_hash"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

// Progress is keyed by email so it works for both credential and Google
// (OAuth, JWT-only) users without requiring a persisted users row.
export const lessonProgress = pgTable(
  "lesson_progress",
  {
    userEmail: text("user_email").notNull(),
    lessonKey: text("lesson_key").notNull(),
    completed: boolean("completed").default(false).notNull(),
    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (t) => [primaryKey({ columns: [t.userEmail, t.lessonKey] })],
);

export const quizResults = pgTable(
  "quiz_results",
  {
    userEmail: text("user_email").notNull(),
    lessonKey: text("lesson_key").notNull(),
    correct: integer("correct").notNull(),
    total: integer("total").notNull(),
    passed: boolean("passed").notNull(),
    at: timestamp("at").defaultNow().notNull(),
  },
  (t) => [primaryKey({ columns: [t.userEmail, t.lessonKey] })],
);
