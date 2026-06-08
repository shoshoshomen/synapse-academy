import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import * as schema from "./schema";

/** True when a database connection string is configured. */
export const hasDatabase = Boolean(process.env.DATABASE_URL);

type DB = ReturnType<typeof drizzle<typeof schema>>;

let cached: DB | null = null;

/**
 * Returns a Drizzle client, or null when no DATABASE_URL is set.
 * Connection is created lazily so the app builds & runs without a database.
 */
export function getDb(): DB | null {
  if (!hasDatabase) return null;
  if (!cached) {
    const client = postgres(process.env.DATABASE_URL!, { prepare: false });
    cached = drizzle(client, { schema });
  }
  return cached;
}
