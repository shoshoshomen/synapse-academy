"use server";

import { AuthError } from "next-auth";
import { eq } from "drizzle-orm";
import bcrypt from "bcryptjs";
import { signIn } from "@/auth";
import { getDb, hasDatabase } from "@/lib/db";
import { users } from "@/lib/db/schema";

export type AuthState = { error?: string } | null;

export async function authAction(
  _prev: AuthState,
  formData: FormData,
): Promise<AuthState> {
  const mode = String(formData.get("mode") ?? "in");
  const email = String(formData.get("email") ?? "")
    .toLowerCase()
    .trim();
  const password = String(formData.get("password") ?? "");
  const name = String(formData.get("name") ?? "").trim();

  if (!hasDatabase) return { error: "generic" };
  if (!email || password.length < 6) return { error: "invalid" };

  try {
    if (mode === "up") {
      const db = getDb();
      if (!db) return { error: "generic" };
      const existing = await db
        .select()
        .from(users)
        .where(eq(users.email, email))
        .limit(1);
      if (existing[0]) return { error: "exists" };
      const passwordHash = await bcrypt.hash(password, 10);
      await db
        .insert(users)
        .values({ email, name: name || null, passwordHash });
    }

    await signIn("credentials", {
      email,
      password,
      redirectTo: "/dashboard",
    });
    return null;
  } catch (e) {
    if (e instanceof AuthError) return { error: "invalid" };
    throw e; // re-throw redirect (NEXT_REDIRECT) and others
  }
}
