"use server";

import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { GATE_COOKIE, gateToken } from "@/lib/gate";

export type GateState = { error?: boolean } | null;

/** Verify the shared password and, on success, drop the unlock cookie. */
export async function unlockAction(
  _prev: GateState,
  formData: FormData,
): Promise<GateState> {
  const password = String(formData.get("password") ?? "");
  const dest = safeDest(String(formData.get("from") ?? ""));

  const expected = process.env.GATE_PASSWORD;
  if (!expected) redirect(dest); // gate disabled → nothing to check

  if (password !== expected) return { error: true };

  const token = await gateToken();
  if (token) {
    const store = await cookies();
    store.set(GATE_COOKIE, token, {
      httpOnly: true,
      sameSite: "lax",
      secure: process.env.NODE_ENV === "production",
      path: "/",
      maxAge: 60 * 60 * 24 * 90, // 90 days
    });
  }
  redirect(dest);
}

/** Only allow relative, same-site redirects back into the app. */
function safeDest(from: string): string {
  return from.startsWith("/") && !from.startsWith("//") ? from : "/";
}
