/**
 * Shared-password gate for lesson bodies.
 *
 * The curriculum (home, /curriculum, phase index) stays public so the school is
 * discoverable; only the lesson body + quiz live behind a single shared password.
 * The gate activates only when GATE_PASSWORD is configured (set it in Vercel) —
 * with no password set, lessons stay open (handy for local dev).
 */

export const GATE_COOKIE = "lesson_gate";

const SALT = "synapse-academy-gate-v1";

/** The gate is active only when a shared password has been configured. */
export function isGateEnabled(): boolean {
  return Boolean(process.env.GATE_PASSWORD);
}

/**
 * Opaque cookie token a visitor must present to read lesson bodies. Derived from
 * the password so it can't be forged without knowing it, and so rotating the
 * password invalidates every existing unlock. Null when the gate is disabled.
 */
export async function gateToken(): Promise<string | null> {
  const pw = process.env.GATE_PASSWORD;
  if (!pw) return null;
  return sha256Base64Url(`${pw}::${SALT}`);
}

/** True for a lesson *body* route: /{locale}/phases/{phase}/{lesson}. */
export function isLessonPath(pathname: string): boolean {
  const seg = pathname.split("/").filter(Boolean);
  // [locale, "phases", phase, lesson] — the phase index (length 3) stays public.
  return seg.length === 4 && seg[1] === "phases";
}

async function sha256Base64Url(input: string): Promise<string> {
  const bytes = new TextEncoder().encode(input);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  let bin = "";
  for (const b of new Uint8Array(digest)) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
