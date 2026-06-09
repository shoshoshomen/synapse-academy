"use client";

import { useActionState, useState } from "react";
import { useTranslations } from "next-intl";
import { authAction, googleSignInAction, type AuthState } from "@/app/actions/auth";

function GoogleIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
      <path
        fill="#4285F4"
        d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84a4.14 4.14 0 0 1-1.8 2.72v2.26h2.92c1.7-1.57 2.68-3.88 2.68-6.62z"
      />
      <path
        fill="#34A853"
        d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.92-2.26c-.8.54-1.84.86-3.04.86-2.34 0-4.32-1.58-5.03-3.7H.96v2.33A9 9 0 0 0 9 18z"
      />
      <path
        fill="#FBBC05"
        d="M3.97 10.72a5.4 5.4 0 0 1 0-3.44V4.95H.96a9 9 0 0 0 0 8.1l3.01-2.33z"
      />
      <path
        fill="#EA4335"
        d="M9 3.58c1.32 0 2.5.45 3.44 1.35l2.58-2.58C13.46.89 11.43 0 9 0A9 9 0 0 0 .96 4.95l3.01 2.33C4.68 5.16 6.66 3.58 9 3.58z"
      />
    </svg>
  );
}

function Field({
  name,
  label,
  type,
  required,
}: {
  name: string;
  label: string;
  type: string;
  required?: boolean;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-sm font-medium text-foreground">
        {label}
      </span>
      <input
        name={name}
        type={type}
        required={required}
        className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition-colors focus:border-primary focus:ring-2 focus:ring-ring/20"
      />
    </label>
  );
}

export function SignInForm({ googleEnabled }: { googleEnabled?: boolean }) {
  const t = useTranslations("auth");
  const [mode, setMode] = useState<"in" | "up">("in");
  const [state, formAction, pending] = useActionState<AuthState, FormData>(
    authAction,
    null,
  );

  const errMsg =
    state?.error === "exists"
      ? t("errorExists")
      : state?.error === "invalid"
        ? t("errorInvalid")
        : state?.error
          ? t("errorGeneric")
          : null;

  return (
    <div className="mt-6 space-y-4">
      {googleEnabled && (
        <>
          <form action={googleSignInAction}>
            <button
              type="submit"
              className="flex h-11 w-full items-center justify-center gap-2.5 rounded-md border border-border bg-background text-sm font-medium text-foreground transition-colors hover:bg-subtle"
            >
              <GoogleIcon />
              {t("continueWithGoogle")}
            </button>
          </form>
          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            <span className="h-px flex-1 bg-border" />
            {t("orDivider")}
            <span className="h-px flex-1 bg-border" />
          </div>
        </>
      )}
      <form action={formAction} className="space-y-4">
        <input type="hidden" name="mode" value={mode} />
      {mode === "up" && (
        <Field name="name" label={t("name")} type="text" />
      )}
      <Field name="email" label={t("email")} type="email" required />
      <Field name="password" label={t("password")} type="password" required />

      {errMsg && <p className="text-sm text-danger">{errMsg}</p>}

      <button
        type="submit"
        disabled={pending}
        className="h-11 w-full rounded-md bg-primary text-sm font-medium text-primary-foreground transition-colors hover:bg-primary-hover disabled:opacity-50"
      >
        {mode === "in" ? t("signInCta") : t("signUpCta")}
      </button>

      <p className="text-center text-sm text-muted-foreground">
        {mode === "in" ? t("noAccount") : t("haveAccount")}{" "}
        <button
          type="button"
          onClick={() => setMode(mode === "in" ? "up" : "in")}
          className="font-medium text-accent hover:underline"
        >
          {mode === "in" ? t("switchToSignUp") : t("switchToSignIn")}
        </button>
      </p>
      </form>
    </div>
  );
}
