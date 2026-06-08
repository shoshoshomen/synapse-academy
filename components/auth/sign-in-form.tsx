"use client";

import { useActionState, useState } from "react";
import { useTranslations } from "next-intl";
import { authAction, type AuthState } from "@/app/actions/auth";

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

export function SignInForm() {
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
    <form action={formAction} className="mt-6 space-y-4">
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
  );
}
