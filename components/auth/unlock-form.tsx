"use client";

import { useActionState } from "react";
import { useTranslations } from "next-intl";
import { unlockAction, type GateState } from "@/app/actions/gate";

export function UnlockForm({ from }: { from: string }) {
  const t = useTranslations("unlock");
  const [state, formAction, pending] = useActionState<GateState, FormData>(
    unlockAction,
    null,
  );

  return (
    <form action={formAction} className="mt-6 space-y-4">
      <input type="hidden" name="from" value={from} />
      <label className="block">
        <span className="mb-1 block text-sm font-medium text-foreground">
          {t("label")}
        </span>
        <input
          name="password"
          type="password"
          inputMode="numeric"
          autoFocus
          required
          className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition-colors focus:border-primary focus:ring-2 focus:ring-ring/20"
        />
      </label>

      {state?.error && <p className="text-sm text-danger">{t("error")}</p>}

      <button
        type="submit"
        disabled={pending}
        className="h-11 w-full rounded-md bg-primary text-sm font-medium text-primary-foreground transition-colors hover:bg-primary-hover disabled:opacity-50"
      >
        {t("cta")}
      </button>
    </form>
  );
}
