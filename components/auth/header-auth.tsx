"use client";

import { useSession, signOut } from "next-auth/react";
import { useTranslations } from "next-intl";
import { Link } from "@/i18n/navigation";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function HeaderAuth() {
  const t = useTranslations("nav");
  const { status } = useSession();

  if (status === "authenticated") {
    return (
      <button
        type="button"
        onClick={() => signOut({ callbackUrl: "/" })}
        className={cn(
          buttonVariants({ variant: "outline", size: "sm" }),
          "hidden sm:inline-flex",
        )}
      >
        {t("signOut")}
      </button>
    );
  }

  return (
    <Link
      href="/sign-in"
      className={cn(
        buttonVariants({ variant: "outline", size: "sm" }),
        "hidden sm:inline-flex",
      )}
    >
      {t("signIn")}
    </Link>
  );
}
