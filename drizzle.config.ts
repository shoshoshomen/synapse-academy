import { loadEnvConfig } from "@next/env";
import type { Config } from "drizzle-kit";

// drizzle-kit runs outside Next.js, so load .env.local (and friends) explicitly.
loadEnvConfig(process.cwd());

export default {
  schema: "./lib/db/schema.ts",
  out: "./drizzle",
  dialect: "postgresql",
  dbCredentials: {
    url: process.env.DATABASE_URL ?? "",
  },
} satisfies Config;
