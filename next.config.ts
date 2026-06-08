import type { NextConfig } from "next";
import createNextIntlPlugin from "next-intl/plugin";

const withNextIntl = createNextIntlPlugin();

const nextConfig: NextConfig = {
  // Pin the workspace root so a stray parent lockfile doesn't confuse Turbopack.
  turbopack: {
    root: process.cwd(),
  },
};

export default withNextIntl(nextConfig);
