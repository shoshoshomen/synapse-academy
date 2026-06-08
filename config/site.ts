/**
 * Single source of truth for school branding.
 * Change the name / domain here to rebrand the whole site.
 */
export const siteConfig = {
  name: "Synapse Academy",
  shortName: "Synapse",
  tagline: "Build AI from scratch.",
  // Update after first deploy
  url: "https://synapse-academy.vercel.app",
  source: {
    repo: "rohitg00/ai-engineering-from-scratch",
    url: "https://github.com/rohitg00/ai-engineering-from-scratch",
    author: "Rohit Ghumare",
    license: "MIT",
  },
  stats: {
    lessons: 503,
    phases: 20,
    hours: 320,
    languages: 4,
  },
  // Quiz pass mark (0..1)
  passThreshold: 0.7,
} as const;

export type SiteConfig = typeof siteConfig;
