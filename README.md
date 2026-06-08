# Synapse Academy

A premium, bilingual e-learning platform for AI engineering — built on the
open-source [**AI Engineering from Scratch**](https://github.com/rohitg00/ai-engineering-from-scratch)
curriculum (503 lessons across 20 phases, MIT licensed).

- **Curriculum + tests.** Every lesson renders rich material (prose, code, mermaid
  diagrams) followed by an auto-graded knowledge-check quiz with explanations.
- **English-first, fully Japanese UI.** The interface switches between English and
  日本語 instantly; lesson content stays in English.
- **Progress tracking.** Lesson completion and quiz scores are saved per learner.
  Works in guest mode out of the box; add a database to sync across devices.
- **Academic, trustworthy design.** Deep navy + warm paper white, serif display
  type, restrained brass accents.

## Tech stack

| Layer        | Choice                                              |
| ------------ | --------------------------------------------------- |
| Framework    | Next.js 16 (App Router) + React 19, TypeScript      |
| Styling      | Tailwind CSS v4 + `@tailwindcss/typography`         |
| i18n         | `next-intl` (en / ja, UI-only)                      |
| Content      | Markdown + JSON synced from the source repo         |
| Rendering    | `react-markdown` + `rehype-highlight` + `mermaid`   |
| Auth (opt.)  | Auth.js v5 (credentials), bcrypt                    |
| Database (opt.) | Drizzle ORM + Postgres (Neon)                    |
| Hosting      | Vercel                                              |

## Local development

```bash
npm install
npm run dev          # http://localhost:3000  → redirects to /en
```

Build & run the production bundle:

```bash
npm run build
npm start
```

## Project structure

```
app/[locale]/                 Localized routes (home, curriculum, phases, lesson, dashboard, sign-in)
  phases/[phase]/             Phase detail (lesson list)
  phases/[phase]/[lesson]/    Lesson page: material + quiz + progress
components/                   UI, lesson renderer (markdown/mermaid/toc/quiz), header/footer
content/                      Synced curriculum: curriculum.json + phases/<id>/<lesson>/{lesson.md,meta.json,quiz.json}
i18n/                         next-intl routing/request/navigation
lib/content.ts               Server-side content loader (curriculum, lessons, quizzes, TOC)
lib/progress.ts              Client progress store (localStorage)
lib/db/                       Optional Drizzle schema + lazy client
messages/{en,ja}.json        UI strings
scripts/sync-content.mjs     Re-sync curriculum from the source repo
```

## Syncing / expanding content

The first release ships the full 20-phase outline plus the opening lesson of
every phase. To re-pull (or after upstream updates):

```bash
npm run sync:content
```

To add more full lessons, copy additional lesson folders from the source repo
into `content/phases/<phaseId>/<lessonId>/` as `lesson.md` + `meta.json`
(+ optional `quiz.json`) and flip `hasContent: true` in `content/curriculum.json`.

## Accounts & cross-device progress (optional)

Without configuration the app runs in **guest mode** — every lesson and quiz
works and progress is stored in the browser. To enable accounts:

1. Add a Postgres database (e.g. **Neon** from the Vercel Marketplace). This sets
   `DATABASE_URL` in your Vercel project. Locally, copy `.env.example` → `.env.local`.
2. Set `AUTH_SECRET` (`openssl rand -base64 32`).
3. Create the tables:
   ```bash
   npm run db:push        # or: npm run db:generate && npm run db:migrate
   ```

When `DATABASE_URL` is present the sign-in page exposes account creation /
sign-in; otherwise it stays in guest mode automatically.

## Deploy to Vercel

1. Push this repository to GitHub.
2. Import it in Vercel (framework auto-detected as Next.js).
3. *(Optional)* Add a Neon database from the Marketplace and set `AUTH_SECRET`.
4. Deploy. Update `siteConfig.url` in `config/site.ts` to your final domain.

## Branding

All school branding lives in [`config/site.ts`](./config/site.ts) — change
`name`, `tagline`, and `url` in one place to rebrand. Colors and typography are
defined in [`app/globals.css`](./app/globals.css).

## License & attribution

The curriculum content is adapted from
[**AI Engineering from Scratch**](https://github.com/rohitg00/ai-engineering-from-scratch)
by **Rohit Goel**, used under the **MIT License**. Attribution is shown on every
lesson and in the footer. This platform's own application code is likewise MIT.
