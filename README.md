<div align="center">

# Synapse Academy

### Don't just use AI. Build it.

A premium, bilingual (English / 日本語) e-learning platform that teaches AI
engineering **from first principles to production** — 503 lessons across 20
phases, every one of them build-first.

[![Live demo](https://img.shields.io/badge/demo-aiengineering--delta.vercel.app-0b7285?style=flat-square)](https://aiengineering-delta.vercel.app/en)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-000000?style=flat-square&logo=next.js)](https://nextjs.org)
[![React 19](https://img.shields.io/badge/React-19-149eca?style=flat-square&logo=react)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178c6?style=flat-square&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![Tailwind v4](https://img.shields.io/badge/Tailwind-v4-38bdf8?style=flat-square&logo=tailwindcss&logoColor=white)](https://tailwindcss.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](./LICENSE)

**503 lessons · 20 phases · English + 日本語 · 320+ hours of material**

[Live demo](https://aiengineering-delta.vercel.app/en) ·
[The 20-phase path](https://aiengineering-delta.vercel.app/en/curriculum) ·
[Tech stack](#tech-stack) ·
[Run it locally](#local-development)

</div>

---

## Why this exists

The people who shape AI don't stop at prompts — they understand it down to the
mathematics and build it with their own hands. Synapse Academy is the path from
first principles to production systems: you derive the math, implement it from
scratch with stdlib and numpy, **and then** see how PyTorch, the diffusers
library and the agent SDKs are just the industrial version of what you built.

Every lesson ships a real artifact — a prompt, a skill, an agent, or an MCP
server — so you finish with a portfolio that proves you can build, not a
certificate that says you watched videos.

> The curriculum content is adapted from the open-source
> [**AI Engineering from Scratch**](https://github.com/rohitg00/ai-engineering-from-scratch)
> by **Rohit Ghumare** (MIT). This repository is the platform that turns it into
> a bilingual, tested, hostable school.

## What's inside

- **503 lessons, every one written out** — prose, derivations, runnable code,
  and `mermaid` diagrams; not an outline with placeholders.
- **Fully bilingual.** UI *and* lesson bodies exist in English and 日本語
  (503 `lesson.md` + 503 `lesson.ja.md`). One click switches the whole site.
- **A test on every lesson.** 338 auto-graded knowledge-check quizzes with
  per-answer explanations confirm understanding on the spot.
- **Progress tracking.** Works in guest mode out of the box (localStorage); add a
  database to sync completion and quiz scores across devices.
- **Account access, gated content.** Optional Auth.js sign-in (email + Google);
  a shared-password gate can protect lesson bodies on a public deployment while
  keeping the curriculum itself discoverable.
- **Academic, trustworthy design.** Deep navy + warm paper white, serif display
  type, restrained brass accents.

## The 20-phase path

Start where you are; every phase builds on the last — from a development
environment to training an LLM from scratch to shipping autonomous agents.

| #  | Phase | Lessons | Focus |
| -- | ----- | ------: | ----- |
| 00 | Setup & Tooling | 12 | uv, GPUs, Docker, Linux, profiling |
| 01 | Math Foundations | 22 | linear algebra → Fourier / probability; a `micrograd` autograd by hand |
| 02 | ML Fundamentals | 18 | classic ML from scratch, then scikit-learn |
| 03 | Deep Learning Core | 13 | backprop by hand → a mini-framework → PyTorch / JAX |
| 04 | Computer Vision | 28 | convolutions from scratch → DiT / FLUX, SAM 3, 3D Gaussian Splatting |
| 05 | NLP | 29 | Word2Vec by hand → RAG, LLM eval, long-context |
| 06 | Speech & Audio | 17 | STFT / MFCC → Whisper → full-duplex (Moshi) |
| 07 | Transformers | 16 | attention from scratch → Flash Attention, KV cache, MoE |
| 08 | Generative AI | 15 | VAE / GAN → DDPM → flow matching / rectified flow |
| 09 | Reinforcement Learning | 12 | MDPs → PPO → RLHF / DPO / GRPO |
| 10 | LLMs From Scratch | 24 | a numpy GPT-2 (124M) → NSA, DeepSeek-V3, MTP |
| 11 | LLM Engineering | 17 | RAG, MCP, LangGraph, function calling, guardrails |
| 12 | Multimodal AI | 25 | CLIP / SigLIP → Transfusion, Janus-Pro, embodied VLA |
| 13 | Tools & Protocols | 23 | MCP in depth, A2A, OpenTelemetry, OAuth 2.1 |
| 14 | Agent Engineering | 42 | the ReAct loop → memory, planning, subagents, every major SDK |
| 15 | Autonomous Systems | 22 | self-improvement, kill switches, circuit breakers |
| 16 | Multi-Agent & Swarms | 25 | supervisor patterns, BFT for LLMs, failure taxonomy |
| 17 | Infrastructure & Production | 28 | vLLM / PagedAttention, prefill/decode split, FinOps, SRE |
| 18 | Ethics, Safety & Alignment | 30 | alignment faking, prompt injection, governance |
| 19 | Capstone Projects | 85 | 17 applied + 58 from-scratch builds, rubric-graded |

## Tech stack

| Layer | Choice |
| ----- | ------ |
| Framework | Next.js 16 (App Router) + React 19, TypeScript |
| Styling | Tailwind CSS v4 + `@tailwindcss/typography` |
| i18n | `next-intl` (en / ja, UI **and** content) |
| Content | Markdown + JSON synced from the source repo |
| Rendering | `react-markdown` + `rehype-highlight` + `mermaid` |
| Auth (opt.) | Auth.js v5 — email/password + Google OAuth, bcrypt |
| Database (opt.) | Drizzle ORM + Postgres (Neon) |
| Hosting | Vercel |

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

Run without `GATE_PASSWORD` set and every lesson body is readable locally — the
password gate is for hosted deployments only.

## Project structure

```
app/[locale]/                 Localized routes (home, curriculum, phases, lesson, dashboard, sign-in)
  phases/[phase]/             Phase detail (lesson list)
  phases/[phase]/[lesson]/    Lesson page: material + quiz + progress
  unlock/                     Shared-password gate for lesson bodies
components/                   UI, lesson renderer (markdown/mermaid/toc/quiz), header/footer
content/                      Curriculum: curriculum.json + phases/<id>/<lesson>/{lesson(.ja).md,meta.json,quiz.json}
i18n/                         next-intl routing/request/navigation
lib/content.ts               Server-side content loader (curriculum, lessons, quizzes, TOC)
lib/gate.ts                  Shared-password gate helpers (hash + cookie)
lib/progress.ts              Client progress store (localStorage)
lib/db/                       Optional Drizzle schema + lazy client
proxy.ts                     Next 16 proxy: i18n routing + lesson-body gate
messages/{en,ja}.json        UI strings
scripts/sync-content.mjs     Re-sync curriculum from the source repo
```

## Content access (password gate)

The **curriculum is public** — home page, the 20-phase outline, and each phase's
lesson list are open to anyone, so the site is discoverable and shareable. The
**lesson bodies** are protected: set `GATE_PASSWORD` on the deployment and
`proxy.ts` redirects any lesson route to `/unlock` until the shared password is
entered (a hashed, http-only cookie remembers it). Leave `GATE_PASSWORD` unset
and the gate is off — ideal for local development.

## Accounts & cross-device progress (optional)

Without configuration the app runs in **guest mode** — every lesson and quiz
works and progress is stored in the browser. To enable accounts:

1. Add a Postgres database (e.g. **Neon** from the Vercel Marketplace). This sets
   `DATABASE_URL`. Locally, copy `.env.example` → `.env.local`.
2. Set `AUTH_SECRET` (`openssl rand -base64 32`).
3. *(Optional)* Set `AUTH_GOOGLE_ID` / `AUTH_GOOGLE_SECRET` to enable Google sign-in.
4. Create the tables:
   ```bash
   npm run db:generate && npm run db:migrate
   ```

When `DATABASE_URL` is present the sign-in page exposes account creation /
sign-in; otherwise it stays in guest mode automatically.

## Deploy to Vercel

1. Push this repository to GitHub.
2. Import it in Vercel (framework auto-detected as Next.js).
3. *(Optional)* Add a Neon database from the Marketplace, set `AUTH_SECRET`, and
   set `GATE_PASSWORD` to lock lesson bodies behind a shared password.
4. Deploy. Update `siteConfig.url` in `config/site.ts` to your final domain.

> Environment-variable changes only take effect after a **redeploy**.

## Branding

All school branding lives in [`config/site.ts`](./config/site.ts) — change
`name`, `tagline`, and `url` in one place to rebrand. Colors and typography are
defined in [`app/globals.css`](./app/globals.css).

## License & attribution

This project is released under the [**MIT License**](./LICENSE).

The curriculum content under `content/` is adapted from
[**AI Engineering from Scratch**](https://github.com/rohitg00/ai-engineering-from-scratch)
by **Rohit Ghumare**, used under the MIT License — its copyright notice is
preserved in [`NOTICE`](./NOTICE), and attribution is shown on every lesson and
in the footer. The platform's own application code is likewise MIT.
