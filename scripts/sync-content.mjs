/**
 * このスクリプトはMITライセンスのrohitg00/ai-engineering-from-scratchから教材を同期する
 *
 * Source: https://github.com/rohitg00/ai-engineering-from-scratch (MIT License)
 * Author: Rohit Goel
 *
 * Usage:
 *   node scripts/sync-content.mjs [--source /tmp/aieng-src] [--output content]
 *
 * This script:
 *   1. Clones the source repo to a temp directory (or uses --source if provided)
 *   2. Walks phases/NN-slug/ directories
 *   3. For each phase, collects lesson metadata
 *   4. For the first lesson of each phase (01-*), copies docs/en.md and quiz.json
 *   5. Writes content/curriculum.json and per-lesson files
 *   6. Cleans up the temp clone (unless --source was provided externally)
 */

import { execSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync, rmSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { tmpdir } from 'node:os';

// ─── Configuration ────────────────────────────────────────────────────────────

const REPO_URL = 'https://github.com/rohitg00/ai-engineering-from-scratch.git';
const SOURCE_META = {
  repo: 'rohitg00/ai-engineering-from-scratch',
  url: 'https://github.com/rohitg00/ai-engineering-from-scratch',
  license: 'MIT',
  author: 'Rohit Ghumare',
};

// Parse CLI args
const args = process.argv.slice(2);
const sourceIdx = args.indexOf('--source');
const outputIdx = args.indexOf('--output');
const providedSource = sourceIdx !== -1 ? args[sourceIdx + 1] : null;
const outputDir = outputIdx !== -1 ? args[outputIdx + 1] : 'content';

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Convert a slug like "setup-and-tooling" to Title Case "Setup And Tooling"
 * with special casing for known abbreviations.
 */
function slugToTitle(slug) {
  const SPECIAL = {
    ai: 'AI',
    ml: 'ML',
    nlp: 'NLP',
    llm: 'LLM',
    llms: 'LLMs',
    rag: 'RAG',
    gpu: 'GPU',
    api: 'API',
    apis: 'APIs',
    mcp: 'MCP',
    dns: 'DNS',
    ci: 'CI',
    cd: 'CD',
    sdk: 'SDK',
    gpt: 'GPT',
    vit: 'ViT',
    rl: 'RL',
    mdp: 'MDP',
    mdps: 'MDPs',
    cot: 'CoT',
    tot: 'ToT',
    dpo: 'DPO',
    rlhf: 'RLHF',
    peft: 'PEFT',
    lora: 'LoRA',
    rag: 'RAG',
    sql: 'SQL',
    ui: 'UI',
    ux: 'UX',
    aws: 'AWS',
    gcp: 'GCP',
    os: 'OS',
  };
  return slug
    .split('-')
    .map((w) => SPECIAL[w.toLowerCase()] ?? w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

/**
 * Parse inline bold-key metadata from en.md.
 * Looks for lines like:  **Type:** Build
 * Returns an object with keys: type, languages, prerequisites, timeEstimate
 */
function parseInlineMetadata(content) {
  const meta = {
    title: '',
    type: '',
    languages: [],
    prerequisites: '',
    timeEstimate: '',
    objectives: [],
  };

  // Title: first H1
  const titleMatch = content.match(/^#\s+(.+)/m);
  if (titleMatch) meta.title = titleMatch[1].trim();

  // Inline bold key-value pairs
  const typeMatch = content.match(/\*\*Type:\*\*\s*(.+)/);
  if (typeMatch) meta.type = typeMatch[1].trim();

  const langMatch = content.match(/\*\*Languages?:\*\*\s*(.+)/);
  if (langMatch) {
    meta.languages = langMatch[1].split(',').map((l) => l.trim()).filter(Boolean);
  }

  const prereqMatch = content.match(/\*\*Prerequisites?:\*\*\s*(.+)/);
  if (prereqMatch) meta.prerequisites = prereqMatch[1].trim();

  const timeMatch = content.match(/\*\*Time:\*\*\s*(.+)/);
  if (timeMatch) meta.timeEstimate = timeMatch[1].trim();

  // Learning Objectives section
  const objSection = content.match(/## Learning Objectives\n([\s\S]*?)(?:\n## |\n---|\n# |$)/);
  if (objSection) {
    meta.objectives = objSection[1]
      .split('\n')
      .filter((l) => l.startsWith('- '))
      .map((l) => l.replace(/^- /, '').trim());
  }

  return meta;
}

/**
 * Strip inline metadata header lines from en.md body so the stored lesson.md
 * is clean markdown content without the repeated header block.
 * We keep the H1, strip the bold key-value block, keep everything else.
 */
function stripMetadataHeader(content) {
  // Remove the block of **Key:** Value lines (they appear between the H1 and ## sections)
  // Pattern: one or more lines matching **Something:** value
  return content.replace(/(\*\*(?:Type|Languages?|Prerequisites?|Time):\*\*\s*.+\n?)+/g, '').trim();
}

/**
 * Get the first paragraph (description) from a README.md.
 * Skips the H1 title line and the blockquote line (> ...) if present.
 */
function extractDescription(readmePath) {
  if (!existsSync(readmePath)) return '';
  const content = readFileSync(readmePath, 'utf8');
  const lines = content.split('\n');
  // Find first non-empty, non-heading, non-quote line
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (trimmed.startsWith('#')) continue;
    if (trimmed.startsWith('>')) {
      // Return the blockquote content as description
      return trimmed.replace(/^>\s*/, '');
    }
    return trimmed;
  }
  return '';
}

/**
 * Read and parse a lesson directory.
 * Returns { id, lessonNumber, slug, title, hasContent } for curriculum.json
 * and optionally { lessonMd, quizJson, metaJson } for first-lesson content files.
 */
function parseLesson(lessonDir, phaseDir, includeContent = false) {
  const lessonSlug = lessonDir; // e.g. "01-dev-environment"
  const match = lessonSlug.match(/^(\d+)-(.+)$/);
  if (!match) return null;

  const lessonNumber = parseInt(match[1], 10);
  const slug = match[2];
  const id = lessonSlug;

  const docsEnPath = join(phaseDir, lessonDir, 'docs', 'en.md');
  const quizPath = join(phaseDir, lessonDir, 'quiz.json');

  let title = slugToTitle(slug);
  let lessonMd = null;
  let quizJson = null;
  let metaJson = null;

  // Try to get title from en.md H1
  if (existsSync(docsEnPath)) {
    const content = readFileSync(docsEnPath, 'utf8');
    const h1 = content.match(/^#\s+(.+)/m);
    if (h1) title = h1[1].trim();

    if (includeContent) {
      lessonMd = stripMetadataHeader(content);
      metaJson = parseInlineMetadata(content);
    }
  }

  if (includeContent && existsSync(quizPath)) {
    try {
      quizJson = JSON.parse(readFileSync(quizPath, 'utf8'));
    } catch {
      quizJson = null;
    }
  }

  return {
    id,
    lessonNumber,
    slug,
    title,
    hasContent: existsSync(docsEnPath),
    // Content fields (only when includeContent=true)
    _lessonMd: lessonMd,
    _quizJson: quizJson,
    _metaJson: metaJson,
    _docsExists: existsSync(docsEnPath),
    _quizExists: existsSync(quizPath),
  };
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  // ── Step 1: Clone or use provided source ──────────────────────────────────
  let srcDir = providedSource ? resolve(providedSource) : null;
  let clonedTmp = false;

  if (!srcDir) {
    srcDir = join(tmpdir(), `aieng-src-${Date.now()}`);
    console.log(`Cloning ${REPO_URL} to ${srcDir} ...`);
    execSync(`git clone --depth 1 ${REPO_URL} "${srcDir}"`, { stdio: 'inherit' });
    clonedTmp = true;
  } else {
    console.log(`Using provided source: ${srcDir}`);
  }

  const phasesDir = join(srcDir, 'phases');
  if (!existsSync(phasesDir)) {
    throw new Error(`phases/ directory not found in ${srcDir}`);
  }

  // ── Step 2: Determine output directory ────────────────────────────────────
  // When run from the project root, outputDir is relative to cwd.
  // Scripts live in scripts/, so we resolve from the script's parent.
  const scriptDir = new URL('.', import.meta.url).pathname;
  const projectRoot = resolve(scriptDir, '..');
  const contentDir = resolve(projectRoot, outputDir);

  console.log(`Writing output to: ${contentDir}`);
  mkdirSync(contentDir, { recursive: true });
  mkdirSync(join(contentDir, 'phases'), { recursive: true });

  // ── Step 3: Walk phases ────────────────────────────────────────────────────
  const phaseDirs = readdirSync(phasesDir)
    .filter((d) => /^\d{2}-/.test(d))
    .sort();

  const phases = [];

  for (const phaseDir of phaseDirs) {
    const phaseMatch = phaseDir.match(/^(\d{2})-(.+)$/);
    if (!phaseMatch) continue;

    const phaseNumber = parseInt(phaseMatch[1], 10);
    const phaseSlug = phaseMatch[2];
    const phaseId = phaseDir;
    const phaseTitle = slugToTitle(phaseSlug);
    const fullPhaseDir = join(phasesDir, phaseDir);

    // Description from README.md
    const description = extractDescription(join(fullPhaseDir, 'README.md'));

    // Collect lessons
    const lessonDirs = readdirSync(fullPhaseDir)
      .filter((d) => /^\d{2}-/.test(d) && statSync(join(fullPhaseDir, d)).isDirectory())
      .sort();

    const lessons = [];
    for (const lessonDir of lessonDirs) {
      // Pull full content for EVERY lesson that has material.
      const parsed = parseLesson(lessonDir, fullPhaseDir, true);
      if (!parsed) continue;

      lessons.push({
        id: parsed.id,
        lessonNumber: parsed.lessonNumber,
        slug: parsed.slug,
        title: parsed.title,
        hasContent: parsed.hasContent,
      });

      // Write content files for every lesson that has a docs/en.md
      if (parsed._docsExists) {
        const lessonOutDir = join(contentDir, 'phases', phaseId, parsed.id);
        mkdirSync(lessonOutDir, { recursive: true });

        // lesson.md
        if (parsed._lessonMd !== null) {
          writeFileSync(join(lessonOutDir, 'lesson.md'), parsed._lessonMd, 'utf8');
        }

        // meta.json
        if (parsed._metaJson !== null) {
          writeFileSync(join(lessonOutDir, 'meta.json'), JSON.stringify(parsed._metaJson, null, 2), 'utf8');
        }

        // quiz.json (only if exists)
        if (parsed._quizJson !== null) {
          writeFileSync(join(lessonOutDir, 'quiz.json'), JSON.stringify(parsed._quizJson, null, 2), 'utf8');
        }
      }
    }

    phases.push({
      id: phaseId,
      phaseNumber,
      slug: phaseSlug,
      title: phaseTitle,
      description,
      lessonCount: lessons.length,
      lessons,
    });

    console.log(`  Phase ${phaseNumber.toString().padStart(2, '0')}: ${phaseTitle} (${lessons.length} lessons)`);
  }

  // ── Step 4: Write curriculum.json ─────────────────────────────────────────
  const curriculum = {
    source: SOURCE_META,
    phases,
  };
  writeFileSync(join(contentDir, 'curriculum.json'), JSON.stringify(curriculum, null, 2), 'utf8');
  console.log(`\nWrote curriculum.json (${phases.length} phases)`);

  // ── Step 5: Cleanup ──────────────────────────────────────────────────────
  if (clonedTmp) {
    console.log(`Removing temp clone: ${srcDir}`);
    rmSync(srcDir, { recursive: true, force: true });
  }

  console.log('\nDone!');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
