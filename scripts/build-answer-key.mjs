/**
 * テスト答え集ジェネレーター（講師用・ローカル専用）
 *
 * content/ 配下の quiz.ja.json（無ければ quiz.json）を読み、
 * フェーズごとに「設問・全選択肢（正解に✓）・解説」をまとめた
 * Markdown 答え集を answers/ に出力する。
 *
 *   node scripts/build-answer-key.mjs
 *
 * 出力は answers/（.gitignore 済み = リポジトリに公開されない）。
 */
import { promises as fs } from "node:fs";
import path from "node:path";

const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "..");
const CONTENT = path.join(ROOT, "content");
const OUT = path.join(ROOT, "answers");

const LETTERS = ["A", "B", "C", "D", "E", "F"];

async function readJson(p) {
  try {
    return JSON.parse(await fs.readFile(p, "utf8"));
  } catch {
    return null;
  }
}

function normalizeQuiz(parsed) {
  if (!parsed) return [];
  const arr = Array.isArray(parsed)
    ? parsed
    : Array.isArray(parsed.questions)
      ? parsed.questions
      : [];
  return arr.filter(
    (q) =>
      q &&
      typeof q.question === "string" &&
      Array.isArray(q.options) &&
      typeof q.correct === "number",
  );
}

async function main() {
  const curriculum = await readJson(path.join(CONTENT, "curriculum.json"));
  if (!curriculum) throw new Error("curriculum.json が読めません");

  await fs.mkdir(OUT, { recursive: true });

  const indexLines = [
    "# テスト答え集（講師用）",
    "",
    "> 各レッスンのテストの正解と解説です。WEBミーティングでの質問対応用。",
    "> このフォルダは .gitignore 済みでリポジトリには公開されません。",
    "",
    "| フェーズ | 答え集 | 設問数 |",
    "| --- | --- | --- |",
  ];

  let grandTotal = 0;

  for (const phase of curriculum.phases) {
    const phaseJa = await readJson(
      path.join(CONTENT, "phases", phase.id, "phase.ja.json"),
    );
    const phaseTitle = phaseJa?.title || phase.title;
    const lessonTitleJa = (slug) => phaseJa?.lessons?.[slug] || null;

    const lines = [
      `# フェーズ${String(phase.phaseNumber).padStart(2, "0")}　${phaseTitle}`,
      "",
      `English: ${phase.title}`,
      "",
    ];
    let phaseCount = 0;

    for (const lesson of phase.lessons) {
      const dir = path.join(CONTENT, "phases", phase.id, lesson.id);
      const quiz = normalizeQuiz(
        (await readJson(path.join(dir, "quiz.ja.json"))) ??
          (await readJson(path.join(dir, "quiz.json"))),
      );
      if (!quiz.length) continue;

      const title = lessonTitleJa(lesson.slug) || lesson.title;
      lines.push(`## ${lesson.lessonNumber}. ${title}`);
      lines.push("");

      quiz.forEach((q, qi) => {
        const stage = q.stage ? `［${q.stage}］` : "";
        lines.push(`**Q${qi + 1}** ${stage} ${q.question}`);
        lines.push("");
        q.options.forEach((opt, oi) => {
          const mark = oi === q.correct ? "✅" : "　";
          lines.push(`- ${mark} ${LETTERS[oi] || oi}. ${opt}`);
        });
        lines.push("");
        if (q.explanation) {
          lines.push(`> **解説:** ${q.explanation}`);
          lines.push("");
        }
        phaseCount += 1;
      });
      lines.push("---");
      lines.push("");
    }

    const fileName = `${phase.id}.md`;
    if (phaseCount > 0) {
      await fs.writeFile(path.join(OUT, fileName), lines.join("\n"), "utf8");
      indexLines.push(
        `| ${String(phase.phaseNumber).padStart(2, "0")} ${phaseTitle} | [${fileName}](./${fileName}) | ${phaseCount} |`,
      );
      grandTotal += phaseCount;
      console.log(`  ${phase.id}: ${phaseCount} 問`);
    } else {
      indexLines.push(
        `| ${String(phase.phaseNumber).padStart(2, "0")} ${phaseTitle} | (テストなし) | 0 |`,
      );
    }
  }

  indexLines.push("", `**合計設問数: ${grandTotal}**`);
  await fs.writeFile(path.join(OUT, "README.md"), indexLines.join("\n"), "utf8");
  console.log(`\n合計 ${grandTotal} 問の答え集を ${OUT} に生成しました。`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
