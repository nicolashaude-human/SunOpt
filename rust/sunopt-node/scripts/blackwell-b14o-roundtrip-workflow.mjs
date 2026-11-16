// rust/sunopt-node/scripts/blackwell-b14o-roundtrip-workflow.mjs
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const nativeRoot = path.resolve(here, "..");
const projectRoot = path.resolve(nativeRoot, "..", "..");
const outbox = path.join(projectRoot, "updates", "outbox");
const hashFile = path.join(projectRoot, "updates", "history", "hash.txt");
const child = path.join(nativeRoot, "scripts", "blackwell-b14o-child.mjs");
const snapshot = path.join(nativeRoot, "blackwell-b12-sync-blackwell.bin");
const reference = path.join(nativeRoot, "benchmarks", "references", "blackwell-b14m-strict-reference.bin");
fs.mkdirSync(outbox, { recursive: true });

function runChild(extra) {
  const args = [child, "--snapshot", snapshot, "--reference", reference, ...extra];
  const p = spawnSync(process.execPath, args, { cwd: projectRoot, encoding: "utf8", windowsHide: true });
  if (p.status !== 0) throw new Error(`child failed ${extra.join(" ")}\nstdout:\n${p.stdout}\nstderr:\n${p.stderr}`);
  const lines = p.stdout.trim().split(/\r?\n/);
  return JSON.parse(lines[lines.length - 1]);
}

function median(values) {
  const a = [...values].sort((x, y) => x - y);
  const m = Math.floor(a.length / 2);
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

function summarize(label, rows) {
  const wall = rows.map(x => x.result.wallMs);
  const kernel = rows.map(x => x.result.kernelMs);
  return {
    label,
    count: rows.length,
    wallMs: wall,
    kernelMs: kernel,
    medianWallMs: median(wall),
    medianKernelMs: median(kernel),
    minWallMs: Math.min(...wall),
    maxWallMs: Math.max(...wall),
    telemetry: rows.map(x => ({ before: x.before, after: x.after })),
    firstResult: rows[0]?.result ?? null,
  };
}

const release = fs.existsSync(hashFile) ? fs.readFileSync(hashFile, "utf8").trim() : "unknown";

// Persistent strict reference, generated outside the outbox.
runChild(["--kind", "reference"]);

// A) B14L pageable/pinned robust A/B.
runChild(["--kind", "b14l", "--pinned", "false"]);
runChild(["--kind", "b14l", "--pinned", "true"]);

const pinnedRows = { pageable: [], pinned: [] };
const abba = ["pageable","pinned","pinned","pageable","pageable","pinned","pinned","pageable","pageable","pinned"];
for (const name of abba) {
  const pinned = name === "pinned";
  pinnedRows[name].push(runChild(["--kind","b14l","--pinned",String(pinned)]));
}

const pageable = summarize("pageable", pinnedRows.pageable);
const pinned = summarize("pinned", pinnedRows.pinned);
const pinnedDeltaPct = 100 * (pinned.medianWallMs - pageable.medianWallMs) / pageable.medianWallMs;
const pinnedConclusion = Math.abs(pinnedDeltaPct) < 10 ? "INCONCLUSIVE_LT_10_PERCENT" : (pinnedDeltaPct < 0 ? "PINNED_FASTER" : "PAGEABLE_FASTER");

// B) Packed FP32 roundtrip variants.
const variants = {
  A: { label: "fp32-pageable-stream-auto", args: ["--kind","b14o","--pinned-input","false","--wait-mode","0","--mapped-output","false","--schedule-spin","false"] },
  B: { label: "fp32-pageable-stream-spin", args: ["--kind","b14o","--pinned-input","false","--wait-mode","0","--mapped-output","false","--schedule-spin","true"] },
  C: { label: "fp32-pageable-event-spin", args: ["--kind","b14o","--pinned-input","false","--wait-mode","1","--mapped-output","false","--schedule-spin","true"] },
  D: { label: "fp32-mapped-event-spin", args: ["--kind","b14o","--pinned-input","false","--wait-mode","1","--mapped-output","true","--schedule-spin","true"] },
};

for (const v of Object.values(variants)) runChild(v.args);

const rows = { A: [], B: [], C: [], D: [] };
const sequence = [
  ["A","B","C","D"],
  ["D","C","B","A"],
  ["B","A","D","C"],
  ["C","D","A","B"],
  ["A","C","B","D"],
];
for (const cycle of sequence) {
  for (const key of cycle) rows[key].push(runChild(variants[key].args));
}

const roundtrip = Object.fromEntries(Object.entries(rows).map(([key, value]) => [key, summarize(variants[key].label, value)]));
const baselineWall = roundtrip.A.medianWallMs;
for (const key of Object.keys(roundtrip)) {
  const item = roundtrip[key];
  item.speedupVsA = baselineWall / item.medianWallMs;
  item.deltaPctVsA = 100 * (item.medianWallMs - baselineWall) / baselineWall;
  item.meaningfulVsA = Math.abs(item.deltaPctVsA) >= 10;
}

const bestKey = Object.keys(roundtrip).reduce((a,b) => roundtrip[a].medianWallMs <= roundtrip[b].medianWallMs ? a : b);

const report = {
  stage: "B14O_ROUNDTRIP_WORKFLOW",
  release,
  measurementPolicy: { measuredRunsPerVariant: 5, warmupRunsPerVariant: 1, medianAuthoritative: true, noiseThresholdPercent: 10 },
  pinnedAB: { pageable, pinned, deltaPct: pinnedDeltaPct, conclusion: pinnedConclusion },
  roundtrip,
  bestRoundtrip: { key: bestKey, ...roundtrip[bestKey] },
  persistentStrictReference: reference,
};

const reportPath = path.join(outbox, "blackwell-b14o-roundtrip-workflow.json");
fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");

const llmPath = path.join(outbox, "llm.txt");
fs.appendFileSync(llmPath, [
  "",
  `=== ${new Date().toISOString()} B14O_ROUNDTRIP_WORKFLOW ===`,
  `release=${release}`,
  `pinned pageableMedianWallMs=${pageable.medianWallMs.toFixed(6)} pinnedMedianWallMs=${pinned.medianWallMs.toFixed(6)} deltaPct=${pinnedDeltaPct.toFixed(3)} conclusion=${pinnedConclusion}`,
  ...Object.entries(roundtrip).map(([k,v]) => `${k} ${v.label} medianKernelMs=${v.medianKernelMs.toFixed(6)} medianWallMs=${v.medianWallMs.toFixed(6)} deltaPctVsA=${v.deltaPctVsA.toFixed(3)} meaningfulVsA=${v.meaningfulVsA}`),
  `bestRoundtrip=${bestKey} ${roundtrip[bestKey].label} wallMs=${roundtrip[bestKey].medianWallMs.toFixed(6)}`,
  `persistentStrictReference=${reference}`,
  `report=${reportPath}`,
  "",
].join("\n"), "utf8");

console.log(JSON.stringify(report, null, 2));
console.log(`report=${reportPath}`);
console.log(`llm=${llmPath}`);
