// SunOpt B14 documentation consistency check. No project files are modified.
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
const outbox = resolve(root, "updates/outbox");
const bPath = resolve(root, "src/app/approach-b/page.tsx");
const aPath = resolve(root, "src/app/acceleration-strategy/page.tsx");
const checks = [];
const check = (label, ok) => checks.push({label, pass: Boolean(ok)});
let error = null;
try {
  const b = readFileSync(bPath, "utf8");
  const a = readFileSync(aPath, "utf8");
  check("B14 table documents measured context", b.includes("{row.context}"));
  check("B14 table displays explicit reference", b.includes("{row.reference}"));
  check("B14Q production sweep A/B reference", b.includes("6.264 s Sweep-Wall") && b.includes("3.550 s Sweep-Wall"));
  check("A11 negative end-to-end result", b.includes("182.099 s") && b.includes("7.356 s"));
  check("Pinned and Pageable explanatory prose", b.includes("Pageable Memory") && b.includes("Pinned Memory"));
  check("Windows WDDM explanation", b.includes("Windows Display Driver Model"));
  check("B14 lesson blocks", b.includes("Was Approach B tatsächlich gezeigt hat"));
  check("B14 generalization failure stated", b.includes("kein erfolgreicher Rücktransfer"));
  check("Old A transfer claim removed", !b.includes("Was Approach A von B übernehmen soll"));
  check("Blackwell physics pre-analysis", a.includes("physikalischen Eigenschaften") && a.includes("Blackwell-GPU"));
  check("No fictitious production graph gain", a.includes("nicht im Jahreslauf belegt"));
  check("No CPU sweep/per tank false chart comparison", !a.includes('beforeSeconds: 148.4553'));
} catch (e) { error = String(e?.stack || e); }
const pass = !error && checks.every(c => c.pass);
const report = {name: "sunopt-b14-documentation", at: new Date().toISOString(), pass, checks, error, modifiedProductionCode: false};
mkdirSync(outbox,{recursive:true});
const stamp = new Date().toISOString().replace(/[-:.]/g, "");
const dest = resolve(outbox, `sunopt-b14-documentation-${stamp}.json`);
writeFileSync(dest, JSON.stringify(report, null, 2) + "\n", "utf8");
if (!pass) writeFileSync(resolve(outbox,"failed-run.txt"), `B14 documentation verification FAILED; report: ${dest}\n${error||checks.filter(c=>!c.pass).map(c=>c.label).join("; ")}\n`, "utf8");
console.log(`SunOpt B14 documentation: ${pass?"PASS":"FAIL"} (${checks.filter(c=>c.pass).length}/${checks.length}); ${dest}`);
if (!pass) process.exitCode=1;
