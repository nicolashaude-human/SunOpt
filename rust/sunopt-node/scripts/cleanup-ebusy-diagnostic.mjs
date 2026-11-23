// Read-only diagnostics for the Project Cleanup EBUSY issue.
// The earlier UI scanner explicitly excluded API routes and the scanner lib.
// This script collects ONLY these two relevant code directories, not secrets.
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const outbox = path.join(root, "updates", "outbox");
const when = new Date().toISOString().replace(/[:.]/g, "-");
const relativeTargets = [
  "src/app/api/debug/project-cleanup/scan",
  "src/lib/project-cleanup",
];
const report = {
  tool: "sunopt-cleanup-ebusy-source-diagnostic-v1",
  createdAt: new Date().toISOString(),
  readOnly: true,
  purpose: "Diagnose EBUSY im Cleanup-Scan-Endpunkt; keine Backend-Änderung",
  files: [],
  warnings: [],
};
function visit(relative, depth=0) {
  if (depth > 4 || report.files.length >= 40) return;
  const full = path.join(root, relative);
  let stat;
  try { stat = fs.statSync(full); } catch(err) {
    report.warnings.push(`${relative}: ${err.code || String(err)}`);
    return;
  }
  if (stat.isDirectory()) {
    let entries;
    try { entries = fs.readdirSync(full, {withFileTypes:true}); } catch(err) {
      report.warnings.push(`${relative}: ${err.code || String(err)}`);
      return;
    }
    for (const e of entries) {
      if (e.isSymbolicLink()) continue;
      if (e.isDirectory() || /\.(ts|tsx|mjs)$/.test(e.name)) visit(path.posix.join(relative, e.name), depth+1);
    }
    return;
  }
  if (stat.size > 300000 || !/\.(ts|tsx|mjs)$/.test(relative)) return;
  try {
    const source = fs.readFileSync(full, "utf8");
    // Only source code from this dedicated cleanup subsystem; never .env or generic files.
    report.files.push({
      path: relative,
      size: stat.size,
      sha256: crypto.createHash("sha256").update(source).digest("hex"),
      source,
    });
  } catch(err) { report.warnings.push(`${relative}: ${err.code || String(err)}`); }
}
for (const p of relativeTargets) visit(p);
fs.mkdirSync(outbox, { recursive:true });
const dest = path.join(outbox, `sunopt-cleanup-ebusy-diagnostic-${when}.json`);
fs.writeFileSync(dest, JSON.stringify(report, null, 2) + "\n");
console.log(`Diagnose gespeichert: ${dest}`);
console.log(`${report.files.length} relevante Quelltextdateien, ${report.warnings.length} Hinweise`);
