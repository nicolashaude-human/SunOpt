// rust/sunopt-node/scripts/fix-blackwell-b14i1-generator.mjs
//
// Fixes the four accidentally escaped template literals at the end of
// make-blackwell-b14i1-lockstep.mjs.
//
// Run from project root:
//   node .\rust\sunopt-node\scripts\fix-blackwell-b14i1-generator.mjs
//
// Safe/idempotent: only the exact malformed tail expressions are replaced.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const target = path.join(here, "make-blackwell-b14i1-lockstep.mjs");

if (!fs.existsSync(target)) {
  throw new Error(`Generator not found: ${target}`);
}

let text = fs.readFileSync(target, "utf8");

const replacements = [
  ['console.log(\\`parent=\\${parent}\\`);', 'console.log(`parent=${parent}`);'],
  ['console.log(\\`release=\\${release}\\`);', 'console.log(`release=${release}`);'],
  ['console.log(\\`package=\\${packagePath}\\`);', 'console.log(`package=${packagePath}`);'],
];

let changed = 0;

for (const [bad, good] of replacements) {
  if (text.includes(bad)) {
    text = text.replaceAll(bad, good);
    changed++;
  }
}

fs.writeFileSync(target, text, "utf8");

console.log(`target=${target}`);
console.log(`replacements=${changed}`);
console.log("B14I1_GENERATOR_FIX_RESULT=OK");
