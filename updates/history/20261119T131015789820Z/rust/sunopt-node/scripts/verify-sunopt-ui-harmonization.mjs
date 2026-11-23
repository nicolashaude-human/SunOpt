import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const targets = {
  theme: "public/styles/sunopt-theme.css",
  navigation: "src/components/site-navigation.tsx",
  drag: "src/components/ui/use-drag-to-scroll.ts",
  widgets: "src/components/ui/sunopt-documentation.tsx",
  docs: "src/app/documentation/code-path/page.tsx",
  layout: "src/app/layout.tsx",
  presentation: "public/presentation/index.html",
  structure: "public/project-structure.html",
};
const files = Object.fromEntries(Object.entries(targets).map(([key, name]) => {
  const full = path.join(root, name);
  if (!fs.existsSync(full)) throw new Error(`MISSING ${name}`);
  return [key, fs.readFileSync(full, "utf8")];
}));
const checks = [
  ["one shared theme", files.theme.includes("--sunopt-green-border")],
  ["Next.js theme link", files.layout.includes("/styles/sunopt-theme.css")],
  ["presentation theme link", files.presentation.includes("/styles/sunopt-theme.css")],
  ["Project Structure theme link", files.structure.includes("/styles/sunopt-theme.css")],
  ["navigation explicit menu", files.navigation.includes('aria-expanded={menuOpen}')],
  ["navigation drag", files.navigation.includes("useDragToScroll")],
  ["navigation keeps primary routes", ["/debug/hot-water","/model-comparison","/tank-comparison"].every((r) => files.navigation.includes(r))],
  ["docs stages", ["Historische Vektoren","Funktions-Wrapper","Worker Manager","Oxc-Parser","Cranelift"].every((s) => files.docs.includes(s))],
  ["documentation shared component", files.docs.includes("SunOptFlowStep")],
];
for (const [title, ok] of checks) console.log(`${ok ? "PASS" : "FAIL"}: ${title}`);
if (checks.some(([, ok]) => !ok)) process.exitCode = 1;
