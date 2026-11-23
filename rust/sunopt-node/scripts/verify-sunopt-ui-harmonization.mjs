import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const read = (name) => fs.readFileSync(path.join(root, name), "utf8");
const checks = [];
function verify(label, result) { checks.push({label, pass: Boolean(result)}); }
const theme=read("public/styles/sunopt-theme.css");
const nav=read("src/components/site-navigation.tsx");
const annual=read("src/app/debug/hot-water/page.tsx");
const accel=read("src/app/acceleration-strategy/page.tsx");
const approachB=read("src/app/approach-b/page.tsx");
const physics=read("src/app/documentation/physics/page.tsx");
const docs=read("src/app/documentation/code-path/page.tsx");
const cleanup=read("src/app/debug/project-cleanup/page.tsx");
const tank=read("src/app/tank-comparison/page.tsx");
verify("central style in static and Next", ["src/app/layout.tsx","public/presentation/index.html","public/project-structure.html"].every(f=>read(f).includes("/styles/sunopt-theme.css")));
verify("three permanent navigation choices", ["Jahressimulation","Modellvergleich","Tankvergleich"].every(t=>nav.includes(t)));
verify("active secondary item immediately after Tankvergleich", nav.includes("{contextItem && <NavLink") && nav.indexOf("{contextItem")>nav.indexOf("PRIMARY_ITEMS.map"));
verify("hover, overlay and icon-only touch disclosure", nav.includes("onPointerEnter={onHeaderPointerEnter}") && theme.includes(".sunopt-hover-pages { position: absolute") && !nav.includes("Weitere Seiten · aktiv"));
verify("drag to scroll retained", nav.includes("useDragToScroll"));
verify("annual simulation gets central form+button+results", annual.includes("sunopt-action-button") && annual.match(/sunopt-form-field/g)?.length>=5 && annual.includes("sunopt-value-box") && annual.includes("Jahressimulation"));
verify("A/B production buttons unchanged in tank page", tank.includes("TankComparisonModeSwitch") && tank.includes("setCausalPath"));
verify("acceleration removes duplicate bottom links", !accel.includes('href="/project-status"') && !accel.includes('href="/tank-comparison"'));
verify("Approach B informative sections no dark surface", !approachB.includes('bg-[var(--sunopt-ink)]'));
verify("code path CPU+CUDA stages in green left-right component", docs.includes("jitStages.map") && docs.includes("cudaStages.map") && docs.includes("SunOptFlowStep"));
verify("physics flow before equations and legend last", physics.indexOf('aria-label="Energiefluss"') < physics.indexOf('aria-label="Formeln"') && physics.indexOf('aria-label="Legende"') > physics.indexOf('aria-label="Formeln"'));
verify("math formulas native MathML", physics.match(/<Formula /g)?.length === 6 && physics.includes('<math display="block"'));
verify("project cleanup bounded EBUSY scan retry", cleanup.includes("attempt < 2") && cleanup.includes("RUN_SUNOPT_CLEANUP_EBUSY_DIAGNOSTIC.cmd"));
const report = {
  tool: "sunopt-ui-harmonization-v2-verify",
  createdAt: new Date().toISOString(),
  passed: checks.every(c=>c.pass),
  checks,
};
const dir=path.join(root, "updates", "outbox");
fs.mkdirSync(dir,{recursive:true});
const target=path.join(dir, "sunopt-ui-harmonization-v2-verify.json");
fs.writeFileSync(target, JSON.stringify(report,null,2)+"\n");
for(const c of checks) console.log(`${c.pass?'PASS':'FAIL'}: ${c.label}`);
console.log(`Report: ${target}`);
if(!report.passed) process.exitCode=1;
