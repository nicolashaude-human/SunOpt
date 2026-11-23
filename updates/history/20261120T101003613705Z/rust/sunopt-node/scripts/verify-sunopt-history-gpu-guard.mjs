// Passive source checks: does NOT change GPU clock, fans, or productive files.
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
const root = process.cwd();
const acc = readFileSync(join(root, "src/app/acceleration-strategy/page.tsx"), "utf8");
const b = readFileSync(join(root, "src/app/approach-b/page.tsx"), "utf8");
const route = readFileSync(join(root, "src/app/api/tank-comparison/route.ts"), "utf8");
const guard = readFileSync(join(root, "rust/sunopt-node/scripts/sunopt-gpu-profile-guard.py"), "utf8");
const rows = [
  ["TypeScript baseline", 1, 200],
  ["Next.js", 1, 170],
  ["Oxc/Cranelift", 1, 27],
  ["CPU optimized", 1, 5.2],
  ["Serial 28-tank", 28, 148.4553],
  ["i7-9700 6 worker", 28, 47.03],
  ["i5-8400 6 worker", 28, 35.5852],
  ["CUDA A earlier", 28, 7.1083],
  ["B14Q A", 28, 6.264],
  ["B14Q B", 28, 3.55],
].map(([stage,tanks,seconds]) => ({stage,tanks,seconds,tanksPerSecond:tanks/seconds,relativeToFirst200Seconds:(tanks/seconds)/0.005}));
const checks = {
  baseline200s: acc.includes("BASELINE_TANKS_PER_SECOND = 1 / 200"),
  historical170s: acc.includes("wallSeconds: 170"),
  jit27s: acc.includes("wallSeconds: 27"),
  cpu5_2s: acc.includes("wallSeconds: 5.2"),
  serialTotalWall148s: acc.includes("wallSeconds: 148.4553"),
  comparedViaThroughput: acc.includes("(Tanks/s) / 0,005"),
  b14qClockQualified: b.includes("3480 MHz") && b.includes("70 %"),
  guardForProductionAAndB: route.includes("startSunoptGpuProfileGuard(causalPath)"),
  guardRestoredAfterSweep: route.includes("await gpuGuard.stop()"),
  exactClockRequest: guard.includes("TARGET_SM_MHZ = 3480"),
  exactFanRequest: guard.includes("TARGET_FAN_PERCENT = 70"),
  runtimeTelemetry: guard.includes("underLoadSmClockMedianMHz"),
  temperatureFailsafe: guard.includes("MAX_GPU_TEMP_C = 80"),
  resetsClock: guard.includes("nvmlDeviceResetGpuLockedClocks"),
  resetsFan: guard.includes("nvmlDeviceSetDefaultFanSpeed_v2"),
};
const passed = Object.values(checks).every(Boolean);
const report = { status: passed ? "PASS" : "FAIL", createdAt: new Date().toISOString(), checks, baselineTanksPerSecond:0.005, rows, note: "Static checks only; 3480 MHz/70% require RTX 5060 Windows/NVML runtime probe and telemetry." };
mkdirSync(join(root,"updates/outbox"),{recursive:true});
const path=join(root,"updates/outbox","sunopt-gpu-history-verify.json");
writeFileSync(path,JSON.stringify(report,null,2)+"\n");
if(!passed) writeFileSync(join(root,"updates/outbox","failed-run.txt"),"SunOpt GPU/history source check failed. See sunopt-gpu-history-verify.json\n");
console.log(JSON.stringify(report,null,2));
process.exit(passed?0:1);
