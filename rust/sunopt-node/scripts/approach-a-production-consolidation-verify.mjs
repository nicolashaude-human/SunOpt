// Approach A production consolidation verification.
//
// This is NOT an optimisation benchmark. It verifies the consolidated A1+A2
// production path, the two retained optional packed utilities, removal of the
// rejected experiment exports, and one production annual result against the
// canonical A7 reference physics digest.

import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import crypto from "node:crypto";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { spawn, spawnSync } from "node:child_process";
import { performance } from "node:perf_hooks";

const here = path.dirname(fileURLToPath(import.meta.url));
const nativeRoot = path.resolve(here, "..");
const projectRoot = path.resolve(nativeRoot, "..", "..");
const outbox = path.join(projectRoot, "updates", "outbox");
const hashFile = path.join(projectRoot, "updates", "history", "hash.txt");
const reportPath = path.join(outbox, "approach-a-production-consolidation-verify.json");
const failedBuild = path.join(outbox, "failed-build.txt");
const failedRun = path.join(outbox, "failed-run.txt");
const serverLogPath = path.join(outbox, "approach-a-production-consolidation-server.log");
const EXPECTED_PHYSICS_SHA256 = "e8a1caf0ddeb8aae89c22cc5049fa8cfa2bdb3464c12dca3113510ad40009672";

fs.mkdirSync(outbox, { recursive: true });
for (const f of [failedBuild, failedRun]) { try { fs.unlinkSync(f); } catch {} }

function fail(file, stage, error, extra = "") {
  fs.writeFileSync(file, [
    `stage=${stage}`,
    `time=${new Date().toISOString()}`,
    `error=${error?.stack ?? error}`,
    extra,
    "",
  ].join("\n"), "utf8");
}
function die(stage, message, extra = "") {
  fail(failedRun, stage, message, extra);
  throw new Error(message);
}
function runChecked(relative, outName) {
  const r = spawnSync(process.execPath, [path.join(projectRoot, relative)], {
    cwd: projectRoot, encoding: "utf8", windowsHide: true,
  });
  fs.writeFileSync(path.join(outbox, outName),
    `status=${r.status}\n\n=== STDOUT ===\n${r.stdout ?? ""}\n\n=== STDERR ===\n${r.stderr ?? ""}\n`);
  if (r.error || r.status !== 0) die("existing-test", `Existing test failed: ${relative}`, `${r.stdout}\n${r.stderr}`);
}
function buildNative() {
  const cmd = process.env.ComSpec || process.env.COMSPEC || "C:\\Windows\\System32\\cmd.exe";
  const r = spawnSync(cmd, ["/d","/s","/c","npm run build"], {
    cwd: nativeRoot, encoding: "utf8", windowsHide: true,
    env: { ...process.env, SUNOPT_CUDA_FAST_MATH: "0" },
  });
  fs.writeFileSync(path.join(outbox, "approach-a-production-consolidation-build.txt"),
    `status=${r.status}\n\n=== STDOUT ===\n${r.stdout ?? ""}\n\n=== STDERR ===\n${r.stderr ?? ""}\n`);
  if (r.error || r.status !== 0) {
    fail(failedBuild, "native-build", r.error ?? `npm run build exited ${r.status}`, `${r.stdout}\n${r.stderr}`);
    process.exit(2);
  }
}
function assertVector(label, actual, expected) {
  const a = Array.from(actual), b = Array.from(expected);
  if (a.length !== b.length) die("utility-parity", `${label}: length mismatch`);
  for (let i=0;i<a.length;i++) {
    if (Math.abs(a[i]-b[i]) > 1e-12) die("utility-parity", `${label}[${i}] ${a[i]} != ${b[i]}`);
  }
}
function encodeNs(r) { return [
  r.kWhNoSunshine,r.bucketDemandEnergy,r.bucketCoolingLossEnergy,r.preBucketHoldingLossEnergy,
  r.allocatedPvThermalEnergy,r.unavoidableThermalDeficit,r.requiredStorageEnergyAtStart,
  r.additionalStorageEnergyNeededAtStart,r.currentWaterlinePower,
  r.currentScheduledHpElectricalPower,r.currentScheduledHpThermalPower,
]; }
function encodeC(r) { return [
  r.cutLevelPower,r.normalPlanSurvives?1:0,r.maximumPvPlanSurvives?1:0,
  r.survivalChecks,r.binarySearchIterations,
]; }
function fillNs(p) { return Float64Array.from([
  p.currentIndex,p.horizonSteps,p.protectedSteps,p.timeStepHours,p.householdReservePower,
  p.maximumHeatPumpElectricalPower,p.cop,p.initialWaterlinePower,p.targetStorageEnergy,
  p.currentStorageEnergy,p.storageEnergyPerKelvin,p.storageConductanceWPerKelvin,
  p.storageMinTemperature,p.ambientTemperature,p.dailyHeatLoss,
]); }
function fillC(p) { return Float64Array.from([
  p.currentIndex,p.horizonSteps,p.currentStorageEnergy,p.minimumReserveEnergy,p.targetStorageEnergy,
  p.maximumHeatPumpElectricalPower,p.cop,p.energyCutLevel,p.timeStepHours,p.storageEnergyPerKelvin,
  p.storageConductanceWPerKelvin,p.storageMinTemperature,p.ambientTemperature,p.dailyHeatLoss,
]); }

function staticSourceGuards() {
  const runtime = fs.readFileSync(path.join(projectRoot,"src","simulation","rust-acceleration-runtime.ts"),"utf8");
  const globalScope = fs.readFileSync(path.join(projectRoot,"src","simulation","global-acceleration-scope.ts"),"utf8");
  const adapters = fs.readFileSync(path.join(projectRoot,"src","simulation","rust-acceleration-adapters.ts"),"utf8");
  const combined = `${runtime}\n${globalScope}\n${adapters}`;
  for (const forbidden of ["SUNOPT_A6_PACKED_INTO","tryRunNativeScopePackedInto","runPackedInto<","APPROACH A6 — PACKED-INTO RESULT DECODERS"]) {
    if (combined.includes(forbidden)) die("source-guard", `Rejected production experiment routing still present: ${forbidden}`);
  }
  for (const required of ["registerAccelerationScope","registerAccelerationDataPair","runNoSunshineKernelContext","runPeakShavingCausalKernelContext"]) {
    if (!runtime.includes(required)) die("source-guard", `Consolidated A1/A2 production primitive missing: ${required}`);
  }
}

function verifyNativeUtilities() {
  const require = createRequire(import.meta.url);
  const native = require(path.join(nativeRoot, "sunopt_node.win32-x64-msvc.node"));
  const required = [
    "registerAccelerationScope","registerAccelerationDataPair",
    "runNoSunshineKernelContext","runPeakShavingCausalKernelContext",
    "runNoSunshineKernelContextBatchPacked","runPeakShavingCausalKernelContextBatchPacked",
    "runNoSunshineKernelContextPackedInto","runPeakShavingCausalKernelContextPackedInto",
  ];
  for (const name of required) if (typeof native[name] !== "function") die("native-export", `Required export missing: ${name}`);

  const removed = [
    "runNoSunshineKernelContextBatch","runPeakShavingCausalKernelContextBatch",
    "runNoSunshineKernelContextPacked","runPeakShavingCausalKernelContextPacked",
    "a5RegisterNoSunshineH1Context","a5RunNoSunshineH1","a5RegisterCausalH1Context","a5RunCausalH1",
    "a5RunNoSunshineNumericOutput","a5RunCausalNumericOutput",
    "a5RegisterNoSunshineH3Context","a5RunNoSunshineH3","a5RegisterCausalH3Context","a5RunCausalH3",
    "a5RunNoSunshineTrusted","a5RunCausalTrusted","a5RunNoSunshineInto","a5RunCausalInto",
    "a5RunNoSunshineScalars","a5RunCausalScalars",
  ];
  for (const name of removed) if (typeof native[name] === "function") die("native-export", `Rejected experimental export still present: ${name}`);

  const source = fs.readFileSync(path.join(projectRoot,"src","simulation","rust-acceleration-scopes.ts"),"utf8");
  const length=35040, pv=new Float64Array(length), load=new Float64Array(length);
  for (let i=0;i<length;i++) { const q=i%96,h=q/4; if(h>=6&&h<=18) pv[i]=8.5*Math.max(0,Math.sin((h-6)/12*Math.PI)); if(q===28) load[i]=0.35; else if(q===80) load[i]=0.28; }
  const exp=Float64Array.from(pv,p=>Math.max(0,p-2));
  const storage={storageEnergyPerKelvin:3000*4.186/3600,storageConductanceWPerKelvin:3.617,storageMinTemperature:60,ambientTemperature:20,dailyHeatLoss:-1};
  const ns={currentIndex:173,horizonSteps:672,protectedSteps:192,timeStepHours:.25,householdReservePower:2,maximumHeatPumpElectricalPower:5/3,cop:3,initialWaterlinePower:10,targetStorageEnergy:52.325,currentStorageEnergy:10,...storage};
  const causal={currentIndex:173,horizonSteps:192,currentStorageEnergy:10,minimumReserveEnergy:2,targetStorageEnergy:52.325,maximumHeatPumpElectricalPower:5/3,cop:3,energyCutLevel:4,timeStepHours:.25,...storage};
  native.clearAccelerationScopeCache();
  const nsScope=native.registerAccelerationScope(source,"rust_noSunshine"), cScope=native.registerAccelerationScope(source,"rust_peakShavingCausal");
  const nsData=native.registerAccelerationDataPair(pv,load), cData=native.registerAccelerationDataPair(exp,load);
  const nsRef=native.runNoSunshineKernelContext(nsScope,nsData,ns), cRef=native.runPeakShavingCausalKernelContext(cScope,cData,causal);
  const nsPacked=fillNs(ns), cPacked=fillC(causal), nsOut=new Float64Array(11), cOut=new Float64Array(5);
  native.runNoSunshineKernelContextPackedInto(nsScope,nsData,nsPacked,nsOut);
  native.runPeakShavingCausalKernelContextPackedInto(cScope,cData,cPacked,cOut);
  assertVector("packedInto.noSunshine",nsOut,encodeNs(nsRef));
  assertVector("packedInto.causal",cOut,encodeC(cRef));
  const nsBatch=native.runNoSunshineKernelContextBatchPacked(nsScope,nsData,nsPacked);
  const cBatch=native.runPeakShavingCausalKernelContextBatchPacked(cScope,cData,cPacked);
  assertVector("packedBatch.noSunshine",encodeNs(nsBatch[0]),encodeNs(nsRef));
  assertVector("packedBatch.causal",encodeC(cBatch[0]),encodeC(cRef));
  return { requiredExports: required, removedExports: removed, packedUtilitiesExact: true };
}

function probeTcp(port, timeoutMs=1000) { return new Promise(resolve => {
  const s=net.createConnection({host:"127.0.0.1",port}); let done=false;
  const finish=v=>{if(done)return;done=true;s.destroy();resolve(v)};
  s.setTimeout(timeoutMs); s.once("connect",()=>finish(true)); s.once("timeout",()=>finish(false)); s.once("error",()=>finish(false));
}); }
async function waitReady(child, port, timeoutMs=120000) {
  const t0=performance.now();
  while(performance.now()-t0<timeoutMs) {
    if(child.exitCode!==null) die("server-start",`Next server exited with code ${child.exitCode}`,`log=${serverLogPath}`);
    if(await probeTcp(port)){await new Promise(r=>setTimeout(r,500)); return;}
    await new Promise(r=>setTimeout(r,250));
  }
  die("server-start-timeout",`Next server not listening on ${port}`,`log=${serverLogPath}`);
}
async function stopServer(child) {
  if(child.exitCode!==null)return;
  if(process.platform==="win32") await new Promise(resolve=>{const k=spawn("taskkill",["/PID",String(child.pid),"/T","/F"],{windowsHide:true,stdio:"ignore"});k.on("exit",resolve);k.on("error",resolve);});
  else { try { child.kill("SIGTERM"); } catch {} }
}
const requestBody={heatPumpThermalPower:5,pvPeakPower:10,gridPriceChfPerKWh:.3,feedInPriceChfPerKWh:.08};
async function requestAnnual(port, kind) {
  const t0=performance.now();
  const r=await fetch(`http://127.0.0.1:${port}/api/tank-comparison`,{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(requestBody),signal:AbortSignal.timeout(180000)});
  const wall=performance.now()-t0, text=await r.text();
  if(!r.ok) die("annual-http",`${kind} annual request HTTP ${r.status}`,text.slice(0,5000));
  const body=JSON.parse(text); if(!Array.isArray(body.points)||body.points.length!==28) die("annual-shape",`${kind} does not contain 28 points`);
  return {wall,body};
}
function physicsDigest(body) {
  const keys=["gridImport","feedIn","selfConsumption","storageLosses","unmetThermalDemand","gridCost","feedInRevenue","netEnergyCost"];
  const canonical={fake:body.fake,year:body.year,heatPumpThermalPower:body.heatPumpThermalPower,pvPeakPower:body.pvPeakPower,gridPriceChfPerKWh:body.gridPriceChfPerKWh,feedInPriceChfPerKWh:body.feedInPriceChfPerKWh,points:body.points.map(p=>Object.fromEntries([["storageVolume",p.storageVolume],...keys.map(k=>[k,p[k]])]))};
  return crypto.createHash("sha256").update(JSON.stringify(canonical)).digest("hex");
}
async function verifyAnnual() {
  const port=3111, cmd=process.env.ComSpec||process.env.COMSPEC||"C:\\Windows\\System32\\cmd.exe";
  const log=fs.createWriteStream(serverLogPath,{flags:"w"});
  const env={...process.env,SUNOPT_CUDA_FAST_MATH:"0"}; delete env.SUNOPT_A6_PACKED_INTO;
  const child=spawn(cmd,["/d","/s","/c",`npm run dev -- --hostname 127.0.0.1 --port ${port}`],{cwd:projectRoot,windowsHide:true,env,stdio:["ignore","pipe","pipe"]});
  child.stdout?.pipe(log); child.stderr?.pipe(log);
  try {
    await waitReady(child,port);
    await requestAnnual(port,"warmup");
    const measured=await requestAnnual(port,"measured");
    const digest=physicsDigest(measured.body);
    if(digest!==EXPECTED_PHYSICS_SHA256) die("annual-parity",`Production physics digest mismatch: ${digest} != ${EXPECTED_PHYSICS_SHA256}`);
    return {exactReferenceDigest:true,physicsSha256:digest,requestWallMs:measured.wall,apiTotalDurationMs:measured.body.totalDurationMs,sweepDurationMs:measured.body.sweepDurationMs};
  } finally { await stopServer(child); log.end(); }
}

async function main(){
  staticSourceGuards();
  buildNative();
  runChecked("scripts/test-global-acceleration-parity.mjs","approach-a-production-consolidation-parity.txt");
  runChecked("scripts/test-global-acceleration-scopes.mjs","approach-a-production-consolidation-scopes.txt");
  const native=verifyNativeUtilities();
  const annual=await verifyAnnual();
  const release=fs.existsSync(hashFile)?fs.readFileSync(hashFile,"utf8").trim():"unknown";
  const report={stage:"APPROACH_A_PRODUCTION_CONSOLIDATION_VERIFY",release,productionDefault:"A1_REGISTERED_SOURCE_PLUS_A2_PERSISTENT_STATIC_CONTEXT",sourceGuards:"PASS",existingParity:"PASS",existingScopes:"PASS",native,annual,decision:"CONSOLIDATION_ACCEPTED"};
  fs.writeFileSync(reportPath,JSON.stringify(report,null,2)+"\n");
  fs.appendFileSync(path.join(outbox,"llm.txt"),["",`=== ${new Date().toISOString()} APPROACH_A_PRODUCTION_CONSOLIDATION_VERIFY ===`,`release=${release}`,"productionDefault=A1+A2","sourceGuards=PASS existingParity=PASS existingScopes=PASS packedUtilities=EXACT",`annualPhysicsSha256=${annual.physicsSha256} exactReferenceDigest=${annual.exactReferenceDigest}`,`annual wall=${annual.requestWallMs.toFixed(3)}ms api=${Number(annual.apiTotalDurationMs).toFixed(3)}ms sweep=${Number(annual.sweepDurationMs).toFixed(3)}ms`,`decision=${report.decision}`,`report=${reportPath}`,""].join("\n"));
  console.log(JSON.stringify(report,null,2));
  console.log(`report=${reportPath}`);
  console.log("APPROACH_A_PRODUCTION_CONSOLIDATION_VERIFY=OK");
}
main().catch(error=>{if(!fs.existsSync(failedRun))fail(failedRun,"unhandled",error);console.error(error);process.exitCode=1;});
