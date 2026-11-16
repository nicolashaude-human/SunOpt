// Approach A6 — integrate A4 packed input + H5 output buffer at runtime level.
//
// Reference:
//   existing integrated object/context runtime path
//
// Candidate:
//   packed input + caller-owned output buffer + safe JS result decode
//
// No default production switch is changed. Adapters use A6 only when
// SUNOPT_A6_PACKED_INTO=1.
//
// Policy:
//   existing parity tests
//   exact runtime result parity
//   5 measured rounds, rotating order, median authoritative
//   >=10% required for acceptance

import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL, fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { performance } from "node:perf_hooks";

const here = path.dirname(fileURLToPath(import.meta.url));
const nativeRoot = path.resolve(here, "..");
const projectRoot = path.resolve(nativeRoot, "..", "..");
const outbox = path.join(projectRoot, "updates", "outbox");
const sourcePath = path.join(projectRoot, "src", "simulation", "rust-acceleration-scopes.ts");
const runtimePath = path.join(projectRoot, "src", "simulation", "rust-acceleration-runtime.ts");
const hashFile = path.join(projectRoot, "updates", "history", "hash.txt");
const failedBuild = path.join(outbox, "failed-build.txt");
const failedRun = path.join(outbox, "failed-run.txt");

fs.mkdirSync(outbox, {recursive:true});
for (const f of [failedBuild, failedRun]) { try { fs.unlinkSync(f); } catch {} }

function failure(file, stage, error, extra="") {
  fs.writeFileSync(file, [
    `stage=${stage}`,
    `time=${new Date().toISOString()}`,
    `error=${error?.stack ?? error}`,
    extra,
    "",
  ].join("\n"));
}
function die(message, extra="") {
  failure(failedRun, "approach-a-a6", message, extra);
  throw new Error(message);
}
function median(v) {
  const s=[...v].sort((a,b)=>a-b), m=Math.floor(s.length/2);
  return s.length%2?s[m]:(s[m-1]+s[m])/2;
}
function build() {
  const cmd=process.env.ComSpec||process.env.COMSPEC||"C:\\Windows\\System32\\cmd.exe";
  const r=spawnSync(cmd,["/d","/s","/c","npm run build"],{
    cwd:nativeRoot,encoding:"utf8",windowsHide:true,
    env:{...process.env,SUNOPT_CUDA_FAST_MATH:"0"},
  });
  fs.writeFileSync(path.join(outbox,"approach-a-a6-build.txt"),
    `status=${r.status}\n\n=== STDOUT ===\n${r.stdout??""}\n\n=== STDERR ===\n${r.stderr??""}\n`);
  if(r.error||r.status!==0){
    failure(failedBuild,"approach-a-a6-build",r.error??`build status ${r.status}`,`${r.stdout}\n${r.stderr}`);
    process.exit(2);
  }
}
function existing(rel,out) {
  const r=spawnSync(process.execPath,[path.join(projectRoot,rel)],{
    cwd:projectRoot,encoding:"utf8",windowsHide:true,
  });
  fs.writeFileSync(path.join(outbox,out),
    `status=${r.status}\n\n=== STDOUT ===\n${r.stdout??""}\n\n=== STDERR ===\n${r.stderr??""}\n`);
  if(r.error||r.status!==0) die(`Existing test failed: ${rel}`,`${r.stdout}\n${r.stderr}`);
}
function timed(fn,reps){
  const t0=performance.now(); let last;
  for(let i=0;i<reps;i++) last=fn();
  const totalMs=performance.now()-t0;
  return {totalMs,perCallUs:totalMs*1000/reps,last};
}
function summary(rows){
  const a=rows.map(r=>r.perCallUs);
  return {perCallUs:a,medianPerCallUs:median(a),minPerCallUs:Math.min(...a),maxPerCallUs:Math.max(...a)};
}
function assertSame(label,a,b){
  const ak=Object.keys(a),bk=Object.keys(b);
  if(ak.length!==bk.length) die(`${label}: shape mismatch`);
  for(const k of ak){
    const av=a[k],bv=b[k];
    if(typeof av==="number"&&typeof bv==="number"){
      const d=Math.abs(av-bv);
      if(!Number.isFinite(av)||!Number.isFinite(bv)||d>1e-12)
        die(`${label}.${k}: ref=${av} candidate=${bv} diff=${d}`);
    } else if(av!==bv) die(`${label}.${k}: mismatch`);
  }
}
function compare(ref,candidate,reps){
  assertSame("parity",ref(),candidate());
  timed(ref,1000); timed(candidate,1000);
  const rows={ref:[],candidate:[]};
  const order=["ref","candidate","candidate","ref","ref","candidate","candidate","ref","ref","candidate"];
  for(const v of order) rows[v].push(timed(v==="ref"?ref:candidate,reps));
  const a=summary(rows.ref),b=summary(rows.candidate);
  const deltaPct=100*(b.medianPerCallUs-a.medianPerCallUs)/a.medianPerCallUs;
  return {
    reference:a,candidate:b,deltaPct,
    speedup:a.medianPerCallUs/b.medianPerCallUs,
    conclusion:Math.abs(deltaPct)<10?"INCONCLUSIVE_LT_10_PERCENT":deltaPct<0?"A6_FASTER":"REFERENCE_FASTER",
    exactParity:true,
  };
}

build();
existing("scripts/test-global-acceleration-parity.mjs","approach-a-a6-parity.txt");
existing("scripts/test-global-acceleration-scopes.mjs","approach-a-a6-scopes.txt");

const runtime=await import(pathToFileURL(runtimePath).href+`?a6=${Date.now()}`);
const require=createRequire(import.meta.url);
const native=require(path.join(nativeRoot,"sunopt_node.win32-x64-msvc.node"));

for(const n of ["runNoSunshineKernelContextPackedInto","runPeakShavingCausalKernelContextPackedInto"])
  if(typeof native[n]!=="function") die(`Missing A6 native export: ${n}`);

const source=fs.readFileSync(sourcePath,"utf8");
const length=35040;
const pv=new Float64Array(length),load=new Float64Array(length);
for(let i=0;i<length;i++){
  const q=i%96,h=q/4;
  if(h>=6&&h<=18) pv[i]=8.5*Math.max(0,Math.sin((h-6)/12*Math.PI));
  if(q===28) load[i]=0.35; else if(q===80) load[i]=0.28;
}
const exp=Float64Array.from(pv,p=>Math.max(0,p-2));
const storage={
  storageEnergyPerKelvin:3000*4.186/3600,
  storageConductanceWPerKelvin:3.617,
  storageMinTemperature:60,
  ambientTemperature:20,
  dailyHeatLoss:-1,
};
const ns={
  currentIndex:173,horizonSteps:672,protectedSteps:192,timeStepHours:.25,
  householdReservePower:2,maximumHeatPumpElectricalPower:5/3,cop:3,
  initialWaterlinePower:10,targetStorageEnergy:52.325,currentStorageEnergy:10,...storage
};
const causal={
  currentIndex:173,horizonSteps:192,currentStorageEnergy:10,minimumReserveEnergy:2,
  targetStorageEnergy:52.325,maximumHeatPumpElectricalPower:5/3,cop:3,
  energyCutLevel:4,timeStepHours:.25,...storage
};

function nsObject(){
  return runtime.tryRunNativeScope({
    scopeId:"rust_noSunshine",exportName:"runNoSunshineKernel",args:[pv,load,ns]
  });
}
const nsPacked=new Float64Array(15),nsOut=new Float64Array(11);
function fillNS(){
  nsPacked[0]=ns.currentIndex;nsPacked[1]=ns.horizonSteps;nsPacked[2]=ns.protectedSteps;
  nsPacked[3]=ns.timeStepHours;nsPacked[4]=ns.householdReservePower;nsPacked[5]=ns.maximumHeatPumpElectricalPower;
  nsPacked[6]=ns.cop;nsPacked[7]=ns.initialWaterlinePower;nsPacked[8]=ns.targetStorageEnergy;
  nsPacked[9]=ns.currentStorageEnergy;nsPacked[10]=ns.storageEnergyPerKelvin;nsPacked[11]=ns.storageConductanceWPerKelvin;
  nsPacked[12]=ns.storageMinTemperature;nsPacked[13]=ns.ambientTemperature;nsPacked[14]=ns.dailyHeatLoss;
}
function decodeNS(o){return{
  kWhNoSunshine:o[0],bucketDemandEnergy:o[1],bucketCoolingLossEnergy:o[2],
  preBucketHoldingLossEnergy:o[3],allocatedPvThermalEnergy:o[4],unavoidableThermalDeficit:o[5],
  requiredStorageEnergyAtStart:o[6],additionalStorageEnergyNeededAtStart:o[7],currentWaterlinePower:o[8],
  currentScheduledHpElectricalPower:o[9],currentScheduledHpThermalPower:o[10],
};}
function nsA6(){
  fillNS();
  return runtime.tryRunNativeScopePackedInto({
    scopeId:"rust_noSunshine",firstStaticArray:pv,secondStaticArray:load,
    packedInput:nsPacked,outputBuffer:nsOut,decodeResult:decodeNS
  });
}

function causalObject(){
  return runtime.tryRunNativeScope({
    scopeId:"rust_peakShavingCausal",exportName:"runPeakShavingCausalKernel",args:[exp,load,causal]
  });
}
const cPacked=new Float64Array(14),cOut=new Float64Array(5);
function fillC(){
  cPacked[0]=causal.currentIndex;cPacked[1]=causal.horizonSteps;cPacked[2]=causal.currentStorageEnergy;
  cPacked[3]=causal.minimumReserveEnergy;cPacked[4]=causal.targetStorageEnergy;cPacked[5]=causal.maximumHeatPumpElectricalPower;
  cPacked[6]=causal.cop;cPacked[7]=causal.energyCutLevel;cPacked[8]=causal.timeStepHours;
  cPacked[9]=causal.storageEnergyPerKelvin;cPacked[10]=causal.storageConductanceWPerKelvin;
  cPacked[11]=causal.storageMinTemperature;cPacked[12]=causal.ambientTemperature;cPacked[13]=causal.dailyHeatLoss;
}
function decodeC(o){return{
  cutLevelPower:o[0],normalPlanSurvives:o[1]!==0,maximumPvPlanSurvives:o[2]!==0,
  survivalChecks:o[3],binarySearchIterations:o[4],
};}
function causalA6(){
  fillC();
  return runtime.tryRunNativeScopePackedInto({
    scopeId:"rust_peakShavingCausal",firstStaticArray:exp,secondStaticArray:load,
    packedInput:cPacked,outputBuffer:cOut,decodeResult:decodeC
  });
}

runtime.clearNativeAccelerationCache();
const comparisons={
  realisticNoSunshine:compare(nsObject,nsA6,5000),
  realisticCausal:compare(causalObject,causalA6,10000),
};

const accepted=Object.values(comparisons).every(x=>x.conclusion==="A6_FASTER");
const release=fs.existsSync(hashFile)?fs.readFileSync(hashFile,"utf8").trim():"unknown";
const report={
  stage:"APPROACH_A_A6_RUNTIME_PACKED_INTO",
  release,
  integration:"A4 packed input + H5 caller-owned output + safe JS decode",
  adapterGate:"SUNOPT_A6_PACKED_INTO=1",
  parity:{existingTests:"PASS",runtimeComparisons:"EXACT"},
  measurementPolicy:{measuredRounds:5,medianAuthoritative:true,noiseThresholdPercent:10},
  comparisons,
  decision:accepted?"ENABLE_A6_DEFAULT_CANDIDATE":"KEEP_A6_EXPERIMENTAL",
  nextCandidate:accepted
    ?"Run full production annual/end-to-end A/B with SUNOPT_A6_PACKED_INTO=0 vs 1 before making A6 the default."
    :"Do not default-enable A6; keep current production path.",
};
const reportPath=path.join(outbox,"approach-a-a6-runtime-packed-into.json");
fs.writeFileSync(reportPath,JSON.stringify(report,null,2)+"\n");
const llm=path.join(outbox,"llm.txt");
fs.appendFileSync(llm,[
  "",`=== ${new Date().toISOString()} APPROACH_A_A6_RUNTIME_PACKED_INTO ===`,
  `release=${release}`,"parity existing=PASS runtimeComparisons=EXACT",
  ...Object.entries(comparisons).map(([n,x])=>`${n} refUs=${x.reference.medianPerCallUs.toFixed(6)} A6Us=${x.candidate.medianPerCallUs.toFixed(6)} speedup=${x.speedup.toFixed(4)}x deltaPct=${x.deltaPct.toFixed(3)} conclusion=${x.conclusion}`),
  `decision=${report.decision}`,`nextCandidate=${report.nextCandidate}`,`report=${reportPath}`,""
].join("\n"));
console.log(JSON.stringify(report,null,2));
console.log(`report=${reportPath}`);
console.log("APPROACH_A_A6_RUNTIME_PACKED_INTO=OK");
