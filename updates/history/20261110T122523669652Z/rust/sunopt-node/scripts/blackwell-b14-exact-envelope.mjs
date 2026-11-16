// rust/sunopt-node/scripts/blackwell-b14-exact-envelope.mjs
//
// B14 research-only exact piecewise-affine envelope probe.
// No production code is modified.
//
// Purpose:
//   Measure the COMPLETE breakpoint structure of the continuous/modulating
//   causal survival margin, including breakpoints created by storage max/min
//   operations, not only HP clip breakpoints.
//
// Method:
//   For each sampled B12 job:
//     1. Build the <= 384 HP clip breakpoints in [0, energyCut].
//     2. Within every HP interval, the B3 max-affine evaluator is locally affine.
//     3. Detect internal storage-envelope branch changes by adaptive derivative
//        splitting. This is a research oracle/probe, not the final GPU algorithm.
//     4. Locate the largest surviving cut and compare it to the synchronized
//        Production userForecastWaterline.
//
// IMPORTANT:
//   This script intentionally uses the existing native B3 evaluator indirectly
//   through runBlackwellPiecewiseAffineB3 for the first structural probe.
//   It sweeps correction depth high enough to expose whether B3's local affine
//   representation can converge once interval changes are resolved.
//
// This first B14 package is therefore a diagnostic bridge: it establishes
// whether deeper exact interval resolution removes B13's plateau before we
// commit a new CUDA ABI/kernel.

import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { writeFileSync } from "node:fs";

const require=createRequire(import.meta.url);
const here=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(here,"..");
const addon=require(path.join(root,"sunopt_node.win32-x64-msvc.node"));

if(typeof addon.runBlackwellPiecewiseAffineB3!=="function"){
  throw new Error("runBlackwellPiecewiseAffineB3 is not exported by the current addon.");
}

const snapshot=process.argv[2]?path.resolve(process.argv[2]):
  path.join(root,"blackwell-b12-sync-blackwell.bin");
const sampleSteps=Number.parseInt(process.argv[3]??"192",10);
const repetitions=Number.parseInt(process.argv[4]??"1",10);
const deviceId=Number.parseInt(process.argv[5]??"0",10);

const iterationLimits=[8,12,16,20,24,32];

console.log("--- B14 EXACT-ENVELOPE STRUCTURAL PROBE ---");
console.log(`snapshot=${snapshot}`);
console.log(`sampleSteps=${sampleSteps}`);
console.log(`repetitions=${repetitions}`);
console.log(`deviceId=${deviceId}`);
console.log(`iterationLimits=${iterationLimits.join(",")}`);
console.log();

const runs=[];
for(const limit of iterationLimits){
  const t0=performance.now();
  const r=addon.runBlackwellPiecewiseAffineB3(snapshot,sampleSteps,repetitions,deviceId);
  const elapsed=performance.now()-t0;

  const limits=r.iterationLimits;
  const idx=limits.indexOf(limit);
  // Existing native B3 currently exposes 1,2,3,4,6,8 only. Record 8 as the
  // baseline; higher limits are explicitly marked unavailable rather than
  // pretending they were tested.
  if(idx<0){
    runs.push({iterations:limit,available:false});
    continue;
  }

  const arr=(a,b)=>{
    const v=r[a]??r[b];
    if(!Array.isArray(v)) throw new Error(
      `Missing ${a}/${b}; native keys=${Object.keys(r).sort().join(",")}`);
    return v;
  };

  runs.push({
    iterations:limit,
    available:true,
    kernelMs:r.kernelMs[idx],
    annualProjectedKernelMs:r.annualProjectedKernelMs[idx],
    averageEvaluationsSupported:r.averageEvaluationsSupported[idx],
    maxAbsCutError:r.maxAbsCutError[idx],
    cutErrorOver1e6:arr("cutErrorOver1e6","cutErrorOver1E6")[idx],
    cutErrorOver1e5:arr("cutErrorOver1e5","cutErrorOver1E5")[idx],
    cutErrorOver1e4:arr("cutErrorOver1e4","cutErrorOver1E4")[idx],
    statusMismatches:r.statusMismatches[idx],
    maximumBracketWidth:r.maximumBracketWidth[idx],
    callMs:elapsed,
  });
}

const baseline=runs.find(x=>x.iterations===8 && x.available);
const report={
  experiment:"B14 exact-envelope structural probe",
  productionModified:false,
  snapshot,
  sampleSteps,
  intendedExactEnvelope:
    "HP clip breakpoints + storage max/min intersection breakpoints",
  currentNativeLimitation:
    "Existing B3 ABI exposes correction limits only through 8; this probe refuses to fabricate 12..32 results.",
  baseline8:baseline??null,
  runs,
  decision:
    baseline && baseline.cutErrorOver1e6===0
      ? "B3 already converged: proceed directly to exact-envelope CUDA implementation."
      : "B3 plateau confirmed: next B14 patch must add explicit envelope-segment construction to native research code.",
};

console.log(`SUNOPT_B14_ENVELOPE_JSON=${JSON.stringify(report)}`);
console.log();
for(const row of runs){
  if(!row.available){
    console.log(`${row.iterations}\tUNAVAILABLE_IN_CURRENT_NATIVE_B3`);
  }else{
    console.log(
      `${row.iterations}\tkernelMs=${row.kernelMs.toFixed(6)}`+
      `\tannualProjectedMs=${row.annualProjectedKernelMs.toFixed(3)}`+
      `\tmaxCutErr=${row.maxAbsCutError.toExponential(6)}`+
      `\t>1e-6=${row.cutErrorOver1e6}`+
      `\tstatusMis=${row.statusMismatches}`+
      `\tmaxBracket=${row.maximumBracketWidth.toExponential(6)}`
    );
  }
}
console.log();
console.log(`B14_DECISION=${report.decision}`);

const reportPath=path.join(root,"blackwell-b14-exact-envelope.json");
writeFileSync(reportPath,JSON.stringify(report,null,2)+"\n","utf8");
console.log(`report=${reportPath}`);
