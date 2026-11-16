// rust/sunopt-node/scripts/blackwell-b14f-warp-precheck.mjs
// B14E vs B14F: exact PWA envelope with serial vs warp max-affine prechecks.
import {createRequire} from "node:module";
import path from "node:path";
import {fileURLToPath} from "node:url";

const require=createRequire(import.meta.url);
const here=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(here,"..");
const addon=require(path.join(root,"sunopt_node.win32-x64-msvc.node"));

function pickExport(names){
  for(const name of names){
    if(typeof addon[name]==="function") return addon[name];
  }
  return null;
}

const runB14e=pickExport([
  "runBlackwellB14EWarpParallel",
  "runBlackwellB14eWarpParallel",
]);
const runB14f=pickExport([
  "runBlackwellB14FWarpPrecheck",
  "runBlackwellB14fWarpPrecheck",
]);

if(runB14e===null||runB14f===null){
  const exports=Object.keys(addon).filter((key)=>/b14|warp|precheck|parallel/i.test(key)).sort();
  throw new Error(`B14E/B14F native export missing. Available: ${JSON.stringify(exports)}`);
}

const snapshot=process.argv[2]?path.resolve(process.argv[2]):path.join(root,"blackwell-b12-sync-blackwell.bin");
const sampleSteps=Number.parseInt(process.argv[3]??"2048",10);
const repetitions=Number.parseInt(process.argv[4]??"1",10);
const deviceId=Number.parseInt(process.argv[5]??"0",10);

console.log("--- B14F CUDA WARP-PRECHECK A/B ---");
console.log(`snapshot=${snapshot}`);
console.log(`sampleSteps=${sampleSteps} repetitions=${repetitions} deviceId=${deviceId}`);

const e=runB14e(snapshot,sampleSteps,repetitions,deviceId);
const f=runB14f(snapshot,sampleSteps,repetitions,deviceId);
const speedup=f.kernelMs>0?e.kernelMs/f.kernelMs:0;
const sameChecksum=e.checksumHex===f.checksumHex;
const exact=f.statusMismatches===0&&f.cutMismatches1E12===0&&f.overflow32Jobs===0&&f.scratchOverflowJobs===0;

console.log(`jobs=${f.sampleJobCount} supported=${f.supportedJobs} unsupported=${f.unsupportedJobs}`);
console.log(`overflow32=${f.overflow32Jobs} scratchOverflow=${f.scratchOverflowJobs}`);
console.log(`statusMismatches=${f.statusMismatches}`);
console.log(`cutMismatch >1e-12=${f.cutMismatches1E12} >1e-9=${f.cutMismatches1E9} >1e-6=${f.cutMismatches1E6}`);
console.log(`maxAbsCutError=${f.maxAbsCutError.toExponential(9)} meanAbsCutError=${f.meanAbsCutError.toExponential(9)}`);
console.log(`maxSegmentsGpu=${f.maxSegmentsGpu} maxWorkSegmentsGpu=${f.maxWorkSegmentsGpu}`);
console.log(`B14E kernelMs=${e.kernelMs.toFixed(3)} annualProjectedMs=${e.annualProjectedKernelMs.toFixed(3)} checksum=${e.checksumHex}`);
console.log(`B14F kernelMs=${f.kernelMs.toFixed(3)} annualProjectedMs=${f.annualProjectedKernelMs.toFixed(3)} checksum=${f.checksumHex}`);
console.log(`kernelSpeedup=${speedup.toFixed(4)}x sameChecksum=${sameChecksum}`);
console.log(`B14F H2D=${f.inputH2DMs.toFixed(3)} D2H=${f.resultD2HMs.toFixed(3)} wall=${f.wallMs.toFixed(3)}`);
console.log(`B14F_RESULT=${exact&&sameChecksum?"EXACT_WARP_PRECHECK":"NEEDS_FIX"}`);
console.log(`SUNOPT_B14F_JSON=${JSON.stringify({baselineB14e:e,candidateB14f:f,kernelSpeedup:speedup,sameChecksum})}`);
