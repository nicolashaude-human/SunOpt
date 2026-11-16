// rust/sunopt-node/scripts/blackwell-b14e-warp-parallel.mjs
// B14D vs B14E exact-envelope A/B: serial ordered algebra vs warp-parallel PWA operations.
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

const runB14d=pickExport([
  "runBlackwellB14DWarp32Merge",
  "runBlackwellB14dWarp32Merge",
]);
const runB14e=pickExport([
  "runBlackwellB14EWarpParallel",
  "runBlackwellB14eWarpParallel",
]);

if(runB14d===null||runB14e===null){
  const exports=Object.keys(addon).filter((key)=>/b14|warp|parallel|merge/i.test(key)).sort();
  throw new Error(`B14D/B14E native export missing. Available: ${JSON.stringify(exports)}`);
}

const snapshot=process.argv[2]?path.resolve(process.argv[2]):path.join(root,"blackwell-b12-sync-blackwell.bin");
const sampleSteps=Number.parseInt(process.argv[3]??"2048",10);
const repetitions=Number.parseInt(process.argv[4]??"1",10);
const deviceId=Number.parseInt(process.argv[5]??"0",10);

console.log("--- B14E CUDA WARP-PARALLEL A/B ---");
console.log(`snapshot=${snapshot}`);
console.log(`sampleSteps=${sampleSteps} repetitions=${repetitions} deviceId=${deviceId}`);

const d=runB14d(snapshot,sampleSteps,repetitions,deviceId);
const e=runB14e(snapshot,sampleSteps,repetitions,deviceId);
const speedup=e.kernelMs>0?d.kernelMs/e.kernelMs:0;
const sameChecksum=d.checksumHex===e.checksumHex;
const exact=e.statusMismatches===0&&e.cutMismatches1E12===0&&e.overflow32Jobs===0&&e.scratchOverflowJobs===0;

console.log(`jobs=${e.sampleJobCount} supported=${e.supportedJobs} unsupported=${e.unsupportedJobs}`);
console.log(`overflow32=${e.overflow32Jobs} scratchOverflow=${e.scratchOverflowJobs}`);
console.log(`statusMismatches=${e.statusMismatches}`);
console.log(`cutMismatch >1e-12=${e.cutMismatches1E12} >1e-9=${e.cutMismatches1E9} >1e-6=${e.cutMismatches1E6}`);
console.log(`maxAbsCutError=${e.maxAbsCutError.toExponential(9)} meanAbsCutError=${e.meanAbsCutError.toExponential(9)}`);
console.log(`maxSegmentsGpu=${e.maxSegmentsGpu} maxWorkSegmentsGpu=${e.maxWorkSegmentsGpu}`);
console.log(`B14D kernelMs=${d.kernelMs.toFixed(3)} annualProjectedMs=${d.annualProjectedKernelMs.toFixed(3)} checksum=${d.checksumHex}`);
console.log(`B14E kernelMs=${e.kernelMs.toFixed(3)} annualProjectedMs=${e.annualProjectedKernelMs.toFixed(3)} checksum=${e.checksumHex}`);
console.log(`kernelSpeedup=${speedup.toFixed(4)}x sameChecksum=${sameChecksum}`);
console.log(`B14E H2D=${e.inputH2DMs.toFixed(3)} D2H=${e.resultD2HMs.toFixed(3)} wall=${e.wallMs.toFixed(3)}`);
console.log(`B14E_RESULT=${exact&&sameChecksum?"EXACT_WARP_PARALLEL":"NEEDS_FIX"}`);
console.log(`SUNOPT_B14E_JSON=${JSON.stringify({baselineB14d:d,candidateB14e:e,kernelSpeedup:speedup,sameChecksum})}`);
