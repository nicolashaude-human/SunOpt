// rust/sunopt-node/scripts/blackwell-b14d-sorted-merge.mjs
// B14C vs B14D exact-envelope A/B: boundary-sort/search vs O(n) sorted PWA merge.
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

const runB14c=pickExport([
  "runBlackwellB14CWarp32",
  "runBlackwellB14cWarp32",
]);
const runB14d=pickExport([
  "runBlackwellB14DWarp32Merge",
  "runBlackwellB14dWarp32Merge",
]);

if(runB14c===null||runB14d===null){
  const exports=Object.keys(addon).filter((key)=>/b14|warp32|merge/i.test(key)).sort();
  throw new Error(`B14C/B14D native export missing. Available: ${JSON.stringify(exports)}`);
}

const snapshot=process.argv[2]?path.resolve(process.argv[2]):path.join(root,"blackwell-b12-sync-blackwell.bin");
const sampleSteps=Number.parseInt(process.argv[3]??"2048",10);
const repetitions=Number.parseInt(process.argv[4]??"1",10);
const deviceId=Number.parseInt(process.argv[5]??"0",10);

console.log("--- B14D CUDA SORTED-MERGE A/B ---");
console.log(`snapshot=${snapshot}`);
console.log(`sampleSteps=${sampleSteps} repetitions=${repetitions} deviceId=${deviceId}`);

const c=runB14c(snapshot,sampleSteps,repetitions,deviceId);
const d=runB14d(snapshot,sampleSteps,repetitions,deviceId);
const speedup=d.kernelMs>0?c.kernelMs/d.kernelMs:0;
const sameChecksum=c.checksumHex===d.checksumHex;
const exact=d.statusMismatches===0&&d.cutMismatches1E12===0&&d.overflow32Jobs===0&&d.scratchOverflowJobs===0;

console.log(`jobs=${d.sampleJobCount} supported=${d.supportedJobs} unsupported=${d.unsupportedJobs}`);
console.log(`overflow32=${d.overflow32Jobs} scratchOverflow=${d.scratchOverflowJobs}`);
console.log(`statusMismatches=${d.statusMismatches}`);
console.log(`cutMismatch >1e-12=${d.cutMismatches1E12} >1e-9=${d.cutMismatches1E9} >1e-6=${d.cutMismatches1E6}`);
console.log(`maxAbsCutError=${d.maxAbsCutError.toExponential(9)} meanAbsCutError=${d.meanAbsCutError.toExponential(9)}`);
console.log(`maxSegmentsGpu=${d.maxSegmentsGpu} maxWorkSegmentsGpu=${d.maxWorkSegmentsGpu}`);
console.log(`B14C kernelMs=${c.kernelMs.toFixed(3)} annualProjectedMs=${c.annualProjectedKernelMs.toFixed(3)} checksum=${c.checksumHex}`);
console.log(`B14D kernelMs=${d.kernelMs.toFixed(3)} annualProjectedMs=${d.annualProjectedKernelMs.toFixed(3)} checksum=${d.checksumHex}`);
console.log(`kernelSpeedup=${speedup.toFixed(4)}x sameChecksum=${sameChecksum}`);
console.log(`B14D H2D=${d.inputH2DMs.toFixed(3)} D2H=${d.resultD2HMs.toFixed(3)} wall=${d.wallMs.toFixed(3)}`);
console.log(`B14D_RESULT=${exact&&sameChecksum?"EXACT_SORTED_MERGE":"NEEDS_FIX"}`);
console.log(`SUNOPT_B14D_JSON=${JSON.stringify({baselineB14c:c,candidateB14d:d,kernelSpeedup:speedup,sameChecksum})}`);
