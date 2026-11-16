// rust/sunopt-node/scripts/blackwell-b14c-warp32.mjs
// B14C research-only CUDA exact-envelope correctness/performance probe.
import {createRequire} from "node:module";
import path from "node:path";
import {fileURLToPath} from "node:url";

const require=createRequire(import.meta.url);
const here=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(here,"..");
const addon=require(path.join(root,"sunopt_node.win32-x64-msvc.node"));

if(typeof addon.runBlackwellB14cWarp32!=="function"){
  throw new Error("runBlackwellB14cWarp32 missing; rebuild sunopt-node first.");
}

const snapshot=process.argv[2]?path.resolve(process.argv[2]):path.join(root,"blackwell-b12-sync-blackwell.bin");
const sampleSteps=Number.parseInt(process.argv[3]??"2048",10);
const repetitions=Number.parseInt(process.argv[4]??"1",10);
const deviceId=Number.parseInt(process.argv[5]??"0",10);

console.log("--- B14C CUDA WARP32 EXACT ENVELOPE ---");
console.log(`snapshot=${snapshot}`);
console.log(`sampleSteps=${sampleSteps} repetitions=${repetitions} deviceId=${deviceId}`);

const r=addon.runBlackwellB14cWarp32(snapshot,sampleSteps,repetitions,deviceId);
console.log(`jobs=${r.sampleJobCount} supported=${r.supportedJobs} unsupported=${r.unsupportedJobs}`);
console.log(`overflow32=${r.overflow32Jobs} scratchOverflow=${r.scratchOverflowJobs}`);
console.log(`statusMismatches=${r.statusMismatches}`);
console.log(`cutMismatch >1e-12=${r.cutMismatches1E12} >1e-9=${r.cutMismatches1E9} >1e-6=${r.cutMismatches1E6}`);
console.log(`maxAbsCutError=${r.maxAbsCutError.toExponential(9)} meanAbsCutError=${r.meanAbsCutError.toExponential(9)}`);
console.log(`maxSegmentsGpu=${r.maxSegmentsGpu} maxWorkSegmentsGpu=${r.maxWorkSegmentsGpu}`);
console.log(`referenceScalarMs=${r.referenceScalarMs.toFixed(3)}`);
console.log(`H2D=${r.inputH2DMs.toFixed(3)} kernel=${r.kernelMs.toFixed(3)} D2H=${r.resultD2HMs.toFixed(3)} wall=${r.wallMs.toFixed(3)}`);
console.log(`annualProjectedKernelMs=${r.annualProjectedKernelMs.toFixed(3)}`);
console.log(`checksum=${r.checksumHex}`);

const exact=r.statusMismatches===0&&r.cutMismatches1E12===0&&r.overflow32Jobs===0&&r.scratchOverflowJobs===0;
console.log(`B14C_RESULT=${exact?"EXACT_WARP32_CANDIDATE":"NEEDS_FIX_OR_WIDER_ENVELOPE"}`);
console.log(`SUNOPT_B14C_JSON=${JSON.stringify(r)}`);
