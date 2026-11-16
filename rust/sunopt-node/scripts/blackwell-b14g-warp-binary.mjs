// rust/sunopt-node/scripts/blackwell-b14g-warp-binary.mjs
// B14F exact PWA vs B14G Production-equivalent 20-step warp binary search.
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

const runB14f=pickExport([
  "runBlackwellB14FWarpPrecheck",
  "runBlackwellB14fWarpPrecheck",
]);
const runB14g=pickExport([
  "runBlackwellB14GWarpBinary",
  "runBlackwellB14gWarpBinary",
]);

if(runB14f===null||runB14g===null){
  const exports=Object.keys(addon).filter((key)=>/b14|warp|binary|precheck/i.test(key)).sort();
  throw new Error(`B14F/B14G native export missing. Available: ${JSON.stringify(exports)}`);
}

const snapshot=process.argv[2]?path.resolve(process.argv[2]):path.join(root,"blackwell-b12-sync-blackwell.bin");
const sampleSteps=Number.parseInt(process.argv[3]??"2048",10);
const repetitions=Number.parseInt(process.argv[4]??"1",10);
const deviceId=Number.parseInt(process.argv[5]??"0",10);

console.log("--- B14G CUDA WARP-BINARY A/B ---");
console.log(`snapshot=${snapshot}`);
console.log(`sampleSteps=${sampleSteps} repetitions=${repetitions} deviceId=${deviceId}`);

const f=runB14f(snapshot,sampleSteps,repetitions,deviceId);
const g=runB14g(snapshot,sampleSteps,repetitions,deviceId);
const speedup=g.kernelMs>0?f.kernelMs/g.kernelMs:0;
const sameChecksum=f.checksumHex===g.checksumHex;
const exact=g.statusMismatches===0&&g.cutMismatches1E12===0&&g.overflow32Jobs===0&&g.scratchOverflowJobs===0;

console.log(`jobs=${g.sampleJobCount} supported=${g.supportedJobs} unsupported=${g.unsupportedJobs}`);
console.log(`statusMismatches=${g.statusMismatches}`);
console.log(`cutMismatch >1e-12=${g.cutMismatches1E12} >1e-9=${g.cutMismatches1E9} >1e-6=${g.cutMismatches1E6}`);
console.log(`maxAbsCutError=${g.maxAbsCutError.toExponential(9)} meanAbsCutError=${g.meanAbsCutError.toExponential(9)}`);
console.log(`B14F kernelMs=${f.kernelMs.toFixed(3)} annualProjectedMs=${f.annualProjectedKernelMs.toFixed(3)} checksum=${f.checksumHex}`);
console.log(`B14G kernelMs=${g.kernelMs.toFixed(3)} annualProjectedMs=${g.annualProjectedKernelMs.toFixed(3)} checksum=${g.checksumHex}`);
console.log(`kernelSpeedup=${speedup.toFixed(4)}x sameChecksum=${sameChecksum}`);
console.log(`B14G H2D=${g.inputH2DMs.toFixed(3)} D2H=${g.resultD2HMs.toFixed(3)} wall=${g.wallMs.toFixed(3)}`);
console.log(`B14G_RESULT=${exact&&sameChecksum?"EXACT_WARP_BINARY":"NEEDS_FIX"}`);
console.log(`SUNOPT_B14G_JSON=${JSON.stringify({baselineB14f:f,candidateB14g:g,kernelSpeedup:speedup,sameChecksum})}`);
