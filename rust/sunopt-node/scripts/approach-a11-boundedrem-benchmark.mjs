/*
 * SunOpt A11 research gate: one unknown-source generic AST compiler produces
 * two CUDA variants. Change is restricted to mathematically equivalent JS
 * numeric remainder lowering, never to a SunOpt equation or scope ID.
 * These snapshots are independent; the annual stateful acceptance follows.
 */
import {Worker} from 'node:worker_threads';
import {spawnSync} from 'node:child_process';
import {readFileSync, writeFileSync, mkdirSync, existsSync} from 'node:fs';
import {performance} from 'node:perf_hooks';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {findSunOptProjectRoot} from './approach-a-generic-typescript.mjs';
const root=findSunOptProjectRoot(import.meta.url)??process.cwd();
const scripts=path.join(root,'rust/sunopt-node/scripts');
const outbox=path.join(root,'updates/outbox');mkdirSync(outbox,{recursive:true});
const localCpu=process.argv.includes('--cpu-local');
const extension=process.platform==='win32'?'.exe':'.bin';
const sourceInput=readFileSync(path.join(root,'src/simulation/rust-acceleration-scopes.ts'),'utf8')+'\n\n'+readFileSync(path.join(root,'src/simulation/approach-a9-energy-scopes.ts'),'utf8');
const sourceHash=createHash('sha256').update(sourceInput).digest('hex');
const arch=process.argv.find(s=>s.startsWith('--arch='))?.slice(7)??'sm_120';
const stamp=new Date().toISOString().replace(/[-:.]/g,'');
const reportFile=path.join(outbox,`approach-a11-boundedrem-${stamp}.json`);
const report={sourceHash,stage:'A11 generic bounded-range remainder lowering (no physics knowledge)',
 createdAt:new Date().toISOString(),benchmarkIsAnnual:false,warmupStepsPerVersion:48,stepsMeasuredPerVersion:256,
 researchOnly:true,readyForProduction:false,pass:false,comparison:null,
 versions:[],parity:null,speedup:null,error:null};
const max=128, header=64, confOffset=header, dynOffset=confOffset+12*max*8,
 outOffset=dynOffset+3*max*8, errOffset=outOffset+max*8;
const length=35040,tanks=28,steps=256,warmupSteps=48,horizon=288;
const exportPower=Float64Array.from({length},(_,i)=>Math.max(0,6.2*Math.sin(i*.053)+3.2));
const demandEnergy=Float64Array.from({length},(_,i)=>i%17===0?1.5:.08);
function getBuild(extra){
 if(localCpu){
  const name=`approach-a9-generated-${sourceHash.slice(0,12)}${extra?'-a11-boundedrem':''}${extension}`;
  return {compiled:true,outputFile:path.join(outbox,name),compilerOptimization:extra?'generic-bounded-remainder-fastpath':'baseline',sourceHash};
 }
 const args=[path.join(scripts,'approach-a9-build.mjs'),`--arch=${arch}`,'--profile',...(extra?['--fast-bounded-remainder']:[])];
 const run=spawnSync(process.execPath,args,{cwd:root,encoding:'utf8',timeout:180000});
 if(run.status!==0)throw Error(`A11 build failed: ${args.join(' ')}\n${run.stdout}\n${run.stderr}`);
 const build=JSON.parse(readFileSync(path.join(outbox,'approach-a9-build-cuda.json'),'utf8'));
 if(!build.compiled||!existsSync(build.outputFile))throw Error('A11 nvcc failed to create binary');
 if(build.compilerOptimization !== (extra?'generic-bounded-remainder-fastpath':'baseline'))throw Error('A11 compiler mode not recorded');
 return build;
}
function makeBuffer(){
 const shared=new SharedArrayBuffer(errOffset+4096),status=new Int32Array(shared,0,16),
  config=new Float64Array(shared,confOffset,12*max),dyn=new Float64Array(shared,dynOffset,3*max),
  out=new Float64Array(shared,outOffset,max),errors=new Uint8Array(shared,errOffset,4096);
 for(let k=0;k<tanks;k++){
  const b=12*k; config[b+2]=20+k;config[b+3]=1.8+.01*k;config[b+4]=3.25;
  config[b+6]=.25;config[b+7]=1.2+.02*k;config[b+8]=2.1+.03*k;
  config[b+9]=60;config[b+10]=17;config[b+11]=k%3===0?-1:.9;
 }
 return {shared,status,dyn,out,errors};
}
async function execute(build){
 const {shared,status,dyn,out,errors}=makeBuffer();
 const worker=new Worker(path.join(scripts,'approach-a9-ipc-worker.mjs'),
  {workerData:{shared,binary:build.outputFile,exportPower,demandEnergy}});
 const result=new Float64Array(steps*tanks);
 function call(op,more){
  Atomics.store(status,0,0);worker.postMessage({op,...more});
  const state=Atomics.wait(status,0,0,120000);
  if(state==='timed-out'||Atomics.load(status,0)!==1){
   const until=errors.indexOf(0);
   throw Error(`A11 ${op} ${state}: ${new TextDecoder().decode(errors.subarray(0,until<0?errors.length:until))}`);
  }
 }
 try{
  call('configure',{tanks});
  // Equal warmup for both variants, excluded from the timed 256 snapshots.
  for(let i=-warmupSteps;i<steps;i++){ 
   for(let k=0;k<tanks;k++){
    const c=20+k, b=k*3,energy=.45*c+((i+12*1000)%12)*.02*c;
    dyn[b]=energy;dyn[b+1]=k%4===1?.9*c:.5*c;
    dyn[b+2]=Math.max(0,Math.min(4,c-energy)/3.25);
   }
   call('step',{index:((i+length)%length),horizon});
   if(i===-1)var start=performance.now();
   if(i>=0)result.set(out.subarray(0,tanks),i*tanks);
  }
  return {wallMs:performance.now()-start,results:result};
 }finally{await worker.terminate();}
}
try{
 const builds=new Map();
 for(const extra of [false,true]) {
  const build=getBuild(extra); builds.set(extra,build);
  report.versions.push({compilerOptimization:build.compilerOptimization,sourceHash:build.sourceHash,
   binary:build.outputFile,profileMode:build.profileMode??false});
 }
 const runs={baseline:[],optimized:[]};
 let reference=null;
 const sameOutputs=(a,b)=>{
  let mismatches=0,maxAbs=0;
  for(let i=0;i<a.length;i++){
   const d=Math.abs(a[i]-b[i]);maxAbs=Math.max(maxAbs,d);
   if(!(d<=1e-10*Math.max(1,Math.abs(a[i]),Math.abs(b[i]))))mismatches++;
  }
  return {mismatches,maxAbs};
 };
 for(const extra of [false,true,true,false]) {
  const res=await execute(builds.get(extra));
  const tag=extra?'optimized':'baseline';
  runs[tag].push(res.wallMs);
  if(reference===null)reference=res.results;
  else {
   const parity=sameOutputs(reference,res.results);
   if(!report.parity)report.parity={comparisons:0,mismatches:0,maxAbs:0};
   report.parity.comparisons+=res.results.length;
   report.parity.mismatches+=parity.mismatches;
   report.parity.maxAbs=Math.max(report.parity.maxAbs,parity.maxAbs);
  }
 }
 if(!report.parity)throw Error('No independent parity comparisons');
 report.parity.status=report.parity.mismatches?'FAIL':'PASS';
 const median=x=>{const sorted=[...x].sort((a,b)=>a-b);const mid=Math.floor(sorted.length/2);return sorted.length%2?sorted[mid]:(sorted[mid-1]+sorted[mid])/2;};
 report.baseline={wallMs:median(runs.baseline),runsMs:runs.baseline};
 report.optimized={wallMs:median(runs.optimized),runsMs:runs.optimized};
 report.speedup=report.baseline.wallMs/report.optimized.wallMs;
 report.pass=report.parity.status==='PASS'&&report.speedup>1.10;
 report.status=report.pass?'PASS_FASTPATH':'BLOCK_NOT_FASTER_OR_PARITY';
 // The annual runner reads the last build report; guarantee that it selects
 // this proven source-generated A11 binary, never the baseline from ABBA.
 if(report.pass&&!localCpu) {
  const build=builds.get(true);
  writeFileSync(path.join(outbox,'approach-a9-build-cuda.json'),JSON.stringify(build,null,2)+'\n');
 }
}catch(e){report.status='FAIL';report.error=String(e?.stack??e).slice(0,16000);}
report.finishedAt=new Date().toISOString();
writeFileSync(reportFile,JSON.stringify(report,null,2)+'\n');
console.log('A11_REPORT',reportFile,'A11_STATUS',report.status,
 'speedup',report.speedup?.toFixed(3),'mismatches',report.parity?.mismatches);
if(!report.pass)process.exitCode=1;
