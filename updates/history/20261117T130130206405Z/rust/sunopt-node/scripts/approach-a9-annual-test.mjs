/*
 * A9 full, real SunOpt annual acceptance runner, NOT a replay benchmark.
 * Runs original FP64 A in one Next process, and then source-generated A9 in
 * another fresh Next process. Each run uses 28 tanks x 35,040 steps and the
 * actual controller's state feedback. Restores the narrow TS patch in finally.
 * Does not deploy or change the visible production A/B button.
 */
import {readFileSync,writeFileSync,mkdirSync,existsSync,createWriteStream} from 'node:fs';
import {spawn,spawnSync} from 'node:child_process';
import path from 'node:path';
import net from 'node:net';
import {performance} from 'node:perf_hooks';
import {setTimeout as delay} from 'node:timers/promises';
import {findSunOptProjectRoot} from './approach-a-generic-typescript.mjs';
const root=findSunOptProjectRoot(import.meta.url)??process.cwd();
const scripts=path.join(root,'rust/sunopt-node/scripts');
const outbox=path.join(root,'updates/outbox');mkdirSync(outbox,{recursive:true});
const stamp=new Date().toISOString().replace(/[-:.]/g,'');
const reportFile=path.join(outbox,`approach-a9-annual-${stamp}.json`);
const route=path.join(root,'src/app/api/tank-comparison/route.ts');
const build=JSON.parse(readFileSync(path.join(outbox,'approach-a9-build-cuda.json'),'utf8'));
const report={schemaVersion:1,createdAt:new Date().toISOString(),name:'A9 source-generated GPU true stateful annual acceptance',
 sourceHash:build.sourceHash,compiledSourceIsLive:false,statefulController:true,
 researchOnly:true,productionModifiedDuringTest:true,productionRestored:false,readyForProduction:false,
 originalA:null,genericA9:null,parity:null,performance:null,status:'INCOMPLETE',error:null,
 benchmarkProtocol:'Separate Next processes; same 2023 request; causalPath A; actual lockstep controller',
 note:'A9 selected only by explicit server environment, never the production UI A/B switch.'};
const body={heatPumpThermalPower:5,pvPeakPower:10,gridPriceChfPerKWh:0.3,feedInPriceChfPerKWh:0.08,causalPath:'A'};
const metrics=['gridImport','feedIn','selfConsumption','storageLosses','unmetThermalDemand','gridCost','feedInRevenue','netEnergyCost'];
const cmd=process.env.ComSpec||process.env.COMSPEC||'C:\\Windows\\System32\\cmd.exe';
const mode=process.argv.includes('--single')?1:2;
function runPatch(op){
 const command=spawnSync(process.execPath,[path.join(scripts,'approach-a9-bridge-patch.mjs'),op],{cwd:root,encoding:'utf8',timeout:10000});
 if(command.status!==0)throw Error(`A9 patch ${op} failure: ${command.stdout} ${command.stderr}`);
 console.log(command.stdout.trim());
}
async function portAvailable(port){return new Promise(resolve=>{
 const s=net.createServer();s.once('error',()=>resolve(false));s.listen(port,'127.0.0.1',()=>s.close(()=>resolve(true)));
});}
async function probe(url){try{
 const res=await fetch(url+'/tank-comparison',{signal:AbortSignal.timeout(15000)});return res.ok;
 }catch{return false;}}
async function openServer(variant){
 const port=3133;
 if(!await portAvailable(port))throw Error('Port 3133 is occupied; stop the previous test/server first.');
 const logFile=path.join(outbox,`approach-a9-server-${variant}-${stamp}.log`);
 const file=createWriteStream(logFile,{flags:'w'});
 const env={...process.env, SUNOPT_CUDA_FAST_MATH:'0',SUNOPT_APPROACH_A_STEP_GRAPH:'0'};
 delete env.SUNOPT_CUDA_CAUSAL_PATH;
 if(variant==='A9'){
  env.SUNOPT_APPROACH_A9_EXPERIMENT='1';
  env.SUNOPT_APPROACH_A9_BINARY=build.outputFile;
 }else{
  delete env.SUNOPT_APPROACH_A9_EXPERIMENT;
  delete env.SUNOPT_APPROACH_A9_BINARY;
 }
 const child=spawn(cmd,['/d','/s','/c',`npm run dev -- --hostname 127.0.0.1 --port ${port}`],
  {cwd:root,env,windowsHide:true,stdio:['ignore','pipe','pipe']});
 child.stdout.pipe(file);child.stderr.pipe(file);
 let failure=null;child.once('error',e=>{failure=e;});
 const url=`http://127.0.0.1:${port}`;
 for(let i=0;i<240;i++){
  if(failure||child.exitCode!==null)throw Error(`Next ${variant} startup failed: ${failure??child.exitCode}; see ${logFile}`);
  if(await probe(url)){await delay(900);return {child,url,file,logFile,variant};}
  await delay(500);
 }
 throw Error(`Next ${variant} startup timed out; see ${logFile}`);
}
async function stopServer(s){
 if(!s)return;
 if(s.child?.exitCode===null){
  if(process.platform==='win32')await new Promise(resolve=>{
   const killer=spawn(cmd,['/d','/s','/c',`taskkill /PID ${s.child.pid} /T /F`],{stdio:'ignore',windowsHide:true});
   killer.once('error',resolve);killer.once('close',resolve);
  });else {s.child.kill('SIGTERM');}
 }
 await new Promise(resolve=>s.file.end(resolve));
}
async function annualCall(url){
 const t0=performance.now();
 const response=await fetch(url+'/api/tank-comparison',{
  method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body),
  signal:AbortSignal.timeout(900000),cache:'no-store'});
 const wallMs=performance.now()-t0;
 const text=await response.text();
 if(!response.ok)throw Error(`HTTP ${response.status} ${text.slice(0,5000)}`);
 const result=JSON.parse(text);
 if(result.causalPath!=='A'||!Array.isArray(result.points)||result.points.length!==28||result.year!==2023 ||
    result.gpuLockstep?.annualSteps!==35040)throw Error('Unexpected year/tanks/steps or wrong controller path');
 for(const p of result.points){
  for(const key of metrics)if(!Number.isFinite(p[key]))throw Error(`Non-finite ${key} for ${p.storageVolume}L`);
 }
 return {wallMs,apiTotalMs:result.totalDurationMs,sweepMs:result.sweepDurationMs,
  steps:result.gpuLockstep.annualSteps,tanks:result.points.length,
  points:result.points,reportedGpu:result.gpuLockstep?.gpu??null};
}
const med=a=>{const s=[...a].sort((a,b)=>a-b);return s.length%2?s[(s.length-1)/2]:(s[s.length/2-1]+s[s.length/2])/2};
function compare(a,b){
 const results={};let pass=true;
 const ref=new Map(a.points.map(p=>[p.storageVolume,p]));
 for(const key of metrics){let maxAbs=0,maxRel=0,worstVolume=null;
  for(const p of b.points){
   const old=ref.get(p.storageVolume);if(!old)throw Error(`A9 unexpected volume ${p.storageVolume}`);
   const diff=Math.abs(old[key]-p[key]);
   const relative=diff/Math.max(1,Math.abs(old[key]),Math.abs(p[key]));
   if(diff>maxAbs){maxAbs=diff;worstVolume=p.storageVolume;}
   maxRel=Math.max(maxRel,relative);
   if(relative>1e-6)pass=false;
  }
  results[key]={maxAbs,maxRel,worstVolume};
 }
 return {status:pass?'PASS':'FAIL',relativeThreshold:1e-6,metrics:results};
}
async function measure(variant){
 let server=null;
 try{
  server=await openServer(variant);
  const rows=[];
  console.log(`A9_ANNUAL ${variant} warmup, 28 tanks x 35040`);
  await annualCall(server.url);
  for(let i=0;i<mode;i++){
   console.log(`A9_ANNUAL ${variant} measured ${i+1}/${mode}`);
   const r=await annualCall(server.url);rows.push(r);
   console.log(`A9_ANNUAL_RESULT ${variant}: sweep ${r.sweepMs.toFixed(1)}ms, API ${r.apiTotalMs.toFixed(1)}ms, wall ${r.wallMs.toFixed(1)}ms`);
  }
  return {medians:{wallMs:med(rows.map(r=>r.wallMs)),apiTotalMs:med(rows.map(r=>r.apiTotalMs)),sweepMs:med(rows.map(r=>r.sweepMs))},
    runs:rows.map(r=>({wallMs:r.wallMs,apiTotalMs:r.apiTotalMs,sweepMs:r.sweepMs})),
    points:rows[0].points,steps:35040,tanks:28,logFile:server.logFile,
    // Native worker-only GPU profile is not meaningful for A9 sidecar; use measured API/sweep.
    profileUsableForA9:false};
 }finally{await stopServer(server);}
}
let applied=false;
try{
 if(!build.compiled||!existsSync(build.outputFile)||build.backend!=='cuda')throw Error('No source-generated CUDA binary built');
 if(!existsSync(route)||!readFileSync(route,'utf8').includes('runTankComparisonGpuLockstep'))throw Error('Unexpected annual API source');
 runPatch('--check');runPatch('--apply');applied=true;
 report.originalA=await measure('OriginalA');
 report.genericA9=await measure('A9');
 const log=readFileSync(report.genericA9.logFile,'utf8');
 const required=['A9 SOURCE-GENERATED CUDA WORKER SELECTED','A9 SOURCE-GENERATED CUDA EXECUTION',
  `A9_INIT_OK source=${build.sourceHash}`];
 const missing=required.filter(phrase=>!log.includes(phrase));
 report.compiledSourceIsLive=missing.length===0;
 if(missing.length)throw Error(`A9 provenance failure; required log markers missing: ${missing.join(', ')}`);
 report.parity=compare(report.originalA,report.genericA9);
 report.performance={originalMedianSweepMs:report.originalA.medians.sweepMs,
  genericA9MedianSweepMs:report.genericA9.medians.sweepMs,
  speedRatioOriginalOverA9:report.originalA.medians.sweepMs/report.genericA9.medians.sweepMs,
  comparableAsAnnual:true,serverProcessesIsolated:true};
 report.status=report.parity.status==='PASS'?'PASS_RESEARCH_ONLY':'FAIL_PARITY';
}catch(e){report.status='FAIL';report.error=String(e?.stack??e).slice(0,12000);console.error('A9_ANNUAL_FAILED',report.error);}
finally{
 if(applied){try{runPatch('--restore');report.productionRestored=true;}
  catch(e){report.productionRestored=false;report.restoreError=String(e);console.error('A9_RESTORE_FAILED',String(e));}}
 report.finishedAt=new Date().toISOString();
 writeFileSync(reportFile,JSON.stringify(report,null,2)+'\n');
 console.log('A9_ANNUAL_REPORT',reportFile);
 console.log('A9_ANNUAL_STATUS',report.status,'provenance',report.compiledSourceIsLive,'restored',report.productionRestored);
 if(report.status!=='PASS_RESEARCH_ONLY'||!report.productionRestored)process.exitCode=1;
}
