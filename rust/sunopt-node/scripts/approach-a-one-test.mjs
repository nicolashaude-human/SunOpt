/*
 * SunOpt Approach A — ONE deterministic acceptance command.
 *
 * Strict separation:
 *   1. Generic source compiler: C++/CUDA parity and native A1/A2 per-call benchmark.
 *   2. CURRENT production 28-tank year, causalPath=A and B, exact same inputs.
 *   3. Mandatory gate: generic 28-tank annual only passes when generic executor
 *      is actually wired. The existing A=FP64 CUDA route is NOT generic A.
 *
 * This script never edits a production source file or selects a solver by omission.
 */
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync, createWriteStream } from 'node:fs';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import net from 'node:net';

const here=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(here,'..','..','..');
const outbox=path.join(root,'updates','outbox');
mkdirSync(outbox,{recursive:true});
const suffix=new Date().toISOString().replace(/[-:.]/g,'');
const reportFile=path.join(outbox,`approach-a-one-test-${suffix}.json`);
const testLog=path.join(outbox,`approach-a-one-test-${suffix}.log`);
const serverLog=path.join(outbox,`approach-a-one-test-server-${suffix}.log`);
const body={heatPumpThermalPower:5,pvPeakPower:10,gridPriceChfPerKWh:0.3,feedInPriceChfPerKWh:0.08};
const metrics=['gridImport','feedIn','selfConsumption','storageLosses','unmetThermalDemand','gridCost','feedInRevenue','netEnergyCost'];
const report={schemaVersion:1,name:'SunOpt A single-command verification (generic CUDA, original A1+A2, full annual baseline)',createdAt:new Date().toISOString(),root,
 input:{year:2023,volumeMinL:300,volumeMaxL:3000,volumeStepL:100,tankCount:28,stepsExpected:35040,...body},
 stages:[],sourceRoute:null,genericCuda:null,originalA1A2:null,primitiveComparison:[],yearlyProduction:null,
 gpuDispatch:null, generic28TankYear:{status:'BLOCK',reason:'Generic compiler is not wired to the full production annual controller. A GPU reference annual run does not prove generic annual integration.'},
 readyForProduction:false,finishedAt:null};
let logger=createWriteStream(testLog,{flags:'w'});
const log=(msg)=>{const line=`[${new Date().toISOString()}] ${msg}`;console.log(line);logger.write(line+'\n');};
const stage=(name,status,info={})=>{report.stages.push({name,status,...info});log(`${status}: ${name}`);};
const save=()=>{report.finishedAt=new Date().toISOString();writeFileSync(reportFile,JSON.stringify(report,null,2)+'\n');};
const cmdExe=process.env.ComSpec||process.env.COMSPEC||'C:\\Windows\\System32\\cmd.exe';
function runNode(args,{cwd=root,timeoutMs=720000}={}){
 return new Promise(resolve=>{
  const child=spawn(process.execPath,args,{cwd,windowsHide:true,env:process.env,stdio:['ignore','pipe','pipe']});
  let stdout='',stderr='',last='';let expired=false;
  const MAX=120000;
  const collect=(which,chunk)=>{const str=chunk.toString(); logger.write(str); if(which==='out')stdout=(stdout+str).slice(-MAX);else stderr=(stderr+str).slice(-MAX);last=(last+str).slice(-MAX);};
  child.stdout.on('data',d=>collect('out',d));child.stderr.on('data',d=>collect('err',d));
  const handle=setTimeout(()=>{expired=true;try{child.kill('SIGTERM')}catch{}},timeoutMs);
  child.once('error',e=>{clearTimeout(handle);resolve({status:'ERROR',message:String(e),stdout,stderr,expired});});
  child.once('close',(code,signal)=>{clearTimeout(handle);resolve({status:code===0&&!expired?'PASS':'FAIL',exitCode:code,signal,stdout,stderr,expired,tail:last.slice(-8000)});});
 });
}
function newestStructured(since){
 return readdirSync(outbox).filter(x=>/^approach-a-structured-verify-.*\.json$/.test(x)).flatMap(x=>{try{return [{name:x,mtime:Date.parse(JSON.parse(readFileSync(path.join(outbox,x),'utf8')).at??0)}]}catch{return []}}).filter(x=>Number.isFinite(x.mtime)&&x.mtime>=since-5000).sort((a,b)=>b.mtime-a.mtime)[0]?.name??null;
}
async function runStructured(){
 log('1/4 Generischer CUDA- und A1+A2-Test (gleiche Testfälle, 2400 Aufrufe pro Funktion)');
 const start=Date.now();
 const result=await runNode([path.join(here,'approach-a-structured-verify.mjs'),'--require-gpu','--require-original-a','--arch=sm_120']);
 const name=newestStructured(start);
 if(!name){stage('Generic compiler and original A1+A2 microbenchmark','FAIL',{error:'No fresh structured report',runner:result});return;}
 const reference=JSON.parse(readFileSync(path.join(outbox,name),'utf8'));
 const gpu=reference.stages.find(x=>x.name.startsWith('G7f:'));
 const native=reference.stages.find(x=>x.name==='Original A1+A2 native benchmark');
 report.structuredReport=path.join(outbox,name);
 report.genericCuda=gpu??{status:'BLOCK',reason:'G7f missing'};
 report.originalA1A2=native??{status:'BLOCK',reason:'Original A1+A2 benchmark missing'};
 report.genericCpuParity=reference.stages.find(x=>x.name.includes('G7e:'))??null;
 report.primitiveComparison=[];
 for(const entry of ['runRustNoSunshineScope','runRustPeakShavingCausalScope']){
  const original=native?.comparisons?.find(x=>x.entry===entry);
  const line=reference.gpuBenchmarks?.find(x=>x.startsWith(`B ${entry} `));
  const gpuWall=line?Number(line.trim().split(/\s+/)[2]):null;
  report.primitiveComparison.push({entry,calls:2400,originalA1A2NapiMs:original?.wallMs??null,genericCudaDeviceResidentMs:Number.isFinite(gpuWall)?gpuWall:null,
    indicativeSpeedRatio:Number.isFinite(gpuWall)&&gpuWall>0&&Number.isFinite(original?.wallMs)?original.wallMs/gpuWall:null,
    comparableAsEndToEnd:false,note:'Same 2400 dynamic numeric inputs; GPU uses 200 batches of 12 with resident arrays, excludes GPU transfers and process/API boundaries. Original A2 uses 2400 N-API calls. Do not present this as annual speedup.'});
 }
 const allNeeded=['G7a:','G7b:','G7c:','G7d:','G7e:','G7f:'];
 const ok=allNeeded.every(prefix=>reference.stages.some(s=>s.name.startsWith(prefix)&&s.status==='PASS'))&&native?.status==='PASS'&&result.status==='PASS';
 stage('Generic CUDA + original native A1+A2',ok?'PASS':'FAIL',{gpuStatus:gpu?.status??'MISSING',originalStatus:native?.status??'MISSING',structuredReport:report.structuredReport,runnerExitCode:result.exitCode,diagnostics:!ok?{gpu:gpu?.diagnostics??gpu?.error??null,stderrTail:result.stderr.slice(-3000)}:undefined});
}
// A8: compares *identical* independent inputs across 200 small launches,
// a single multi-SM launch and the old native A1/A2 path. These are neither
// chronological annual inputs nor production API invocations.
async function runDispatch(){
 log('2/4 Generischer GPU-Dispatcher: 2400 identische Eingaben, 200x12 vs. Grid 64/128/256');
 const start=Date.now();
 const result=await runNode([path.join(here,'approach-a-generic-dispatch-bench.mjs'),'--require-gpu','--require-original-a','--arch=sm_120']);
 const matches=readdirSync(outbox).filter(x=>/^approach-a-generic-dispatch-.*\.json$/.test(x)).flatMap(x=>{
  try{const value=JSON.parse(readFileSync(path.join(outbox,x),'utf8'));return Date.parse(value.createdAt)>=start-5000?[{name:x,value}]:[]}catch{return []}
 });
 matches.sort((a,b)=>b.value.createdAt.localeCompare(a.value.createdAt));
 if(!matches.length){stage('Generic A8 GPU dispatcher and original A1+A2 comparison','FAIL',{reason:'No fresh dispatcher report',runner:result});return;}
 const item=matches[0];
 report.gpuDispatch={reportFile:path.join(outbox,item.name),...item.value};
 const ok=result.status==='PASS'&&item.value.stage==='BENCHMARK_PASS_NO_ANNUAL'&&item.value.gpu?.status==='PASS'&&item.value.originalA1A2?.status==='PASS';
 stage('Generic A8 GPU dispatcher and original A1+A2 comparison',ok?'PASS':'FAIL',{
  gpuStatus:item.value.gpu?.status,originalA1A2Status:item.value.originalA1A2?.status,
  gpuGridResults:item.value.comparisons,runnerExitCode:result.exitCode,
  note:'Independent-snapshot dispatch ONLY; full annual GPU controller not integrated',
  diagnostic:ok?undefined:result.tail.slice(-3500)});
}
function ensureProductionRoute(){
 const route=path.join(root,'src','app','api','tank-comparison','route.ts');
 if(!existsSync(route))throw Error(`Missing API route ${route}`);
 const text=readFileSync(route,'utf8');
 const usesCurrentGpuLockstep=text.includes('runTankComparisonGpuLockstep')&&text.includes('causalPath');
 report.sourceRoute={path:route,usesCurrentGpuLockstep,hasGenericCompilerReference:/approach-a-structured-compiler|approach-a-generic-compiler/.test(text)};
 if(!usesCurrentGpuLockstep)throw Error('Current annual route no longer matches previously inspected GPU lockstep; will not assume its solver classification.');
 // Explicitly distinguish the existing FP64 CUDA A path from GENERIC source A.
 report.generic28TankYear={status:'BLOCK',reason:'Current production endpoint invokes tank-comparison-gpu-lockstep, not the new source-derived generic CUDA executor. Annual A reference cannot be relabeled generic A.',proof:route};
}
async function probeUrl(url){
 try{const r=await fetch(`${url}/tank-comparison`,{signal:AbortSignal.timeout(5000)});const t=await r.text();return r.ok&&/SunOpt/i.test(t);}catch{return false;}
}
function portCheck(port){return new Promise(resolve=>{const s=net.createConnection({host:'127.0.0.1',port});const finish=x=>{s.destroy();resolve(x)};s.setTimeout(1000);s.once('connect',()=>finish(true));s.once('error',()=>finish(false));s.once('timeout',()=>finish(false));});}
async function openServer(){
 const specified=process.env.SUNOPT_A_TEST_BASE_URL;
 if(specified){const url=specified.replace(/\/$/,'');if(!/^https?:\/\/127\.0\.0\.1:\d+$/.test(url)&&!/^https?:\/\/localhost:\d+$/.test(url))throw Error('SUNOPT_A_TEST_BASE_URL must be localhost');if(!await probeUrl(url))throw Error(`Selected Next server is not responding: ${url}`);return {url,owned:false};}
 for(const p of [3000,3001]){const url=`http://127.0.0.1:${p}`;if(await probeUrl(url)){log(`Using existing SunOpt server ${url} (server environment cannot be controlled)`);return {url,owned:false};}}
 const port=3129;
 if(await portCheck(port))throw Error(`Port ${port} occupied by an unrelated service; set SUNOPT_A_TEST_BASE_URL explicitly`);
 const logFile=createWriteStream(serverLog,{flags:'w'});
 const env={...process.env,SUNOPT_APPROACH_A_STEP_GRAPH:'0',SUNOPT_CUDA_FAST_MATH:'0'};
 delete env.SUNOPT_CUDA_CAUSAL_PATH;
 const child=spawn(cmdExe,['/d','/s','/c',`npm run dev -- --hostname 127.0.0.1 --port ${port}`],{cwd:root,env,windowsHide:true,stdio:['ignore','pipe','pipe']});
 child.stdout?.pipe(logFile);child.stderr?.pipe(logFile);
 let error=null;child.once('error',e=>{error=e});
 const url=`http://127.0.0.1:${port}`;
 for(let i=0;i<120;i++){
  if(error||child.exitCode!==null){await stopServer({child,logFile});throw Error(`Next dev server failed: ${error??child.exitCode}; read ${serverLog}`);}
  if(await probeUrl(url)){await delay(1000);return {url,owned:true,child,logFile};}
  await delay(1000);
 }
 await stopServer({child,logFile});throw Error(`Next server startup timeout; read ${serverLog}`);
}
async function stopServer(server){
 if(!server?.owned&&!server?.child)return;
 const child=server.child;
 if(child&&child.exitCode===null){
  if(process.platform==='win32')await new Promise(resolve=>{
    const killer=spawn(cmdExe,['/d','/s','/c',`taskkill /PID ${child.pid} /T /F`],{windowsHide:true,stdio:'ignore'});
    killer.once('error',resolve);killer.once('close',resolve);
  });else {try{child.kill('SIGTERM')}catch{}}
 }
 server.logFile?.end();
}
async function callAnnual(url,causalPath){
 const start=performance.now();
 const r=await fetch(`${url}/api/tank-comparison`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({...body,causalPath}),signal:AbortSignal.timeout(180000),cache:'no-store'});
 const wallMs=performance.now()-start;
 const t=await r.text();if(!r.ok)throw Error(`Annual path ${causalPath} HTTP ${r.status}: ${t.slice(0,2500)}`);
 const j=JSON.parse(t);if(j.causalPath!==causalPath)throw Error(`Annual ${causalPath} returned ${j.causalPath}`);
 if(!Array.isArray(j.points)||j.points.length!==28)throw Error(`Annual ${causalPath} returned ${j.points?.length} tanks instead of 28`);
 const volumes=j.points.map(p=>p.storageVolume).sort((a,b)=>a-b);
 for(let i=0;i<28;i++)if(volumes[i]!==300+i*100)throw Error(`Unexpected volume list ${volumes.join(',')}`);
 if(j.gpuLockstep?.annualSteps!==35040)throw Error(`Annual ${causalPath} steps ${j.gpuLockstep?.annualSteps} != 35040`);
 if(j.year!==2023)throw Error(`Annual ${causalPath} year ${j.year} != 2023`);
 for(const p of j.points)for(const key of metrics)if(typeof p[key]!=='number'||!Number.isFinite(p[key]))throw Error(`Annual ${causalPath} invalid ${key} for ${p.storageVolume}L`);
 return {causalPath,wallMs,apiTotalMs:j.totalDurationMs,sweepMs:j.sweepDurationMs,gpu:j.gpuLockstep?.gpu??null,points:j.points};
}
function comparePoints(a,b){
 const m={};let exact=true;
 const get=p=>new Map(p.map(x=>[x.storageVolume,x]));
 const aa=get(a),bb=get(b);
 for(const key of metrics){let maxAbs=0,maxRel=0,tank=null,exactCount=0;
  for(const [v,x] of aa){const y=bb.get(v);if(!y)throw Error(`B missing ${v}L`);const diff=Math.abs(x[key]-y[key]);if(diff!==0)exact=false;else exactCount++;
   const rel=diff/Math.max(1,Math.abs(x[key]),Math.abs(y[key]));if(diff>maxAbs){maxAbs=diff;tank=v;}maxRel=Math.max(maxRel,rel);
  }
  m[key]={maxAbs,maxRel,maxAbsAtVolumeL:tank,exactCount};
 }
 const minVolume=(points,key)=>points.reduce((a,b)=>b[key]<a[key]?b:a).storageVolume;
 return {classification:exact?'EXACT':'DIFFERENT',metrics:m,
  optimumVolumeNetEnergyCost:{A:minVolume(a,'netEnergyCost'),B:minVolume(b,'netEnergyCost')},
  optimumVolumeGridImport:{A:minVolume(a,'gridImport'),B:minVolume(b,'gridImport')}};
}
const median=a=>{if(!a.length)return null;const s=[...a].sort((x,y)=>x-y);return s.length%2?s[s.length>>1]:(s[s.length/2-1]+s[s.length/2])/2};
async function runAnnual(){
 log('3/4 Vollständiger 28-Tank-Jahreslauf über die EXISTIERENDE API: explizit A FP64 gegen B FP32 (nicht generischer Compiler)');
 ensureProductionRoute();
 let server=null;
 try{
  server=await openServer();
  const rows=[];
  for(const pathName of ['A','B']){log(`Warmup ${pathName} (nicht gewertet)`);await callAnnual(server.url,pathName);}
  for(const pathName of ['A','B','B','A']){
   log(`Gemessener Jahreslauf ${pathName}`);const item=await callAnnual(server.url,pathName);
   rows.push(item);log(`Jahr ${pathName}: API ${Number(item.apiTotalMs).toFixed(1)} ms, Sweep ${Number(item.sweepMs).toFixed(1)} ms, wall ${item.wallMs.toFixed(1)} ms`);
  }
  const by=c=>rows.filter(x=>x.causalPath===c);
  const [A,B]=[by('A'),by('B')];
  const parity=comparePoints(A[0].points,B[0].points);
  const med={A:{wallMs:median(A.map(x=>x.wallMs)),apiTotalMs:median(A.map(x=>x.apiTotalMs)),sweepMs:median(A.map(x=>x.sweepMs))},B:{wallMs:median(B.map(x=>x.wallMs)),apiTotalMs:median(B.map(x=>x.apiTotalMs)),sweepMs:median(B.map(x=>x.sweepMs))}};
  const existingAoverB={sweep:med.A.sweepMs/med.B.sweepMs,apiTotal:med.A.apiTotalMs/med.B.apiTotalMs,requestWall:med.A.wallMs/med.B.wallMs};
  const json={status:'PASS',note:'These are EXISTING FP64 CUDA path A and B14P CUDA path B. Neither is the new generic source compiler.',serverUrl:server.url,serverOwned:server.owned,
   warmupsPerPath:1,measuredPerPath:2,order:['A','B','B','A'],medians:med,existingAoverB,parity,
   measurements:rows.map(x=>({causalPath:x.causalPath,wallMs:x.wallMs,apiTotalMs:x.apiTotalMs,sweepMs:x.sweepMs,steps:35040,tanks:28})),
   samplePoints:{A:A[0].points,B:B[0].points}};
  report.yearlyProduction=json;
  stage('28-tank 35040-step EXISTING production A/B annual', 'PASS', {parity:parity.classification,medians:med});
 }catch(e){report.yearlyProduction={status:'FAIL',error:String(e?.stack??e).slice(0,6000),serverLog};stage('28-tank 35040-step EXISTING production A/B annual','FAIL',{error:String(e)});}
 finally{await stopServer(server);}
}
function finishGate(){
 const gpuPass=report.genericCuda?.status==='PASS';
 const gridPass=report.gpuDispatch?.gpu?.status==='PASS';
 const nativePass=report.originalA1A2?.status==='PASS';
 const yearPass=report.yearlyProduction?.status==='PASS';
 report.summary={genericCUDAParity:gpuPass?'PASS':'FAIL',genericGridDispatcher:gridPass?'PASS':'FAIL',originalA1A2Benchmark:nativePass?'PASS':'FAIL',existingFullYear:yearPass?'PASS':'FAIL',generic28TankFullYear:'BLOCK',productionGate:'BLOCK',
  explanation:'Native generic compiler is not integrated with the annual simulation. Existing annual A/B measurements are baseline only. Neither can establish generic-vs-original annual speedup.'};
 report.readyForProduction=false;
 stage('Generic A production acceptance','BLOCK',{reason:report.summary.explanation});
}
// No external tools required for self-test. Validates reporting and parity helper.
if(process.argv.includes('--self-test')){
 const a=[{storageVolume:300,...Object.fromEntries(metrics.map(k=>[k,1]))}];
 const b=structuredClone(a);
 if(comparePoints(a,b).classification!=='EXACT')throw Error('self-test parity equality failed');
 b[0].gridImport+=1;
 if(comparePoints(a,b).classification!=='DIFFERENT')throw Error('self-test parity difference failed');
 log('SELF_TEST_PASS');logger.end();process.exit(0);
}
try{
 log('SunOpt Approach A ONE TEST — begins. Does not modify production.');
 await runStructured();
 await runDispatch();
 await runAnnual();
 finishGate();
}catch(e){stage('Unhandled combined runner failure','FAIL',{error:String(e?.stack??e).slice(0,8000)});finishGate();}
finally{save();log(`REPORT ${reportFile}`);log(`LOG ${testLog}`);log('PRODUCTION_GATE=BLOCK until source-generated A actually executes 28-tank annual parity');logger.end();}
process.exitCode=report.stages.some(s=>s.status==='FAIL')?1:0;
