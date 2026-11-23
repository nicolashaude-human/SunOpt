/* A9 isolated ABI/parity test, runs compiled CUDA or CPU sidecar. */
import {readFileSync,writeFileSync,mkdirSync,existsSync} from 'node:fs';
import {Worker} from 'node:worker_threads';
import path from 'node:path';
import {performance} from 'node:perf_hooks';
import {compileStructuredSource} from './approach-a-structured-compiler.mjs';
import {compileGenericFullSource} from './approach-a-generic-full-source.mjs';
import {findSunOptProjectRoot} from './approach-a-generic-typescript.mjs';
const root=findSunOptProjectRoot(import.meta.url)??process.cwd();
const outbox=path.join(root,'updates/outbox');mkdirSync(outbox,{recursive:true});
const backend=process.argv.includes('--cpu')?'cpu':'cuda';
const extra=readFileSync(path.join(root,'src/simulation/approach-a9-energy-scopes.ts'),'utf8');
const source=readFileSync(path.join(root,'src/simulation/rust-acceleration-scopes.ts'),'utf8')+'\n\n'+extra;
const ast=compileStructuredSource(source),reference=compileGenericFullSource(source);
const buildFile=path.join(outbox,`approach-a9-build-${backend}.json`);
if(!existsSync(buildFile))throw Error(`Build metadata missing: ${buildFile}`);
const build=JSON.parse(readFileSync(buildFile,'utf8'));
if(!build.compiled||!existsSync(build.outputFile)||build.sourceHash!==ast.hash){
 throw Error('A9 executable does not match current source/has not compiled');
}
const max=128,confOffset=64,dynOffset=confOffset+12*max*8,outOffset=dynOffset+3*max*8,errOffset=outOffset+max*8;
const shared=new SharedArrayBuffer(errOffset+4096);
const status=new Int32Array(shared,0,16),conf=new Float64Array(shared,confOffset,12*max);
const dyn=new Float64Array(shared,dynOffset,3*max),out=new Float64Array(shared,outOffset,max);
const errors=new Uint8Array(shared,errOffset,4096);
const length=process.argv.includes('--year-length')?35040:192;
const tanks=28,steps=length===35040?96:128,horizon=length===35040?288:64;
const exportPower=new Float64Array(length),demandEnergy=new Float64Array(length);
for(let i=0;i<length;i++){
 exportPower[i]=Math.max(0,8*Math.sin(i*0.11)+2.7);demandEnergy[i]=i%17===0?1.5:0.08;
}
const state=new Float64Array(tanks);
const common=[];
for(let k=0;k<tanks;k++){
 const b=12*k;
 conf[b+2]=20+k;conf[b+3]=1.8+k*.01;conf[b+4]=3.25;
 conf[b+6]=.25;conf[b+7]=1.2+k*.02;conf[b+8]=2.1+k*.03;
 conf[b+9]=60;conf[b+10]=17;conf[b+11]=k%3===0?-1:k%3===1?0.9:0;
 state[k]=(20+k)*(.4+.01*(k%9));
 common.push({timeStepHours:conf[b+6],targetStorageEnergy:conf[b+2],storageEnergyPerKelvin:conf[b+7],
  storageConductanceWPerKelvin:conf[b+8],storageMinTemperature:conf[b+9],ambientTemperature:conf[b+10],dailyHeatLoss:conf[b+11]});
}
const worker=new Worker(path.join(root,'rust/sunopt-node/scripts/approach-a9-ipc-worker.mjs'),
 {workerData:{shared,binary:build.outputFile,exportPower,demandEnergy}});
const report={createdAt:new Date().toISOString(),backend,sourceHash:ast.hash,
 compiledFromSource:true,fullProductionAnnual:false,tanks,steps,horizon,annualValues:length,comparisons:0,mismatches:[],
 durationMs:null,sidecarStepWallMs:0,typescriptReferenceWallMs:0,
 fixtureChecks:{nonzeroRequestedElectricalEnergy:0,positiveMinimumReserveEnergy:0,
  positiveCutLevel:0,zeroCutLevel:0},
 status:'INCOMPLETE',productionGate:'BLOCK'};
function call(op,other){
 Atomics.store(status,0,0);worker.postMessage({op,...other});
 const result=Atomics.wait(status,0,0,60000);
 if(result==='timed-out'||Atomics.load(status,0)!==1){
  const end=errors.indexOf(0);throw Error(`A9 ${op} ${result}: ${new TextDecoder().decode(errors.subarray(0,end<0?errors.length:end))}`);
 }
}
const approx=(a,b)=>Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=1e-8*Math.max(1,Math.abs(a),Math.abs(b));
try{
 call('configure',{tanks});
 const begin=performance.now();
 for(let t=0;t<steps;t++){
  for(let k=0;k<tanks;k++){
   // The target energy is stride-12 field #2 (not #0, which is reserved).
   // Preserve positive reserves/requests to exercise the real Causal solver.
   const b=k*3,c=conf[k*12+2];
   dyn[b]=state[k];
   // Exercise both feasible and reserve-constrained plans. This is test data,
   // not a compiler special case: the generic source decides the outcome.
   dyn[b+1]=k%4===1?.9*c:k%4===2?.7*c:Math.min(3.5+k*.1,c);
   dyn[b+2]=Math.max(0,Math.min(c-state[k],4)/conf[k*12+4]);
   if(dyn[b+1]>0)report.fixtureChecks.positiveMinimumReserveEnergy++;
   if(dyn[b+2]>0)report.fixtureChecks.nonzeroRequestedElectricalEnergy++;
  }
  call('step',{index:t%length,horizon});
  for(let k=0;k<tanks;k++){
   const b=k*3,commonData=common[k];
   const energy=reference.run('runA9EnergyCutLevel',{
    exportPower,currentIndex:t%length,horizonSteps:horizon,
    requestedElectricalEnergy:dyn[b+2],maximumHeatPumpElectricalPower:conf[k*12+3],timeStepHours:.25,
   });
   const ref=reference.run('runRustPeakShavingCausalScope',{
      ...commonData,exportPower,demandEnergy,currentIndex:t%length,horizonSteps:horizon,
      currentStorageEnergy:dyn[b],minimumReserveEnergy:dyn[b+1],
      maximumHeatPumpElectricalPower:conf[k*12+3],cop:conf[k*12+4],energyCutLevel:energy,
   });
   if(out[k]>1e-8)report.fixtureChecks.positiveCutLevel++;
   else if(out[k]===0)report.fixtureChecks.zeroCutLevel++;
   if(!approx(out[k],ref.cutLevelPower)&&report.mismatches.length<16)
      report.mismatches.push({step:t,tank:k,actual:out[k],reference:ref.cutLevelPower});
   report.comparisons++;
   const delivered=Math.min(conf[k*12+3],Math.max(0,exportPower[t%length]-out[k]));
   // Stateful test fixture only. Real physical step remains in SunOpt's controller.
   state[k]=Math.min(conf[k*12+2],Math.max(0,state[k]+delivered*.25*conf[k*12+4]-demandEnergy[t%length]));
  }
 }
 report.durationMs=performance.now()-begin;
 report.status=report.mismatches.length?'FAIL':'PASS';
 if(Object.values(report.fixtureChecks).some(n=>n===0)){
  report.status='FAIL';report.error='Degenerate fixture: positive reserve, requested energy, positive cuts and zero cuts must all be exercised';
 }
}catch(e){report.status='FAIL';report.error=String(e?.stack??e).slice(0,8000);}
finally{
 await worker.terminate();
 const file=path.join(outbox,`approach-a9-verify-${backend}-${new Date().toISOString().replace(/[-:.]/g,'')}.json`);
 writeFileSync(file,JSON.stringify(report,null,2)+'\n');
 console.log('A9_VERIFY_REPORT',file);
 console.log('A9_VERIFY_STATUS',report.status,'comparisons',report.comparisons,'mismatchCount',report.mismatches.length);
 if(report.status!=='PASS')process.exitCode=1;
}
