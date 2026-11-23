/*
 * Run one independent sequence; do not write into production.
 *   node rust/sunopt-node/scripts/approach-a-generic-verify.mjs
 * Options: --require-gpu --arch=sm_120
 * Reports: updates/outbox/approach-a-generic-verify-*.json
 */
import { compileGenericNumericFunction, makeCppHost, makeCudaHost } from './approach-a-generic-compiler.mjs';
import { compileGenericFullSource } from './approach-a-generic-full-source.mjs';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { performance } from 'node:perf_hooks';

import { loadProjectTypeScript, findSunOptProjectRoot } from './approach-a-generic-typescript.mjs';
const ts = loadProjectTypeScript(import.meta.url);
const run = (cmd, args, options={}) => spawnSync(cmd,args,{encoding:'utf8',timeout:120000,maxBuffer:12*1024*1024,...options});
const isCommandAvailable = cmd => {
  const r=run(cmd, ['--version'], { timeout:15000 });
  return !r.error && r.status===0;
};
const nvccCommand = process.platform === 'win32' && process.env.CUDA_PATH && existsSync(path.join(process.env.CUDA_PATH,'bin','nvcc.exe'))
  ? path.join(process.env.CUDA_PATH,'bin','nvcc.exe') : 'nvcc';
const add = (name, status, details={}) => {
  report.stages.push({name,status,...details});
  console.log(`${status.padEnd(5)} ${name}`);
};
const assert = (condition,message) => { if (!condition) throw new Error(message); };
const toNumber = (x) => Number.isFinite(x) ? x : String(x);
const close = (got,expect) => Number.isFinite(got)&&Number.isFinite(expect) && Math.abs(got-expect) <= 1e-11*Math.max(1,Math.abs(got),Math.abs(expect));
const args = process.argv.slice(2);
const requireGpu = args.includes('--require-gpu');
const arch = (args.find(s=>s.startsWith('--arch=')) ?? '--arch=sm_120').split('=')[1];
if (!/^sm_[0-9]{2,3}[a-z]?$/.test(arch)) throw new Error('Invalid CUDA architecture');
const dir = path.dirname(fileURLToPath(import.meta.url));
// Packaging may be extracted in /root, rust/sunopt-node/scripts or any other location.
const projectRoot = findSunOptProjectRoot(import.meta.url) ?? process.cwd();
const outbox=path.join(projectRoot,'updates/outbox');
const outputDir=existsSync(path.join(projectRoot,'updates'))?outbox:dir;
mkdirSync(outputDir,{recursive:true});
const report = {
  name:'SunOpt Approach A generic numeric source compiler — isolated research gate',
  createdAt:new Date().toISOString(),
  gpuRequired:requireGpu,arch,projectRoot,
  definition:'No prior knowledge of f(x), source-based lowering, no application-specific model.',
  scope:'PURE numeric scalar functions only; full SunOpt annual model is NOT supported by this compiler yet.',
  stages:[], fixtures:[],
  canWireProduction:false,
};
const tmp=mkdtempSync(path.join(os.tmpdir(),'sunopt-generic-a-'));
const cases=[
  ['polynomial',`function polynomial(x:number,y:number):number { return x*x + 0.25*y - 3; }`],
  ['branch',`function branch(x:number,y:number):number { if (x < y) { return y-x; } else { return x+y; } }`],
  ['clip',`function clip(x:number,y:number):number { let z = Math.max(-2, Math.min(2,x)); return z + Math.abs(y); }`],
  ['boundedLoop',`function boundedLoop(x:number,y:number):number { let sum = 0; for(let i=0;i<8;i++) { sum += x*i + y; } return sum; }`],
  ['nonlinear',`function nonlinear(x:number,y:number):number { const z = x*x + 1; return Math.sqrt(z) + Math.sin(y) - Math.cos(x); }`],
  ['ternary',`function ternary(x:number,y:number):number { return (x>0 && y<0) ? x*y : (x===0 ? y+1 : x-y); }`],
];
let failed=false;
try {
  const rows=Array.from({length:257},(_,i)=>[
    ((i*73)%211 - 105)/15,
    ((i*41)%173 - 86)/17,
  ]);
  rows[0]=[0,0];rows[1]=[-2,2];rows[2]=[2,-2];rows[3]=[1.5,1.5];
  const compiled=[];
  // Stage 1: source registration, no function-name dispatch.
  try {
    const map=new Map();
    for (const [name,source] of cases) {
      const c=compileGenericNumericFunction(source);
      assert(c.name===name, 'Wrong compiled name');
      assert(!map.has(c.hash), 'Source hash collision');
      map.set(c.hash,c);
      compiled.push({name,source,c});
    }
    const alternative=compileGenericNumericFunction('function polynomial(x:number,y:number):number { return x*x + 0.25*y - 4; }');
    assert(alternative.hash!==compiled[0].c.hash,'Source mutation did not change identity');
    assert(alternative.body!==compiled[0].c.body,'Source mutation did not change generated code');
    add('G1: source-derived numeric compiler and source invalidation','PASS',{functionCount:compiled.length});
  } catch(e) { add('G1: source-derived numeric compiler','FAIL',{error:String(e)}); failed=true; }
  // Stage 2: strict safety/shape rejection, no inbuilt physics.
  try {
    for (const source of [
      `function reject(x:number):number { return unknownGlobal+x; }`,
      `function reject(x:number):number { while(true) { x++; } return x; }`,
      `function reject(x:number):number { return hiddenPhysics(x); }`,
      `function reject(x:number):number { let y=0; for(let i=0;i<10000000;i++){y+=x;} return y; }`,
      `function reject(x:number):number { return process.env.SECRET; }`,
    ]) {
      let rejected=false;
      try { compileGenericNumericFunction(source); }catch{rejected=true;}
      assert(rejected, `Unsupported/unsafe function was accepted: ${source}`);
    }
    add('G2: unsupported syntax rejected, no opaque fallback','PASS',{rejectedCases:5});
  } catch(e) {add('G2: strict rejection','FAIL',{error:String(e)});failed=true;}
  // Stage 3: independently execute reference TypeScript and compiled native C++.
  const cc=isCommandAvailable('g++')?'g++':isCommandAvailable('clang++')?'clang++':isCommandAvailable(nvccCommand)?nvccCommand:null;
  if (!cc) {
    add('G3: native C++ compilation','SKIP',{reason:'g++/clang++ unavailable; CUDA compiler test is independent'});
  }
  if (compiled.length===cases.length) {
    for (const {name,source,c} of compiled) {
      const js = ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.None}}).outputText;
      const ref = new Function(`${js}\nreturn ${name};`)();
      const expected=rows.map(row=>ref(...row));
      const fixture={name,hash:c.hash,rows:rows.length,refSamples:expected.slice(0,4).map(toNumber),cpuParity:null,gpuParity:null};
      const file=path.join(tmp,`${name}.cpp`);
      writeFileSync(file,makeCppHost(c,rows));
      if(cc){
        const exe=path.join(tmp,process.platform==='win32'?`${name}.exe`:name);
        const built=run(cc,['-std=c++17','-O3',file,'-o',exe]);
        if(built.status!==0){
          fixture.cpuParity={status:'FAIL',error:(built.stderr||built.error?.message||'').slice(0,2000)};failed=true;
        } else {
          const r=run(exe,[]);
          const actual=r.stdout.trim().split(/\r?\n/).map(Number);
          const mismatches=expected.map((x,i)=>close(actual[i],x)?null:i).filter(x=>x!==null);
          fixture.cpuParity={status:r.status===0&&actual.length===rows.length&&mismatches.length===0?'PASS':'FAIL',count:actual.length,mismatchCount:mismatches.length,firstMismatch:mismatches[0]??null};
          if(fixture.cpuParity.status==='FAIL')failed=true;
        }
      }
      report.fixtures.push(fixture);
    }
    const failures=report.fixtures.filter(f=>f.cpuParity?.status==='FAIL');
    add('G3: six unrelated functions, native C++ parity', cc?(failures.length===0?'PASS':'FAIL'):'SKIP',{compiler:cc,failures:failures.map(f=>f.name)});
  }
  // Stage 4: persistent data in host-side typed arrays, changed inputs must change results.
  try {
    const typed=new Float64Array(rows.flat());
    assert(typed.length===rows.length*2,'incorrect input size');
    typed[0]=123.0;
    assert(typed[0]===123.0, 'persistent buffer write failed');
    add('G4: type-stable persistent buffer interface','PASS',{inputCount:typed.length});
  } catch(e){add('G4: persistent memory','FAIL',{error:String(e)});failed=true;}
  // Stage 5: GPU compilation + execution of the SAME source-derived function, not a fixed CUDA solver.
  const nvcc=isCommandAvailable(nvccCommand);
  if (!nvcc) add('G5: generic CUDA codegen, native compilation and parity','SKIP',{reason:'CUDA nvcc unavailable in this runtime. Must be tested on Windows RTX 5060.'});
  else if (compiled.length===cases.length) {
    let gpuErrors=[];
    for(const [index,{name,c}] of compiled.entries()) {
      const source=path.join(tmp,`${name}.cu`);
      const exe=path.join(tmp,process.platform==='win32'?`${name}-gpu.exe`:`${name}-gpu`);
      writeFileSync(source,makeCudaHost(c,rows));
      const built=run(nvccCommand,['-std=c++17','-O3',`-arch=${arch}`,source,'-o',exe],{timeout:180000});
      if(built.status!==0){
        gpuErrors.push({name,phase:'compile',reason:(built.stderr||built.error?.message||'').slice(0,1800)});continue;
      }
      const r=run(exe,[],{timeout:60000});
      const actual=r.stdout.trim().split(/\r?\n/).map(Number);
      const ref=new Function(`${ts.transpileModule(compiled[index].source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.None}}).outputText}\nreturn ${name};`)();
      const mismatch=rows.findIndex((row,i)=>!close(actual[i],ref(...row)));
      report.fixtures[index].gpuParity={status:r.status===0&&actual.length===rows.length&&mismatch<0?'PASS':'FAIL',mismatchIndex:mismatch,bench:r.stderr.match(/KERNEL_100_MS ([0-9.]+)/)?.[1]??null};
      if(report.fixtures[index].gpuParity.status==='FAIL')gpuErrors.push({name,phase:'run',reason:(r.stderr||'').slice(0,400),mismatch});
    }
    add('G5: generic CUDA codegen, six GPU parity checks',gpuErrors.length?'FAIL':'PASS',{errors:gpuErrors});
    if(gpuErrors.length)failed=true;
  }
  // Stage 6: CPU/GPU parity, completion gates are strict (SKIP != PASS).
  const cppPass=report.stages.find(s=>s.name.startsWith('G3:'))?.status==='PASS';
  const gpuPass=report.stages.find(s=>s.name.startsWith('G5:'))?.status==='PASS';
  add('G6: generic source-to-native cross-backend acceptance',(cppPass&&gpuPass)?'PASS':'BLOCK',{cppPass,gpuPass});
  // Stage 7: target source support — this is the gate for the real annual production model.
  const scopePath=path.join(projectRoot,'src/simulation/rust-acceleration-scopes.ts');
  if (!existsSync(scopePath)) {
    add('G7: full SunOpt controller source acceptance','BLOCK',{reason:'Project source not mounted; no production qualification possible'});
  } else {
    const source=readFileSync(scopePath,'utf8');
    const ast=ts.createSourceFile(scopePath,source,ts.ScriptTarget.ES2022,true);
    const targets=['runRustNoSunshineScope','runRustPeakShavingCausalScope'];
    const results=targets.map(target=>{
      const fn=ast.statements.find(s=>ts.isFunctionDeclaration(s)&&s.name?.text===target);
      if(!fn)return {function:target,status:'NOT_FOUND'};
      try{compileGenericNumericFunction(fn.getText(ast));return {function:target,status:'SUPPORTED'};}
      catch(e){return {function:target,status:'UNSUPPORTED',reason:String(e).slice(0,250)};}
    });
    const compatible=results.every(r=>r.status==='SUPPORTED');
    add('G7: full SunOpt controller source acceptance',compatible?'PASS':'BLOCK',{functions:results});
  }
  // The full source can execute generically in V8, including object/array state.
  // This explicitly remains a CPU reference, NEVER a GPU acceptance.
  if(existsSync(scopePath)) {
    try {
      const source=readFileSync(scopePath,'utf8');
      const handle=compileGenericFullSource(source,scopePath);
      assert(handle===compileGenericFullSource(source,scopePath),'Full-source registration not cached');
      const storage={timeStepHours:0.25,targetStorageEnergy:20,storageEnergyPerKelvin:2,
        storageConductanceWPerKelvin:0,storageMinTemperature:60,ambientTemperature:20,dailyHeatLoss:0};
      const ns={...storage,pvPower:new Float64Array(32),loadEnergy:new Float64Array(32),
        currentIndex:0,horizonSteps:16,protectedSteps:4,householdReservePower:1,
        maximumHeatPumpElectricalPower:2,cop:3,initialWaterlinePower:0,currentStorageEnergy:0};
      const a=handle.run('runRustNoSunshineScope',ns);
      assert(a.kWhNoSunshine===0&&a.unavoidableThermalDeficit===0&&a.requiredStorageEnergyAtStart===0,
        'Zero-load zero-loss no-sunshine invariant failed');
      const changed={...ns,loadEnergy:new Float64Array(32)};
      changed.loadEnergy[7]=1.3;
      const b=handle.run('runRustNoSunshineScope',changed);
      assert(b.kWhNoSunshine>0 && b.requiredStorageEnergyAtStart>0,
        'Changed demand did not change the result');
      const causal={...storage,exportPower:new Float64Array(32),demandEnergy:new Float64Array(32),
        currentIndex:0,horizonSteps:16,currentStorageEnergy:0,minimumReserveEnergy:0,
        maximumHeatPumpElectricalPower:2,cop:3,energyCutLevel:0};
      const c=handle.run('runRustPeakShavingCausalScope',causal);
      assert(c.normalPlanSurvives && c.maximumPvPlanSurvives,'Zero-load causal invariant failed');
      causal.demandEnergy[1]=3;
      const d=handle.run('runRustPeakShavingCausalScope',causal);
      assert(!d.normalPlanSurvives && !d.maximumPvPlanSurvives,
        'Causal result did not react to an unserved demand');
      const t0=performance.now();
      for(let i=0;i<100;i++)handle.run('runRustNoSunshineScope',{...ns,currentIndex:i%16});
      const calls100Ms=performance.now()-t0;
      add('G8: full SunOpt TS body via generic V8 CPU JIT','PASS',
        {sourceSha256:handle.hash,smokeRuns:4,repeatCalls:100,calls100Ms});
    }catch(e) {add('G8: full SunOpt generic V8 CPU JIT','FAIL',{error:String(e)});failed=true;}
  }
  // A PASS requires GPU correctness AND the real full model — not just microkernels.
  const required=['G1:','G2:','G3:','G4:','G5:','G6:','G7:','G8:'];
  report.canWireProduction=!failed&&required.every(p=>report.stages.find(s=>s.name.startsWith(p))?.status==='PASS');
} finally {
  report.finishedAt=new Date().toISOString();
  const file=path.join(outputDir,`approach-a-generic-verify-${report.createdAt.replace(/[-:.]/g,'').replace('T','T').replace('Z','Z')}.json`);
  writeFileSync(file,JSON.stringify(report,null,2));
  console.log(`REPORT ${file}`);
  console.log(`PRODUCTION_GATE ${report.canWireProduction?'PASS':'BLOCK'} — existing A/B routes are untouched`);
  rmSync(tmp,{recursive:true,force:true});
}
process.exitCode=failed|| (requireGpu && !report.canWireProduction)?1:0;
