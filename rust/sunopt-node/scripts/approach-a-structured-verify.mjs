/*
 * Read-only Approach A research test. Generated C++/CUDA from source AST,
 * compare with trusted TypeScript and original native A1+A2 if available.
 * No production source, API route or CUDA native addon is modified.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs';
import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import os from 'node:os';
import { performance } from 'node:perf_hooks';
import { compileStructuredSource, getStructuredCompilerCacheSize } from './approach-a-structured-compiler.mjs';
import { compileGenericFullSource } from './approach-a-generic-full-source.mjs';
import { findSunOptProjectRoot } from './approach-a-generic-typescript.mjs';
const root=findSunOptProjectRoot(import.meta.url)??process.cwd();
const outputDir=path.join(root,'updates','outbox');mkdirSync(outputDir,{recursive:true});
const sourcePath=path.join(root,'src','simulation','rust-acceleration-scopes.ts');
const requireGpu=process.argv.includes('--require-gpu');
const requireOriginalA=process.argv.includes('--require-original-a');
const report={name:'Approach A generic structured source CPU/CUDA compiler',at:new Date().toISOString(),root,stages:[],comparisons:[],readyForProduction:false};
const add=(name,status,details={})=>{report.stages.push({name,status,...details});console.log(`${status}: ${name}`);};
const check=(value,msg)=>{if(!value)throw new Error(msg);};
const tmp=mkdtempSync(path.join(os.tmpdir(),'sunopt-structured-a-'));
const exec=(cmd,args,opts={})=>spawnSync(cmd,args,{encoding:'utf8',maxBuffer:64*1024*1024,timeout:180000,...opts});
// Preserve every failure channel. nvcc can put diagnostics on stdout,
// and on Windows spawn may fail without producing stderr at all.
function processDiagnostics(result, command, args){
 return {command,arguments:args,exitCode:result.status,signal:result.signal,
  spawnError:result.error?{code:result.error.code,message:result.error.message}:null,
  stdout:String(result.stdout??'').slice(-12000),stderr:String(result.stderr??'').slice(-12000)};
}

const num=x=>Number.isInteger(x)?`${x}.0`:String(x);
const approx=(a,b)=>a===b||(Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=1e-10*Math.max(1,Math.abs(a),Math.abs(b)));
const isNvidia=()=>{
 const candidate=process.platform==='win32'&&process.env.CUDA_PATH?path.join(process.env.CUDA_PATH,'bin','nvcc.exe'):'nvcc';
 const r=exec(candidate,['--version'],{timeout:8000});return r.status===0?candidate:null;
};
const arch=(process.argv.find(x=>x.startsWith('--arch='))??'--arch=sm_120').split('=')[1];
if(!/^sm_\d{2,3}[a-z]?$/.test(arch))throw Error('Unsafe CUDA architecture');
function makeCases(){
 const dataLength=192;
 const makeArrays=(seed,mode)=>{
  const a=new Float64Array(dataLength),b=new Float64Array(dataLength);
  for(let i=0;i<dataLength;i++){
   const sunlight=Math.max(0,9*Math.sin((i+seed*17)/18.6));
   a[i]=mode===0?0:(mode===1?1.2:sunlight+((i*29+seed*13)%7)*0.12);
   b[i]=mode===0?0:(mode===1?.15:((i*7+seed*11)%13===0?3.5:.12));
  }
  return {a,b};
 };
 const fixtures=[];
 for(let f=0;f<12;f++){
  const mode=f%4,arrays=makeArrays(f,mode);
  const base={timeStepHours:.25,targetStorageEnergy:25+f*2.5,storageEnergyPerKelvin:1.5+f*.1,storageConductanceWPerKelvin:2.1+f*.08,storageMinTemperature:60,ambientTemperature:18+f, dailyHeatLoss:f%3===0?0:f%3===1?1.1:-1};
  const ns={...base,pvPower:arrays.a,loadEnergy:arrays.b,currentIndex:(f*31)%dataLength,horizonSteps:48+f*5,protectedSteps:Math.min(24,48+f*5),householdReservePower:0.6+f*.04, maximumHeatPumpElectricalPower:1.2+f*.05,cop:3.0,initialWaterlinePower:.1,currentStorageEnergy:f*2};
  const ca={...base,exportPower:arrays.a,demandEnergy:arrays.b,currentIndex:(f*31)%dataLength,horizonSteps:48+f*5,currentStorageEnergy:f*2,minimumReserveEnergy:Math.min(5+f,base.targetStorageEnergy*.5),maximumHeatPumpElectricalPower:1.2+f*.05,cop:3.0,energyCutLevel:.4+f*.12};
  fixtures.push({entry:'runRustNoSunshineScope',input:ns},{entry:'runRustPeakShavingCausalScope',input:ca});
 }
 return fixtures;
}
const entryNames=['runRustNoSunshineScope','runRustPeakShavingCausalScope']; // test fixture names only; NOT used by code generator
function cppInput(program,type,input,index,registry,device=false){
 const meta=program.interfaces[type];
 if(!meta)throw Error(`Unknown input struct ${type}`);
 const parts=[];
 for(const parent of meta.bases)parts.push(cppInput(program,parent,input,index,registry,device));
 for(const [field,kind] of Object.entries(meta.ownFields)){
  const value=input[field];
  if(kind==='a_span'){
   check(value instanceof Float64Array,'Expected typed array for '+field);
   const label=`arr_${index}_${field}`;
   registry.push({label,values:value});
   parts.push(`a_span{${device?'d_'+label:label}, ${value.length}}`);
  }else if(kind==='bool')parts.push(value?'true':'false');
  else parts.push(num(value));
 }
 return `${type}{${parts.join(', ')}}`;
}
function resultKeys(program,type){return Object.keys(program.structs[type]??{});}
function generateCppHarness(program,cases,mode){
 const registry=[],values=[];
 for(let i=0;i<cases.length;i++){
  const f=program.functions.find(x=>x.name===cases[i].entry);check(f,'Missing entry function '+cases[i].entry);
  check(f.args.length===1,'Structured harness expects one input record');
  values.push(cppInput(program,f.args[0].type,cases[i].input,i,registry,mode==='gpu'));
 }
 const declarations=registry.map(({label,values})=>`static double ${label}[] = {${[...values].map(num).join(',')}};`).join('\n');
 const printResult=(entry,id,indexVar='j')=>{
  const f=program.functions.find(x=>x.name===entry),keyList=resultKeys(program,f.result);
  return `std::cout << "C ${id} " << ${keyList.map(k=>`r.${k}`).join(' << " " << ')} << "\\n";`;
 };
 const fragments=cases.map((c,i)=>{
  const f=program.functions.find(x=>x.name===c.entry);
  if(mode==='cpu')return `{ const ${f.args[0].type} arg = ${values[i]}; const auto r=${f.name}(arg); ${printResult(c.entry,i)} }`;
  return '';
 });
 if(mode==='cpu'){
   const bench=[];
   for(const entry of entryNames){
     const f=program.functions.find(x=>x.name===entry);
     const indices=cases.map((c,i)=>c.entry===entry?i:null).filter(i=>i!==null);
     const resultKey=resultKeys(program,f.result)[0];
     const lengthKey=Object.entries(program.structs[f.args[0].type]).find(([k,v])=>v==='a_span')?.[0];
     bench.push(`{ std::vector<${f.args[0].type}> inputs = {${indices.map(i=>values[i]).join(',')}};\nvolatile double checksum=0;\nconst int reps=2400;\nauto start=std::chrono::steady_clock::now();\nfor(int i=0;i<reps;i++){auto arg=inputs[i%inputs.size()];${lengthKey?`arg.currentIndex=fmod(arg.currentIndex+i, arg.${lengthKey}.length);`:''} auto r=${entry}(arg);checksum+=r.${resultKey};}\nauto end=std::chrono::steady_clock::now();\nstd::cerr<<"BENCH GENERIC_CPU ${entry} "<<std::chrono::duration<double,std::milli>(end-start).count()<<" "<<reps<<" "<<checksum<<"\\n";\n}`);
   }
   return `${program.cpp}\n#include <iostream>\n#include <iomanip>\n#include <vector>\n#include <chrono>\n${declarations}\nint main(){std::cout<<std::setprecision(17)<<std::boolalpha;${fragments.join('\n')}\n${bench.join('\n')}\nreturn 0;}`;
 }
 let gpuSource=`#include <cuda_runtime.h>\n#include <iostream>\n#include <iomanip>\n#include <vector>\n#include <chrono>\n${program.cpp}\n#define CUDA_ASSERT(x) do {auto er=(x);if(er!=cudaSuccess){std::cerr<<cudaGetErrorString(er)<<"\\n";return 2;}}while(0)\n${declarations}\n`;
 for(const entry of entryNames){
  const f=program.functions.find(x=>x.name===entry);
  const arrayName=Object.entries(program.structs[f.args[0].type]).find(([,v])=>v==='a_span')?.[0];
  gpuSource+=`\n__global__ void K_${entry}(const ${f.args[0].type}* in, ${f.result}* out, int count, double shift) {int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<count){auto arg=in[j];${arrayName?`if(shift>=0.0) arg.currentIndex=fmod(arg.currentIndex+shift+j, double(arg.${arrayName}.length));`:''}out[j]=${entry}(arg);}}`; 
 }
 gpuSource+='\nint main(){std::cout<<std::setprecision(17)<<std::boolalpha;\n';
 for(const {label,values} of registry){
  gpuSource+=`double *d_${label}=nullptr; CUDA_ASSERT(cudaMalloc((void**)&d_${label},${values.length}*sizeof(double))); CUDA_ASSERT(cudaMemcpy(d_${label},${label},${values.length}*sizeof(double),cudaMemcpyHostToDevice));\n`;
 }
 for(const entry of entryNames){
  const f=program.functions.find(x=>x.name===entry);
  const positions=cases.map((c,i)=>c.entry===entry?i:null).filter(v=>v!==null);
  gpuSource+=`{std::vector<${f.args[0].type}> host = {${positions.map(i=>values[i]).join(',')}};\n`;
  gpuSource+=`const int N=${positions.length}; ${f.args[0].type} *dInputs=nullptr; ${f.result} *dResults=nullptr;\n`;
  gpuSource+='CUDA_ASSERT(cudaMalloc((void**)&dInputs,sizeof(host[0])*N)); CUDA_ASSERT(cudaMalloc((void**)&dResults,sizeof('+f.result+')*N));\n';
  gpuSource+='CUDA_ASSERT(cudaMemcpy(dInputs,host.data(),sizeof(host[0])*N,cudaMemcpyHostToDevice));\n';
  gpuSource+=`K_${entry}<<<(N+63)/64,64>>>(dInputs,dResults,N,-1.0); CUDA_ASSERT(cudaDeviceSynchronize());\n`;
  gpuSource+=`std::vector<${f.result}> out(N); CUDA_ASSERT(cudaMemcpy(out.data(),dResults,sizeof(${f.result})*N,cudaMemcpyDeviceToHost));\n`;
  for(const [j,index] of positions.entries())gpuSource+=`{ auto r=out[${j}]; ${printResult(entry,index)} }\n`;
  // Same 2400 dynamic input evaluations per function as baseline A2 (12 jobs * 200 batches). GPU arrays resident; transfer and compilation excluded.
  gpuSource+=`auto t0=std::chrono::steady_clock::now();for(int repeat=0;repeat<200;repeat++)K_${entry}<<<(N+63)/64,64>>>(dInputs,dResults,N,double(repeat*N)); CUDA_ASSERT(cudaDeviceSynchronize());auto t1=std::chrono::steady_clock::now();\n`;
  gpuSource+=`std::cerr<<"B ${entry} "<<std::chrono::duration<double,std::milli>(t1-t0).count()<<"\\n";\n`;
  gpuSource+='CUDA_ASSERT(cudaFree(dInputs));CUDA_ASSERT(cudaFree(dResults));}\n';
 }
 for(const {label} of registry)gpuSource+=`CUDA_ASSERT(cudaFree(d_${label}));\n`;
 gpuSource+='return 0;}';
 return gpuSource;
}
function parseRows(stdout){const rows=new Map();for(const line of stdout.split(/\r?\n/)){
 if(!line.startsWith('C '))continue;const a=line.split(' ');rows.set(Number(a[1]),a.slice(2).map(s=>s==='true'?true:s==='false'?false:Number(s)));
 }return rows;}
function checkParity(program,cases,rows,label,baseline){
 const errors=[];
 for(let i=0;i<cases.length;i++){
  const data=rows.get(i),target=baseline[i];
  if(!data){errors.push({case:i,reason:'no native output'});continue;}
  const f=program.functions.find(x=>x.name===cases[i].entry),keys=resultKeys(program,f.result);
  if(data.length!==keys.length){errors.push({case:i,reason:'result length mismatch'});continue;}
  for(let j=0;j<keys.length;j++)if(!approx(data[j],target[keys[j]]))errors.push({case:i,key:keys[j],reference:target[keys[j]],native:data[j]});
 }
 add(label,errors.length?'FAIL':'PASS',{cases:cases.length,mismatches:errors.slice(0,8),mismatchCount:errors.length});
 return errors.length===0;
}
function findOldNative(){const p=path.join(root,'rust','sunopt-node','sunopt_node.win32-x64-msvc.node');if(!existsSync(p))return null;
 try{return createRequire(path.join(root,'package.json'))(p);}catch(e){add('Baseline A native addon availability','BLOCK',{error:String(e)});return null;}
}
function compareOriginalA(binding,source,cases,baseline){
 if(!binding){add('Original A1+A2 native benchmark','SKIP',{reason:'Native addon not found on this machine'});return;}
 try{
  const handleByName={runRustNoSunshineScope:binding.registerAccelerationScope(source,'rust_noSunshine'),runRustPeakShavingCausalScope:binding.registerAccelerationScope(source,'rust_peakShavingCausal')};
  const summaries=[];
  for(const entry of entryNames){
   const subset=cases.map((x,i)=>({...x,index:i})).filter(x=>x.entry===entry);
   const calls=entry==='runRustNoSunshineScope'?'runNoSunshineKernelContext':'runPeakShavingCausalKernelContext';
   check(typeof binding[calls]==='function','Missing original A2 context export');
   const specs=subset.map(x=>{
    const input=x.input;
    const arrays=entry==='runRustNoSunshineScope'?[input.pvPower,input.loadEnergy]:[input.exportPower,input.demandEnergy];
    const dataHandle=binding.registerAccelerationDataPair(...arrays);
    const params=Object.fromEntries(Object.entries(input).filter(([,v])=>!(v instanceof Float64Array)));
    return {dataHandle,params,x};
   });
   let mismatches=[];
   for(const s of specs){
    const actual=binding[calls](handleByName[entry],s.dataHandle,s.params),ref=baseline[s.x.index];
    for(const [k,v] of Object.entries(ref))if(!approx(v,actual[k]))mismatches.push({index:s.x.index,key:k,originalNative:actual[k],reference:v});
   }
   if(mismatches.length)throw Error('Original A native parity failed: '+JSON.stringify(mismatches.slice(0,4)));
   for(const s of specs)binding[calls](handleByName[entry],s.dataHandle,s.params);
   const repetitions=2400, t0=performance.now();
   for(let i=0;i<repetitions;i++){const s=specs[i%specs.length],parameters={...s.params,currentIndex:(s.params.currentIndex+i)%s.x.input[entry==='runRustNoSunshineScope'?'pvPower':'exportPower'].length};binding[calls](handleByName[entry],s.dataHandle,parameters);}
   const ms=performance.now()-t0;
   summaries.push({entry,repetitions,wallMs:ms,callMicroseconds:ms*1000/repetitions});
  }
  add('Original A1+A2 native benchmark','PASS',{comparisons:summaries});
 }catch(e){add('Original A1+A2 native benchmark','FAIL',{error:String(e)});}
}
let compiled,source,fixtures,baseline;
try{
 if(!existsSync(sourcePath))throw Error(`Missing source ${sourcePath}`);
 source=readFileSync(sourcePath,'utf8');
 compiled=compileStructuredSource(source,{filename:sourcePath});
 const wanted=entryNames.every(n=>compiled.functions.some(x=>x.name===n));
 check(wanted,'Missing required functions');
 check(compileStructuredSource(source,{filename:sourcePath})===compiled,'A1 compiler cache not reusing registered source');
 check(getStructuredCompilerCacheSize()>0,'A1 compiler cache empty');
 add('G7a: typed AST -> complete C++ generation and A1 source reuse','PASS',{sourceHash:compiled.hash,generatedBytes:compiled.cpp.length,functions:compiled.functions.map(x=>x.name),cacheSize:getStructuredCompilerCacheSize()});
 try{
  const alt=compileStructuredSource(source.replace('const ENERGY_EPSILON =','const ENERGY_EPSILON ='),{filename:sourcePath});
  check(alt.hash===compiled.hash,'hash unstable');
  // Changing a source operation must change the emitted native program.
  const changed=compileStructuredSource(source.replace('Math.min(\n    maximum,','Math.max(\n    maximum,'),{filename:sourcePath});
  check(changed.hash!==compiled.hash && changed.cpp!==compiled.cpp,'compiler ignores source mutation');
  add('G7b: generic source-invalidation test','PASS');
 }catch(e){add('G7b: source mutation test','FAIL',{error:String(e)});}
 // An unrelated object+array program must be lowered through the exact same compiler.
 try{
  const unrelated=`interface V { samples: Float64Array; offset: number; scale: number; }
interface R { value: number; flag: boolean; }
function arbitrary(x: V): R { const {samples,offset,scale}=x; let s=0; for(let i=0;i<3;i++){s+=samples[(offset+i)%samples.length] ?? 0;} return {value:s*scale, flag:s>0}; }`;
  const other=compileStructuredSource(unrelated);
  check(!other.cpp.includes('runRustNoSunshineScope'),'Compiler introduced application functions');
  const p=path.join(tmp,'unrelated.cpp'), exe=path.join(tmp,process.platform==='win32'?'unrelated.exe':'unrelated');
  writeFileSync(p,`${other.cpp}\n#include <iostream>\nint main(){double a[3]={1.0,2.0,3.0};V x{a_span{a,3},1.0,2.0};auto r=arbitrary(x);std::cout<<r.value<<" "<<r.flag<<"\\n";}`);
  const localCompiler=process.env.CXX|| (process.platform==='win32'?(isNvidia()??'g++'):'g++');
  const build=exec(localCompiler,['-std=c++17','-O2',p,'-o',exe]);
  check(build.status===0,'Unrelated native compiler: '+String(build.stderr));
  const run=exec(exe,[]);check(run.status===0&&run.stdout.trim()==='12 1','Unrelated native result: '+run.stdout);
  let rejected=false;try{compileStructuredSource('function bad(x:number):number{return mystery(x);}');}catch{rejected=true;}
  check(rejected,'Unknown external function not rejected');
  add('G7b2: unrelated structured function and strict unknown-function rejection','PASS');
 }catch(e){add('G7b2: unrelated structured function and strict rejection','FAIL',{error:String(e).slice(0,1800)});}
 // G7f0: CUDA extended-lambda lowering must use value captures while
 // preserving writes to variables in an enclosing function. This is
 // deliberately an unrelated numeric program, not a SunOpt special case.
 try {
  const closureSource=`interface Samples { data: Float64Array; count: number; }
interface ClosureResult { value: number; calls: number; }
function reduceSamples(x: Samples): ClosureResult {
 const { data, count } = x;
 let calls = 0;
 function sample(i: number): number { return data[i % data.length] ?? 0; }
 function visit(i: number): number { calls += 1; return sample(i); }
 let sum = 0;
 for (let i = 0; i < count; i++) { sum += visit(i); }
 return { value: sum, calls };
}`;
  const closure=compileStructuredSource(closureSource,{filename:'independent-closure.ts'});
  check(!closure.cpp.includes('[&] A_HD'),'Reference-capturing extended lambda still generated');
  check(closure.cpp.includes('a_capture_write_visit_calls'),'Mutable outer variable was not converted to explicit pointer');
  const testCpp=path.join(tmp,'closure-conversion.cpp');
  const testExe=path.join(tmp,process.platform==='win32'?'closure-conversion.exe':'closure-conversion');
  writeFileSync(testCpp,closure.cpp+`\n#include <iostream>\nint main(){ double data[3]={1,2,3}; Samples s{a_span{data,3},5.0};auto r=reduceSamples(s);std::cout<<r.value<<" "<<r.calls<<"\\n";}`);
  const compiler=process.platform==='win32'?(isNvidia()??'g++'):(process.env.CXX??'g++');
  const build=exec(compiler,['-std=c++17','-O2',testCpp,'-o',testExe]);
  check(build.status===0,'Closure CPU compile failed: '+String(build.stdout??'')+' '+String(build.stderr??''));
  const run=exec(testExe,[]);
  check(run.status===0&&String(run.stdout??'').trim()==='9 5','Closure mutation mismatch: '+String(run.stdout??'')+' '+String(run.stderr??''));
  add('G7f0: generic closure conversion, write-through and CUDA-safe capture syntax','PASS');
 }catch(e){add('G7f0: generic closure conversion, write-through and CUDA-safe capture syntax','FAIL',{error:String(e).slice(0,5000)});}
 const ref=compileGenericFullSource(source,sourcePath);fixtures=makeCases();baseline=fixtures.map(f=>ref.run(f.entry,f.input));
 add('G7c: independent trusted TypeScript reference','PASS',{cases:fixtures.length});
 const cpu=path.join(tmp,'structured.cpp'),exe=path.join(tmp,process.platform==='win32'?'structured.exe':'structured');
 writeFileSync(cpu,generateCppHarness(compiled,fixtures,'cpu'));
 const cc=process.platform==='win32'&&isNvidia()?isNvidia():process.env.CXX||'g++';
 const ccArgs=cc.toLowerCase().includes('nvcc')?['-std=c++17','-O3',cpu,'-o',exe]:['-std=c++17','-O3',cpu,'-o',exe];
 const built=exec(cc,ccArgs);
 if(built.status!==0){add('G7d: structured native C++ compilation','FAIL',{compiler:cc,error:(built.stderr||built.error?.message||'').slice(0,4500)});}
 else{
  add('G7d: structured native C++ compilation','PASS',{compiler:cc});
  const run=exec(exe,[],{timeout:120000});
  if(run.status!==0)add('G7e: structured native C++ parity','FAIL',{error:(run.stderr||run.error?.message||'').slice(0,2000)});
  else {checkParity(compiled,fixtures,parseRows(run.stdout),'G7e: structured native C++ parity',baseline); report.cpuBenchmarks=run.stderr.split(/\r?\n/).filter(x=>x.startsWith('BENCH '));}
 }
  const generatedCuda=generateCppHarness(compiled,fixtures,'gpu');
  // G7f1: prevent device fmod(double,int) resolving to host-only overload.
  const unsafeModulus=/fmod\(arg\.currentIndex\+shift\+j,\s*arg\.[A-Za-z_][A-Za-z0-9_]*\.length\)/.test(generatedCuda);
  check(!unsafeModulus,'G7f1: device modulo still uses mixed double/int operands');
  check(/fmod\(arg\.currentIndex\+shift\+j,\s*double\(arg\.[A-Za-z_][A-Za-z0-9_]*\.length\)\)/.test(generatedCuda),'G7f1: device modulus cast missing');
  add('G7f1: device modulo operands lowered to double/double','PASS');
  const nvcc=isNvidia();
 if(!nvcc)add('G7f: structured CUDA compilation and GPU parity','SKIP',{reason:'CUDA compiler not available in this environment'});
 else{
  const cuda=path.join(tmp,'structured.cu'),cudaExe=path.join(tmp,process.platform==='win32'?'structured-gpu.exe':'structured-gpu');
  writeFileSync(cuda,generatedCuda);
  const g=exec(nvcc,['--std=c++17','-O3','--expt-extended-lambda',`-arch=${arch}`,cuda,'-o',cudaExe],{timeout:300000});
  if(g.status!==0){
   const diagnostics=processDiagnostics(g,nvcc,['--std=c++17','-O3','--expt-extended-lambda',`-arch=${arch}`,cuda,'-o',cudaExe]);
   const saved=path.join(outputDir,'approach-a-structured-failed-generated.cu');
   writeFileSync(saved,readFileSync(cuda));
   add('G7f: structured CUDA compilation and GPU parity','FAIL',{phase:'compile',diagnostics,generatedCudaFile:saved});
  }
  else{
   const r=exec(cudaExe,[],{timeout:180000});
   if(r.status!==0)add('G7f: structured CUDA compilation and GPU parity','FAIL',{phase:'run',diagnostics:processDiagnostics(r,cudaExe,[])});
   else checkParity(compiled,fixtures,parseRows(r.stdout),'G7f: structured CUDA compilation and GPU parity',baseline);
   report.gpuBenchmarks=String(r.stderr??'').split(/\r?\n/).filter(x=>x.startsWith('B '));
  }
 }
 compareOriginalA(findOldNative(),source,fixtures,baseline);
} catch(e){add('Structured compilation test runner','FAIL',{error:String(e.stack??e).slice(0,6000)});}
finally{
 report.finishedAt=new Date().toISOString();
 report.readyForProduction=false; // Annual 28-tank parity and same-workload benchmark not implemented.
 report.reason='Research gate only: no 28-tank annual parity or fair end-to-end timing comparison.';
 const filename=`approach-a-structured-verify-${report.at.replace(/[-:.]/g,'')}.json`;
 const location=path.join(outputDir,filename);
 writeFileSync(location,JSON.stringify(report,null,2));
 console.log(`REPORT ${location}`);console.log('PRODUCTION_GATE BLOCK — productive button A unchanged');
 rmSync(tmp,{recursive:true,force:true});
}
process.exitCode=report.stages.some(s=>s.status==='FAIL') || (requireGpu&&!report.stages.some(s=>s.name.startsWith('G7f:')&&s.status==='PASS')) || (requireOriginalA&&!report.stages.some(s=>s.name==='Original A1+A2 native benchmark'&&s.status==='PASS')) ? 1 : 0;
