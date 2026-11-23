/*
 * SunOpt Approach A — generic GPU dispatcher A8 research benchmark.
 * Strict: the compiler reads f(x) from TypeScript; CUDA glue only schedules
 * arbitrary independent typed-struct jobs. No application physics in glue.
 * Three modes on the exact same 2400 independent dynamic input records:
 *   200 launches x 12 jobs; 1 launch x 2400 jobs with 64/128/256 threads.
 * Independent snapshots are NOT the stateful 28-tank annual controller.
 * Does NOT modify production code or native addons.
 */
import { readFileSync, existsSync, mkdirSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs';
import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import os from 'node:os';
import { performance } from 'node:perf_hooks';
import { compileStructuredSource } from './approach-a-structured-compiler.mjs';
import { compileGenericFullSource } from './approach-a-generic-full-source.mjs';
import { findSunOptProjectRoot } from './approach-a-generic-typescript.mjs';
const root = findSunOptProjectRoot(import.meta.url) ?? process.cwd();
const sourceFile = path.join(root, 'src/simulation/rust-acceleration-scopes.ts');
const outbox = path.join(root, 'updates/outbox');
mkdirSync(outbox, {recursive:true});
const now = new Date().toISOString(), suffix=now.replace(/[-:.]/g,'');
const reportFile = path.join(outbox,`approach-a-generic-dispatch-${suffix}.json`);
const report={schemaVersion:1, name:'Approach A generic independent GPU dispatch benchmark', createdAt:now,
 root, sourceSha256:null, gpuRequired:process.argv.includes('--require-gpu'),
 comparisons:[], passes:[], failures:[], stage:'INCOMPLETE', readyForProduction:false,
 annualSimulation:'NOT_EXECUTED; independent dynamic snapshots are not stateful annual controller'};
const fail=e=>{report.failures.push(String(e?.stack??e).slice(0,8000));console.error('FAIL:',String(e));};
const assert=(yes,what)=>{if(!yes)throw Error(what);};
const run=(command,args,timeout=180000)=>spawnSync(command,args,{encoding:'utf8',timeout,maxBuffer:40*1024*1024});
const arch=process.argv.find(a=>a.startsWith('--arch='))?.slice(7)??'sm_120';
if(!/^sm_\d{2,3}[a-z]?$/.test(arch))throw Error('Invalid CUDA arch');
function nvcc(){const c=process.platform==='win32'&&process.env.CUDA_PATH?path.join(process.env.CUDA_PATH,'bin','nvcc.exe'):'nvcc';return run(c,['--version'],15000).status===0?c:null;}
function fixture(seed){
 const N=192,pv=new Float64Array(N),load=new Float64Array(N),mode=seed%4;
 for(let i=0;i<N;i++){
  const sun=Math.max(0,9*Math.sin((i+seed*17)/18.6));
  pv[i]=mode===0?0:(mode===1?1.2:sun+((i*29+seed*13)%7)*0.12);
  load[i]=mode===0?0:(mode===1?.15:((i*7+seed*11)%13===0?3.5:.12));
 }
 const common={timeStepHours:.25,targetStorageEnergy:25+seed*2.5,storageEnergyPerKelvin:1.5+seed*.1,
  storageConductanceWPerKelvin:2.1+seed*.08,storageMinTemperature:60,ambientTemperature:18+seed,
  dailyHeatLoss:seed%3===0?0:seed%3===1?1.1:-1};
 const idx=(seed*31)%N;
 return {
  runRustNoSunshineScope:{...common,pvPower:pv,loadEnergy:load,currentIndex:idx,horizonSteps:48+seed*5,
   protectedSteps:Math.min(24,48+seed*5),householdReservePower:.6+seed*.04,
   maximumHeatPumpElectricalPower:1.2+seed*.05,cop:3,initialWaterlinePower:.1,currentStorageEnergy:seed*2},
  runRustPeakShavingCausalScope:{...common,exportPower:pv,demandEnergy:load,currentIndex:idx,
   horizonSteps:48+seed*5,currentStorageEnergy:seed*2,minimumReserveEnergy:Math.min(5+seed,common.targetStorageEnergy*.5),
   maximumHeatPumpElectricalPower:1.2+seed*.05,cop:3,energyCutLevel:.4+seed*.12}
 };
}
const names=['runRustNoSunshineScope','runRustPeakShavingCausalScope']; // benchmark selector, not compiler lowering
const num=v=>Number.isInteger(v)?String(v)+'.0':String(v);
function initStruct(program,typename,input,fixtureIdx,registry){
 const meta=program.interfaces[typename];assert(meta,'No struct metadata '+typename);
 const fields=[];
 for(const base of meta.bases)fields.push(initStruct(program,base,input,fixtureIdx,registry));
 for(const [field,kind] of Object.entries(meta.ownFields)){
  const value=input[field];
  if(kind==='a_span'){
   assert(value instanceof Float64Array,'Invalid array '+field);
   const key=`arr_${fixtureIdx}_${field}`;
   if(!registry.has(key))registry.set(key,value);
   fields.push(`a_span{d_${key}, ${value.length}}`);
  } else if(kind==='bool')fields.push(value?'true':'false');
  else {assert(typeof value==='number' && Number.isFinite(value),'Invalid '+field);fields.push(num(value));}
 }
 return `${typename}{${fields.join(', ')}}`;
}
function generateCuda(program,fixtures,N=2400){
 const registry=new Map(),entries=[];
 for(const name of names){
  const f=program.functions.find(x=>x.name===name);
  assert(f?.args?.length===1 && f.args[0].type in program.interfaces,`Entry signature ${name}`);
  const cases=fixtures.map((base,index)=>initStruct(program,f.args[0].type,base[name],index,registry));
  const lengthField=Object.entries(program.structs[f.args[0].type]).find(([,type])=>type==='a_span')?.[0];
  assert(lengthField,'No array for dynamic input index');
  entries.push({name,f,cases,lengthField,fields:Object.keys(program.structs[f.result])});
 }
 let out=`#include <cuda_runtime.h>\n#include <iostream>\n#include <iomanip>\n#include <vector>\n#include <chrono>\n#include <cmath>\n${program.cpp}\n`;
 out+=`#define CU_CHECK(e) do{cudaError_t a_err=(e); if(a_err!=cudaSuccess){std::cerr<<"CUDA_ERROR "<<__LINE__<<" "<<cudaGetErrorString(a_err)<<"\\n";return 3;}}while(0)\n`;
 out+=`#define CU_KERNEL_CHECK() CU_CHECK(cudaGetLastError())\n`;
 for(const [k,array] of registry)out+=`static double h_${k}[] = {${[...array].map(num).join(',')}};\n`;
 for(const {name,f} of entries){
  out+=`__global__ void D_${name}(const ${f.args[0].type}* in, ${f.result}* out, int first, int count) {int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<count)out[first+i]=${name}(in[first+i]);}\n`;
 }
 out+=`int main(){std::cout<<std::setprecision(17)<<std::boolalpha;\n`;
 for(const [k,array] of registry){out+=`double *d_${k}=nullptr; CU_CHECK(cudaMalloc((void**)&d_${k},sizeof(h_${k}))); CU_CHECK(cudaMemcpy(d_${k},h_${k},sizeof(h_${k}),cudaMemcpyHostToDevice));\n`;}
 for(const {name,f,cases,lengthField,fields} of entries){
  out+=`{\nconst int N=${N};std::vector<${f.args[0].type}> host;host.reserve(N);\n`;
  out+=`std::vector<${f.args[0].type}> templates={${cases.join(',')}};\n`;
  out+=`for(int k=0;k<N;k++){auto v=templates[k%templates.size()];v.currentIndex=fmod(v.currentIndex+double(k),double(v.${lengthField}.length));host.push_back(v);}\n`;
  out+=`${f.args[0].type}* dInputs=nullptr;${f.result}* dOutputs=nullptr;\n`;
  out+=`CU_CHECK(cudaMalloc((void**)&dInputs,sizeof(host[0])*N));CU_CHECK(cudaMalloc((void**)&dOutputs,sizeof(${f.result})*N));\n`;
  out+=`auto transferBegin=std::chrono::steady_clock::now();CU_CHECK(cudaMemcpy(dInputs,host.data(),sizeof(host[0])*N,cudaMemcpyHostToDevice));auto transferEnd=std::chrono::steady_clock::now();\n`;
  out+=`std::cerr<<"TRANSFER_H2D ${name} "<<std::chrono::duration<double,std::milli>(transferEnd-transferBegin).count()<<"\\n";\n`;
  // One warmup for GPU clock, excludes compilation and allocations from hot path.
  out+=`D_${name}<<<(N+127)/128,128>>>(dInputs,dOutputs,0,N);CU_KERNEL_CHECK();CU_CHECK(cudaDeviceSynchronize());\n`;
  out+=`for(int mode=0;mode<4;mode++){\nconst int bs=mode==0?64:(mode==1?64:(mode==2?128:256));\n`;
  out+=`cudaEvent_t start,stop;CU_CHECK(cudaEventCreate(&start));CU_CHECK(cudaEventCreate(&stop));\n`;
  out+=`auto wallBegin=std::chrono::steady_clock::now();CU_CHECK(cudaEventRecord(start));\n`;
  out+=`if(mode==0){for(int offset=0;offset<N;offset+=12)D_${name}<<<1,64>>>(dInputs,dOutputs,offset,12);}\n`;
  out+=`else{D_${name}<<<(N+bs-1)/bs,bs>>>(dInputs,dOutputs,0,N);}\n`;
  out+=`CU_KERNEL_CHECK();CU_CHECK(cudaEventRecord(stop));CU_CHECK(cudaEventSynchronize(stop));\n`;
  out+=`auto wallEnd=std::chrono::steady_clock::now();float deviceMs=0;CU_CHECK(cudaEventElapsedTime(&deviceMs,start,stop));\n`;
  out+=`std::vector<${f.result}> results(N);auto d2hBegin=std::chrono::steady_clock::now();CU_CHECK(cudaMemcpy(results.data(),dOutputs,sizeof(${f.result})*N,cudaMemcpyDeviceToHost));auto d2hEnd=std::chrono::steady_clock::now();\n`;
  out+=`std::cerr<<"TIMING ${name} "<<mode<<" "<<std::chrono::duration<double,std::milli>(wallEnd-wallBegin).count()<<" "<<deviceMs<<" "<<std::chrono::duration<double,std::milli>(d2hEnd-d2hBegin).count()<<"\\n";\n`;
  // Only bulk values from each mode, independent of the old original native code.
  out+=`for(int k=0;k<N;k++){auto r=results[k];std::cout<<"C ${name} "<<mode<<" "<<k<<" "<<${fields.map(p=>`r.${p}`).join('<<" "<<')}<<"\\n";}\n`;
  out+=`CU_CHECK(cudaEventDestroy(start));CU_CHECK(cudaEventDestroy(stop));}\nCU_CHECK(cudaFree(dInputs));CU_CHECK(cudaFree(dOutputs));}\n`;
 }
 for(const k of registry.keys())out+=`CU_CHECK(cudaFree(d_${k}));\n`;
 out+='return 0;}';
 return {out,entries,registryCount:registry.size};
}
function parseGPU(program,fixtures,text){
 const rows=new Map();
 for(const l of text.split(/\r?\n/)){
  if(!l.startsWith('C '))continue;
  const parts=l.trim().split(/\s+/);
  const [name,mode,k]=[parts[1],Number(parts[2]),Number(parts[3])];
  const vals=parts.slice(4).map(v=>v==='true'?true:v==='false'?false:Number(v));
  if(!names.includes(name)||!Number.isInteger(mode)||!Number.isInteger(k))throw Error(`Bad result row ${l.slice(0,80)}`);
  rows.set(`${name}:${mode}:${k}`,vals);
 }
 return rows;
}
function originalNative(){const p=path.join(root,'rust/sunopt-node/sunopt_node.win32-x64-msvc.node');
 if(!existsSync(p))return null;return createRequire(path.join(root,'package.json'))(p);
}
const approx=(a,b)=>a===b||(Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=1e-10*Math.max(1,Math.abs(a),Math.abs(b)));
function nativeBench(binding,source,fixtures,ref,N){
 if(!binding)return {status:'SKIP',reason:'No original sunopt-node binding available'};
 const result={status:'PASS',comparisons:[]};
 for(const name of names){
  const scope=name==='runRustNoSunshineScope'?'rust_noSunshine':'rust_peakShavingCausal';
  const method=name==='runRustNoSunshineScope'?'runNoSunshineKernelContext':'runPeakShavingCausalKernelContext';
  const handle=binding.registerAccelerationScope(source,scope);
  const contexts=fixtures.map(base=>{
   const input=base[name];
   const arrays=name==='runRustNoSunshineScope'?[input.pvPower,input.loadEnergy]:[input.exportPower,input.demandEnergy];
   const data=binding.registerAccelerationDataPair(...arrays);
   return {input,params:Object.fromEntries(Object.entries(input).filter(([,v])=>!(v instanceof Float64Array))),data,length:arrays[0].length};
  });
  const initial=contexts[0],first=binding[method](handle,initial.data,initial.params);
  const errors=[];
  for(const [field,value] of Object.entries(ref[name][0]))if(!approx(first[field],value))errors.push({index:0,field,original:first[field],reference:value});
  assert(!errors.length,'Original native parity mismatch '+JSON.stringify(errors));
  // A1/A2: registered handles and two persistent Float64Arrays for every fixture.
  const calls=contexts.map((c,i)=>({c,params:{...c.params}}));
  const begin=performance.now();
  let checksum=0;
  for(let k=0;k<N;k++){
   const z=calls[k%calls.length];z.params.currentIndex=(z.c.input.currentIndex+k)%z.c.length;
   const got=binding[method](handle,z.c.data,z.params);
   checksum+=Number(Object.values(got)[0]);
  }
  const wallMs=performance.now()-begin;
  result.comparisons.push({entry:name,calls:N,wallMs,checksum});
 }
 return result;
}
function summarizeGpu(program,fixtures,raw,ref,N){
 const rows=parseGPU(program,fixtures,raw.stdout),errors=[],timing=[];
 const totalExpected=names.length*4*N;
 assert(rows.size===totalExpected,`GPU rows missing/duplicate: ${rows.size} expected ${totalExpected}`);
 for(const name of names){
  const f=program.functions.find(x=>x.name===name),fields=Object.keys(program.structs[f.result]);
  for(let mode=0;mode<4;mode++)for(let k=0;k<N;k++){
   const vals=rows.get(`${name}:${mode}:${k}`),reference=ref[name][k];
   if(!vals||vals.length!==fields.length){errors.push({name,mode,k,reason:'fields'});continue;}
   for(let v=0;v<fields.length;v++)if(!approx(vals[v],reference[fields[v]])){
    if(errors.length<16)errors.push({name,mode,k,field:fields[v],gpu:vals[v],reference:reference[fields[v]]});
   }
  }
 }
 for(const line of raw.stderr.split(/\r?\n/)){
  const a=line.trim().split(/\s+/);
  if(a[0]==='TIMING')timing.push({entry:a[1],mode:Number(a[2]),wallMs:Number(a[3]),cudaEventMs:Number(a[4]),downloadMs:Number(a[5])});
  if(a[0]==='TRANSFER_H2D')report.passes.push({name:'A2 dynamic GPU input transfer',entry:a[1],transferMs:Number(a[2])});
 }
 assert(timing.length===8,`Expected 8 measured configurations, got ${timing.length}: ${raw.stderr.slice(-2000)}`);
 report.gpu={status:errors.length?'FAIL':'PASS',jobsPerFunction:N,staticArrayCount:fixtures.length*2*names.length,
  launchModes:{0:'200 x 12 jobs / 1 block',1:'1 x 2400 jobs / block 64',2:'1 x 2400 jobs / block 128',3:'1 x 2400 jobs / block 256'},
  timing,failures:errors};
 assert(!errors.length,`GPU parity mismatch: ${JSON.stringify(errors.slice(0,3))}`);
 report.passes.push({name:'Exact workload parity across all 4 GPU launch shapes',cases:totalExpected,status:'PASS'});
}
const tmp=mkdtempSync(path.join(os.tmpdir(),'sunopt-a-dispatch-'));
try{
 assert(existsSync(sourceFile),'Source file missing '+sourceFile);
 const source=readFileSync(sourceFile,'utf8');
 const program=compileStructuredSource(source,{filename:sourceFile}); report.sourceSha256=program.hash;
 const again=compileStructuredSource(source,{filename:sourceFile});assert(again===program,'A1 compiled object not reused');
 const fixt=Array.from({length:12},(_,i)=>fixture(i)),N=2400;
 const compiledRef=compileGenericFullSource(source,sourceFile);
 const reference={};
 for(const name of names){
  reference[name]=[];
  for(let k=0;k<N;k++){
   const input={...fixt[k%fixt.length][name]};
   const arr=name==='runRustNoSunshineScope'?input.pvPower:input.exportPower;
   input.currentIndex=(input.currentIndex+k)%arr.length;
   reference[name].push(compiledRef.run(name,input));
  }
 }
 report.passes.push({name:'A1 generic source compilation cached once; TS reference independent',status:'PASS',functions:names,inputsPerFunction:N});
 try {report.originalA1A2=nativeBench(originalNative(),source,fixt,reference,N);}catch(e){report.originalA1A2={status:'FAIL',error:String(e)};fail(e);}
 const generated=generateCuda(program,fixt,N);
 if(process.argv.includes('--emit-cuda'))writeFileSync(path.join(outbox,`approach-a-generic-dispatch-source-${suffix}.cu`),generated.out);
 assert(!generated.out.includes('D_B14P'),'Unexpected domain-specific dispatch');
 assert(generated.out.includes('dInputs')&&generated.out.includes('dOutputs')&&generated.out.includes('for(int mode=0;mode<4;mode++)'),'Generated multi-grid dispatcher incomplete');
 report.passes.push({name:'A8 four CUDA dispatcher shapes emitted',status:'PASS',sourceBytes:generated.out.length,staticBufferPairs:generated.registryCount/2});
 const cudaFile=path.join(tmp,'generic-dispatch.cu'),exe=path.join(tmp,process.platform==='win32'?'generic-dispatch.exe':'generic-dispatch');
 writeFileSync(cudaFile,generated.out);
 const cc=nvcc();
 if(!cc){report.gpu={status:'SKIP',reason:'CUDA compiler unavailable on this machine'};if(report.gpuRequired)fail('CUDA required but nvcc unavailable');}
 else{
  const flags=['--std=c++17','-O3','--expt-extended-lambda',`-arch=${arch}`,cudaFile,'-o',exe];
  const build=run(cc,flags,300000);
  if(build.status!==0){
   report.gpu={status:'FAIL',phase:'compile',exitCode:build.status,stdout:String(build.stdout??'').slice(-12000),stderr:String(build.stderr??'').slice(-12000)};
   const debug=path.join(outbox,`approach-a-generic-dispatch-failed-${suffix}.cu`);writeFileSync(debug,generated.out);report.gpu.generatedCudaFile=debug;
   fail('CUDA compile failed: '+String(build.stdout??'').slice(-2000)+' '+String(build.stderr??'').slice(-2000));
  }else{
   const result=run(exe,[],300000);
   if(result.status!==0){report.gpu={status:'FAIL',phase:'execute',exitCode:result.status,stdout:String(result.stdout??'').slice(-1500),stderr:String(result.stderr??'').slice(-6000)};fail('CUDA execution failed: '+String(result.stderr).slice(-1500));}
   else summarizeGpu(program,fixt,result,reference,N);
  }
 }
 if(report.gpu?.status==='PASS' && report.originalA1A2?.status==='PASS'){
  report.comparisons=names.map(name=>{
   const old=report.originalA1A2.comparisons.find(x=>x.entry===name);
   const measures=report.gpu.timing.filter(x=>x.entry===name);
   const micro=measures.find(x=>x.mode===0),bulk=measures.filter(x=>x.mode>0).sort((a,b)=>a.wallMs-b.wallMs)[0];
   return {name,calls:N,originalA1A2WallMs:old.wallMs,genericMicrobatchWallMs:micro.wallMs,
    bestParallelWallMs:bulk.wallMs,bestParallelThreads:([null,64,128,256])[bulk.mode],
    parallelVersusMicrobatch:micro.wallMs/bulk.wallMs,
    parallelVersusOriginalA1A2:old.wallMs/bulk.wallMs,
    comparableAsEndToEnd:false,
    note:'Identical 2400 independent input snapshots, but distinct CPU N-API vs GPU dispatch architectures. GPU excludes A1 compile/static setup and annual scheduling; no annual speedup inferred.'};
  });
 }
 report.stage=report.failures.length||report.gpu?.status==='FAIL'||report.originalA1A2?.status==='FAIL'?'FAIL':report.gpu?.status==='PASS'?'BENCHMARK_PASS_NO_ANNUAL':'LOCAL_SMOKE_ONLY';
}catch(e){fail(e);report.stage='FAIL';}
finally{
 report.finishedAt=new Date().toISOString();
 writeFileSync(reportFile,JSON.stringify(report,null,2)+'\n');
 console.log('REPORT '+reportFile);console.log('STATUS '+report.stage);console.log('ANNUAL_GATE BLOCK — this is independent snapshot replay, not the annual stateful controller.');
 rmSync(tmp,{recursive:true,force:true});
}
process.exitCode=report.stage==='FAIL'?1:report.gpuRequired&&report.gpu?.status!=='PASS'?1:process.argv.includes('--require-original-a')&&report.originalA1A2?.status!=='PASS'?1:0;
