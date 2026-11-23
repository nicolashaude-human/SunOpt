/*
 * Approach A9 research only: compile TWO separate TypeScript source files into
 * one generic CUDA/CPU translation unit, then build a persistent lockstep
 * sidecar. No application physics is embedded in the TypeScript AST compiler.
 * The native glue maps the existing stride-12 / stride-3 controller ABI.
 * Building does not modify production code or the active A/B solver.
 */
import {readFileSync,writeFileSync,mkdirSync,existsSync} from 'node:fs';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {compileStructuredSource} from './approach-a-structured-compiler.mjs';
import {findSunOptProjectRoot} from './approach-a-generic-typescript.mjs';

const root=findSunOptProjectRoot(import.meta.url)??process.cwd();
const sourcePath=path.join(root,'src/simulation/rust-acceleration-scopes.ts');
const energyPath=path.join(root,'src/simulation/approach-a9-energy-scopes.ts');
const outbox=path.join(root,'updates/outbox');
const kind=process.argv.includes('--cpu')?'cpu':'cuda';
const generateOnly=process.argv.includes('--generate-only');
const profile=process.argv.includes('--profile');
const arch=process.argv.find(x=>x.startsWith('--arch='))?.slice(7)??'sm_120';
if(!/^sm_\d{2,3}[a-z]?$/.test(arch))throw Error('Unsafe CUDA architecture');
if(!existsSync(sourcePath)||!existsSync(energyPath))throw Error('A9 source files missing');
mkdirSync(outbox,{recursive:true});
const source=readFileSync(sourcePath,'utf8')+'\n\n'+readFileSync(energyPath,'utf8');
const compiled=compileStructuredSource(source,{filename:'approach-a9-combined.ts'});
if(compileStructuredSource(source,{filename:'approach-a9-combined.ts'})!==compiled)throw Error('A1 codegen cache not reused');
const requirements=['runA9EnergyCutLevel','runRustPeakShavingCausalScope'];
for(const name of requirements)if(!compiled.functions.some(f=>f.name===name))throw Error(`Missing source function ${name}`);
const digest=createHash('sha256').update(source).digest('hex');
const sourceFile=path.join(outbox,`approach-a9-generated-${digest.slice(0,12)}.${kind==='cpu'?'cpp':'cu'}`);
const outputFile=path.join(outbox,`approach-a9-generated-${digest.slice(0,12)}${process.platform==='win32'?'.exe':'.bin'}`);
const buildReport=path.join(outbox,`approach-a9-build-${kind}.json`);
// The driver is intentionally only an ABI adapter/dispatcher. The formulas
// are called from the source-derived compiled functions, not reproduced here.
const driver=String.raw`
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif
#ifdef A9_CUDA
#include <cuda_runtime.h>
#endif

${compiled.cpp}

// Existing SunOpt packed-parameter ABI, not a hardcoded f(x) implementation.
// Stride 12: current, reserve, target, maxHP, COP, requested, dt,
//            energyPerKelvin, conductance, minTemp, ambientTemp, dailyLoss.
// Stride 3 dynamic: currentEnergy, minimumReserve, requestedElectricEnergy.
A_HD static inline double a9_dispatch_one(
 const double* exportPower, const double* demandEnergy, int length,
 const double* constants, const double* dynamic,
 int currentIndex, int horizonSteps
) {
 const double targetStorageEnergy=constants[2];
 const double maxHpPower=constants[3];
 const double cop=constants[4];
 const double dt=constants[6];
 const StorageKernelParameters storage{
   dt, targetStorageEnergy, constants[7], constants[8],
   constants[9], constants[10], constants[11]
 };
 const A9EnergySearchInput energyInput{
   a_span{exportPower,length}, double(currentIndex), double(horizonSteps),
   dynamic[2], maxHpPower, dt
 };
 const double energyCut = runA9EnergyCutLevel(energyInput);
 const RustPeakShavingCausalScopeInput causalInput{
   storage,
   a_span{exportPower,length}, a_span{demandEnergy,length},
   double(currentIndex), double(horizonSteps), dynamic[0], dynamic[1],
   maxHpPower, cop, energyCut
 };
 return runRustPeakShavingCausalScope(causalInput).cutLevelPower;
}

#ifdef A9_CUDA
#define A9_CHECK(expr) do {cudaError_t er=(expr); if(er!=cudaSuccess){ \
 fprintf(stderr,"A9_CUDA_ERROR line=%d %s\n",__LINE__,cudaGetErrorString(er));fflush(stderr);exit(8);}}while(false)
__global__ void a9_lockstep_kernel(
 const double* exportPower, const double* demandEnergy,
 int length, const double* packed, const double* dynamic,
 int currentIndex, int horizonSteps, int tanks, double* out
) {
 const int tank=blockIdx.x;
 if(tank>=tanks||threadIdx.x!=0)return;
 out[tank]=a9_dispatch_one(exportPower,demandEnergy,length,
   packed+12*tank,dynamic+3*tank,currentIndex,horizonSteps);
}
#endif

static bool read_exact(void* p,size_t size){return fread(p,1,size,stdin)==size;}
static void write_exact(const void* p,size_t size){
 if(fwrite(p,1,size,stdout)!=size || fflush(stdout)!=0){
  fprintf(stderr,"A9_PIPE_WRITE_ERROR\n");exit(12);
 }
}
struct A9Runtime {
 int length=0,tanks=0;
 std::vector<double> hp, demand, packed, dynamic, cuts;
#ifdef A9_CUDA
 double *d_hp=nullptr,*d_demand=nullptr,*d_packed=nullptr,*d_dynamic=nullptr,*d_cuts=nullptr;
 cudaStream_t stream=nullptr;
#ifdef A9_PROFILE
 cudaEvent_t e0=nullptr,e1=nullptr,e2=nullptr,e3=nullptr;
 unsigned long long profCalls=0;
 double profH2d=0,profKernel=0,profD2h=0,profStepWall=0;
#endif
 void init_gpu(){
  A9_CHECK(cudaSetDevice(0));
  A9_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  A9_CHECK(cudaMalloc((void**)&d_hp,hp.size()*sizeof(double)));
  A9_CHECK(cudaMalloc((void**)&d_demand,demand.size()*sizeof(double)));
  A9_CHECK(cudaMalloc((void**)&d_packed,packed.size()*sizeof(double)));
  A9_CHECK(cudaMalloc((void**)&d_dynamic,dynamic.size()*sizeof(double)));
  A9_CHECK(cudaMalloc((void**)&d_cuts,cuts.size()*sizeof(double)));
  A9_CHECK(cudaMemcpyAsync(d_hp,hp.data(),hp.size()*sizeof(double),cudaMemcpyHostToDevice,stream));
  A9_CHECK(cudaMemcpyAsync(d_demand,demand.data(),demand.size()*sizeof(double),cudaMemcpyHostToDevice,stream));
  A9_CHECK(cudaMemcpyAsync(d_packed,packed.data(),packed.size()*sizeof(double),cudaMemcpyHostToDevice,stream));
  A9_CHECK(cudaStreamSynchronize(stream));
#ifdef A9_PROFILE
  A9_CHECK(cudaEventCreate(&e0));
  A9_CHECK(cudaEventCreate(&e1));
  A9_CHECK(cudaEventCreate(&e2));
  A9_CHECK(cudaEventCreate(&e3));
#endif
 }
 ~A9Runtime(){
  if(d_hp)cudaFree(d_hp);if(d_demand)cudaFree(d_demand);
  if(d_packed)cudaFree(d_packed);if(d_dynamic)cudaFree(d_dynamic);
  if(d_cuts)cudaFree(d_cuts);
#ifdef A9_PROFILE
  if(e0)cudaEventDestroy(e0);
  if(e1)cudaEventDestroy(e1);
  if(e2)cudaEventDestroy(e2);
  if(e3)cudaEventDestroy(e3);
#endif
  if(stream)cudaStreamDestroy(stream);
 }
#endif
 void step(int currentIndex,int horizonSteps){
#ifdef A9_CUDA
#ifdef A9_PROFILE
  const auto t0=std::chrono::steady_clock::now();
  A9_CHECK(cudaEventRecord(e0,stream));
#endif
  A9_CHECK(cudaMemcpyAsync(d_dynamic,dynamic.data(),dynamic.size()*sizeof(double),cudaMemcpyHostToDevice,stream));
#ifdef A9_PROFILE
  A9_CHECK(cudaEventRecord(e1,stream));
#endif
  // 1 block per tank: distinct tank states execute on distinct SMs.
  // Time steps remain SERIAL and no independent-snapshot replay is used.
  a9_lockstep_kernel<<<tanks,32,0,stream>>>(d_hp,d_demand,length,d_packed,
     d_dynamic,currentIndex,horizonSteps,tanks,d_cuts);
  A9_CHECK(cudaGetLastError());
#ifdef A9_PROFILE
  A9_CHECK(cudaEventRecord(e2,stream));
#endif
  A9_CHECK(cudaMemcpyAsync(cuts.data(),d_cuts,cuts.size()*sizeof(double),cudaMemcpyDeviceToHost,stream));
#ifdef A9_PROFILE
  A9_CHECK(cudaEventRecord(e3,stream));
#endif
  A9_CHECK(cudaStreamSynchronize(stream));
#ifdef A9_PROFILE
  float h2d=0,compute=0,d2h=0;
  A9_CHECK(cudaEventElapsedTime(&h2d,e0,e1));
  A9_CHECK(cudaEventElapsedTime(&compute,e1,e2));
  A9_CHECK(cudaEventElapsedTime(&d2h,e2,e3));
  profH2d+=h2d;profKernel+=compute;profD2h+=d2h;
  profStepWall+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
  ++profCalls;
  if(profCalls==1||profCalls%5000==0){
   fprintf(stderr,"A9_DEVICE_PROFILE calls=%llu h2dMs=%.3f kernelMs=%.3f d2hMs=%.3f stepWallMs=%.3f\n",
     profCalls,profH2d,profKernel,profD2h,profStepWall);
   fflush(stderr);
  }
#endif
#else
  for(int tank=0;tank<tanks;tank++)cuts[tank]=a9_dispatch_one(
      hp.data(),demand.data(),length,packed.data()+12*tank,
      dynamic.data()+3*tank,currentIndex,horizonSteps);
#endif
 }
};
int main(){
#ifdef _WIN32
 _setmode(_fileno(stdin),_O_BINARY);_setmode(_fileno(stdout),_O_BINARY);
#endif
 A9Runtime context;
 unsigned long long stepCount=0;
 while(true){
  int32_t op=0;
  if(!read_exact(&op,sizeof(op)))return 0;
  if(op==1){
   int32_t n=0,tanks=0;
   if(!read_exact(&n,4)||!read_exact(&tanks,4)||n<1||n>40000||tanks<1||tanks>128){
    fprintf(stderr,"A9_INVALID_INITIAL_HEADER\n");return 2;
   }
   if(context.length){fprintf(stderr,"A9_DUPLICATE_INIT\n");return 2;}
   context.length=n;context.tanks=tanks;
   context.hp.resize(n);context.demand.resize(n);context.packed.resize(tanks*12);
   context.dynamic.resize(tanks*3);context.cuts.resize(tanks);
   if(!read_exact(context.hp.data(),n*8ull)||!read_exact(context.demand.data(),n*8ull)||
      !read_exact(context.packed.data(),tanks*12ull*8ull)){
     fprintf(stderr,"A9_INCOMPLETE_CONFIG\n");return 2;
   }
#ifdef A9_CUDA
   context.init_gpu();
#endif
   const char ready='R';write_exact(&ready,1);
   fprintf(stderr,"A9_INIT_OK source=${digest} tanks=%d annualValues=%d backend=${kind}\n",tanks,n);fflush(stderr);
  }else if(op==2){
   int32_t index=0,horizon=0;
   if(!context.length||!read_exact(&index,4)||!read_exact(&horizon,4)||
      index<0||index>=context.length||horizon<1||horizon>context.length||
      !read_exact(context.dynamic.data(),context.tanks*3ull*8ull)){
     fprintf(stderr,"A9_INVALID_STEP_INPUT\n");return 3;
   }
   context.step(index,horizon);
   for(double cut:context.cuts)if(!std::isfinite(cut)){
     fprintf(stderr,"A9_NONFINITE_RESULT at step=%llu\n",stepCount);return 4;
   }
   write_exact(context.cuts.data(),context.tanks*8ull);
   ++stepCount;
  }else if(op==3){
   fprintf(stderr,"A9_FINISHED steps=%llu\n",stepCount);fflush(stderr);return 0;
  }else{
   fprintf(stderr,"A9_BAD_COMMAND %d\n",op);return 5;
  }
 }
}
`;
const prefix=(kind==='cuda'?'#define A9_CUDA 1\n':'')+(profile?'#define A9_PROFILE 1\n':'');
writeFileSync(sourceFile,prefix+driver);
const command=kind==='cuda'?(process.platform==='win32'&&process.env.CUDA_PATH?path.join(process.env.CUDA_PATH,'bin/nvcc.exe'):'nvcc'):(process.env.CXX||'g++');
const args=kind==='cuda'?['--std=c++17','-O3','--expt-extended-lambda',`-arch=${arch}`,sourceFile,'-o',outputFile]:['-std=c++17','-O3',sourceFile,'-o',outputFile];
const report={schemaVersion:1,createdAt:new Date().toISOString(),backend:kind,sourceHash:digest,sourceFile,outputFile,
 requirements,sourceDerived:true,compilerLoweringUsesPrewrittenPhysics:false,
 driver:'SunOpt stride-12/static + stride-3/dynamic ABI adapter; calls generated TS functions',
 compiled:false,generated:true,arch,gpu:kind==='cuda',profileMode:profile,readyForProduction:false};
if(!generateOnly){
 const run=spawnSync(command,args,{encoding:'utf8',timeout:240000,maxBuffer:12*1024*1024});
 report.compile={command,args,exitCode:run.status,signal:run.signal,stdout:(run.stdout??'').slice(-20000),stderr:(run.stderr??'').slice(-20000),error:run.error?.message};
 report.compiled=run.status===0 && existsSync(outputFile);
}
writeFileSync(buildReport,JSON.stringify(report,null,2)+'\n');
console.log(`A9_BUILD_REPORT ${buildReport}`);
console.log(`A9_BUILD_STATUS ${report.compiled?'PASS':generateOnly?'GENERATED':'FAIL'}`);
console.log(`A9_BINARY ${outputFile}`);
if(!report.compiled&&!generateOnly){console.error(report.compile?.stdout,report.compile?.stderr);process.exitCode=1;}
