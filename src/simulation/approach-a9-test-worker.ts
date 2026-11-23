/*
 * SunOpt A9 experiment; DISABLED unless SUNOPT_APPROACH_A9_EXPERIMENT=1.
 * Generic source-generated CUDA runs as a persistent native sidecar.
 * The real existing controller retains its 28 independent tank states and
 * executes the 35,040 annual steps serially. No production button changes.
 */
import { Worker } from 'node:worker_threads';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';

const MAX_TANKS=128, HEADER=64, CONFIG=12*MAX_TANKS, DYNAMIC=3*MAX_TANKS;
const CONFIG_OFFSET=HEADER, DYNAMIC_OFFSET=CONFIG_OFFSET+8*CONFIG;
const OUTPUT_OFFSET=DYNAMIC_OFFSET+8*DYNAMIC;
const ERROR_OFFSET=OUTPUT_OFFSET+8*MAX_TANKS;
const BYTES=ERROR_OFFSET+4096;
const WAIT_MS=120000;

type WorkerApi={
 configureLockstepPacked:(packed:Float64Array)=>void;
 runLockstepDynamicInto:(index:number,horizon:number,dynamic:Float64Array,output:Float64Array)=>void;
 profile:()=>unknown;
 resetProfile?:()=>void;
};

export function wrapApproachA9Worker<T extends WorkerApi>(
 original:T, exportPower:Float64Array, demandEnergy:Float64Array,
):T {
 const binary=process.env.SUNOPT_APPROACH_A9_BINARY;
 if(!binary||!path.isAbsolute(binary)||!existsSync(binary)){
   throw new Error('A9 experimental binary missing. Set SUNOPT_APPROACH_A9_BINARY after compiling the source-derived CUDA sidecar.');
 }
 if(exportPower.length!==demandEnergy.length || exportPower.length===0){
   throw new Error('A9 annual arrays are inconsistent');
 }
 const workerFile=path.join(process.cwd(),'rust','sunopt-node','scripts','approach-a9-ipc-worker.mjs');
 if(!existsSync(workerFile))throw new Error(`A9 IPC worker missing: ${workerFile}`);
 const shared=new SharedArrayBuffer(BYTES);
 const status=new Int32Array(shared,0,16);
 const config=new Float64Array(shared,CONFIG_OFFSET,CONFIG);
 const dynamicBuffer=new Float64Array(shared,DYNAMIC_OFFSET,DYNAMIC);
 const results=new Float64Array(shared,OUTPUT_OFFSET,MAX_TANKS);
 const errorBytes=new Uint8Array(shared,ERROR_OFFSET,4096);
 const worker=new Worker(workerFile,{workerData:{shared,binary,
    exportPower:Float64Array.from(exportPower),demandEnergy:Float64Array.from(demandEnergy)}});
 worker.unref();
 let tanks=0,calls=0,elapsedMs=0;
 function transact(op:'configure'|'step',data:Record<string,number>){
   Atomics.store(status,0,0);
   worker.postMessage({op,...data});
   const state=Atomics.wait(status,0,0,WAIT_MS);
   const result=Atomics.load(status,0);
   if(state==='timed-out'||result!==1){
      const zero=errorBytes.indexOf(0);
      const diagnostic=new TextDecoder().decode(errorBytes.subarray(0,zero<0?errorBytes.length:zero));
      void worker.terminate();
      throw new Error(`A9 ${op} ${state} result=${result}: ${diagnostic||'No worker diagnostic'}`);
   }
 }
 console.log('[SunOpt] A9 SOURCE-GENERATED CUDA WORKER SELECTED (experiment only)',{binary,annualValues:exportPower.length});
 return new Proxy(original,{
  get(target,property,receiver){
   if(property==='configureLockstepPacked')return (packed:Float64Array)=>{
     target.configureLockstepPacked(packed);
     if(packed.length%12!==0||packed.length===0||packed.length>CONFIG){
       throw new Error('A9 invalid packed constant count');
     }
     tanks=packed.length/12;
     config.set(packed);
     transact('configure',{tanks});
   };
   if(property==='runLockstepDynamicInto')return (
      index:number,horizon:number,dynamic:Float64Array,output:Float64Array,
   )=>{
      if(!tanks||dynamic.length!==tanks*3||output.length!==tanks){
        throw new Error('A9 dynamic/output ABI mismatch');
      }
      dynamicBuffer.set(dynamic);
      const start=performance.now();
      transact('step',{index,horizon});
      output.set(results.subarray(0,tanks));
      elapsedMs+=performance.now()-start;calls++;
      if(calls===1||calls%5000===0){
        console.log('[SunOpt] A9 SOURCE-GENERATED CUDA EXECUTION',{
          calls,tanks,annualStep:index,elapsedMs:Number(elapsedMs.toFixed(3)),
        });
      }
   };
   if(property==='profile')return ()=>{
      const base=target.profile() as object;
      return {...base,causalCalls:calls,causalJobs:calls*tanks,
        wallMs:elapsedMs,hostWaitMs:elapsedMs,
        // Sidecar GPU kernel timing is not available from the old native profiler.
        // Do not attribute these values to source-generated CUDA.
        kernelMs:0,parameterH2dMs:0,resultD2hMs:0,gpuTotalMs:0,
        a9SourceGenerated:true,a9TimingIsEndToEndIpc:true};
   };
   if(property==='resetProfile')return ()=>{
      calls=0;elapsedMs=0;target.resetProfile?.();
   };
   const value=Reflect.get(target,property,target);
   return typeof value==='function'?value.bind(target):value;
  },
 }) as T;
}
