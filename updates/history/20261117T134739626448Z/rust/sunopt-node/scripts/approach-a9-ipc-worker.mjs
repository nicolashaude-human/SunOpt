/*
 * A9 experiment: Node worker thread mediates synchronous controller calls to
 * a persistent CPU/CUDA sidecar, without restarting CUDA or re-uploading
 * annual arrays. No f(x) formula appears here.
 */
import {workerData,parentPort} from 'node:worker_threads';
import {spawn} from 'node:child_process';

const MAX_TANKS=128, HEADER=64, CONFIG=12*MAX_TANKS, DYNAMIC=3*MAX_TANKS;
const CONFIG_OFFSET=HEADER, DYNAMIC_OFFSET=CONFIG_OFFSET+8*CONFIG;
const OUTPUT_OFFSET=DYNAMIC_OFFSET+8*DYNAMIC;
const ERROR_OFFSET=OUTPUT_OFFSET+8*MAX_TANKS;
const codes=new Int32Array(workerData.shared,0,16);
const config=new Float64Array(workerData.shared,CONFIG_OFFSET,CONFIG);
const dynamic=new Float64Array(workerData.shared,DYNAMIC_OFFSET,DYNAMIC);
const output=new Float64Array(workerData.shared,OUTPUT_OFFSET,MAX_TANKS);
const errorBytes=new Uint8Array(workerData.shared,ERROR_OFFSET,4096);
let child=null, unread=Buffer.alloc(0), pending=null, deadError=null;
let tanks=0;
function fail(error){
 const str=String(error?.stack??error).slice(0,errorBytes.length-1);
 const bytes=new TextEncoder().encode(str);errorBytes.fill(0);errorBytes.set(bytes.slice(0,errorBytes.length-1));
 Atomics.store(codes,0,-1);Atomics.notify(codes,0,1);
}
function success(){Atomics.store(codes,0,1);Atomics.notify(codes,0,1);}
function append(b){
 unread=unread.length?Buffer.concat([unread,b]):b;
 if(pending&&unread.length>=pending.n){
  const req=pending;pending=null;
  const result=unread.subarray(0,req.n);unread=unread.subarray(req.n);req.resolve(result);
 }
}
function take(n){
 if(deadError)return Promise.reject(deadError);
 if(unread.length>=n){const result=unread.subarray(0,n);unread=unread.subarray(n);return Promise.resolve(result);}
 if(pending)return Promise.reject(Error('A9 concurrent read forbidden'));
 return new Promise((resolve,reject)=>{pending={n,resolve,reject};});
}
function encodeFloat64(values){
 const result=Buffer.allocUnsafe(values.length*8);
 for(let i=0;i<values.length;i++)result.writeDoubleLE(values[i],i*8);
 return result;
}
const u32=(a)=>{const b=Buffer.alloc(4);b.writeInt32LE(a,0);return b;};
function ensureChild(){
 if(child)return;
 child=spawn(workerData.binary,[],{stdio:['pipe','pipe','pipe'],windowsHide:true});
 child.stdout.on('data',append);
 child.stderr.on('data',b=>process.stderr.write(b));
 child.on('error',e=>{deadError=e;if(pending){pending.reject(e);pending=null;}fail(e);});
 child.on('close',(code,signal)=>{
  deadError=Error(`A9 sidecar exit=${code} signal=${signal}`);
  if(pending){pending.reject(deadError);pending=null;}
  if(Atomics.load(codes,0)===0)fail(deadError);
 });
}
function write(data){return new Promise((resolve,reject)=>{
 if(deadError)return reject(deadError);
 child.stdin.write(data,error=>error?reject(error):resolve());
});}
async function configure(message){
 ensureChild();tanks=message.tanks;
 if(tanks<1||tanks>MAX_TANKS)throw Error('A9 tank count outside 1..128');
 const hp=Float64Array.from(workerData.exportPower,v=>Math.max(0,v));
 const demand=Float64Array.from(workerData.demandEnergy,v=>Math.max(0,v));
 if(hp.length!==demand.length)throw Error('A9 annual series mismatch');
 const init=Buffer.concat([u32(1),u32(hp.length),u32(tanks),
   encodeFloat64(hp),encodeFloat64(demand),encodeFloat64(config.subarray(0,tanks*12))]);
 await write(init);
 const marker=await take(1);
 if(marker[0]!==82)throw Error(`A9 sidecar bad ready marker ${marker[0]}`);
 console.log(`[SunOpt] A9 source-generated sidecar initialized (${tanks} tanks, ${hp.length} annual records)`);
 success();
}
async function step(message){
 if(!child||!tanks)throw Error('A9 step before configure');
 const header=Buffer.concat([u32(2),u32(message.index),u32(message.horizon)]);
 await write(Buffer.concat([header,encodeFloat64(dynamic.subarray(0,tanks*3))]));
 const raw=await take(tanks*8);
 for(let i=0;i<tanks;i++){
  const v=raw.readDoubleLE(i*8);
  if(!Number.isFinite(v))throw Error(`A9 nonfinite cut at tank ${i}`);
  output[i]=v;
 }
 success();
}
let queue=Promise.resolve();
parentPort.on('message',msg=>{
 queue=queue.then(async()=>{
  if(msg.op==='configure')await configure(msg);
  else if(msg.op==='step')await step(msg);
  else if(msg.op==='shutdown'){if(child&&!deadError)await write(u32(3));child?.stdin.end();success();}
  else throw Error('A9 unsupported message '+msg.op);
 }).catch(fail);
});
