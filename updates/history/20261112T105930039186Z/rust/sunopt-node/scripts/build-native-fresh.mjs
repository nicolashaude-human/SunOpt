// rust/sunopt-node/scripts/build-native-fresh.mjs
// Force Cargo/NVCC to notice freshly installed source updates, rebuild the
// napi-rs addon in the correct directory, then verify the requested B14E export.
import fs from "node:fs";
import path from "node:path";
import {spawnSync} from "node:child_process";
import {createRequire} from "node:module";
import {fileURLToPath} from "node:url";

const here=path.dirname(fileURLToPath(import.meta.url));
const nativeRoot=path.resolve(here,"..");
const inputs=[
  path.join(nativeRoot,"src","cuda_context.rs"),
  path.join(nativeRoot,"cuda","cuda_context.cu"),
  path.join(nativeRoot,"cuda","cuda_context.h"),
  path.join(nativeRoot,"build.rs"),
].filter((p)=>fs.existsSync(p));

const now=new Date();
console.log(`nativeRoot=${nativeRoot}`);
for(const p of inputs){
  fs.utimesSync(p,now,now);
  const s=fs.statSync(p);
  console.log(`touched=${p} mtime=${s.mtime.toISOString()}`);
}

let build;
if(process.platform==="win32"){
  // Node 24 on Windows can reject direct spawnSync("npm.cmd", ...) with EINVAL.
  // Run npm through the existing Developer Command Prompt environment instead.
  const comspec=process.env.ComSpec || "C:\\Windows\\System32\\cmd.exe";
  console.log(`buildShell=${comspec}`);
  console.log("buildCommand=npm run build");
  build=spawnSync(comspec,["/d","/s","/c","npm run build"],{
    cwd:nativeRoot,
    stdio:"inherit",
    shell:false,
    env:process.env,
  });
}else{
  console.log("buildCommand=npm run build");
  build=spawnSync("npm",["run","build"],{
    cwd:nativeRoot,
    stdio:"inherit",
    shell:false,
    env:process.env,
  });
}

if(build.error){
  console.error(build.error);
  process.exit(1);
}
if((build.status??1)!==0){
  process.exit(build.status??1);
}

const nodePath=path.join(nativeRoot,"sunopt_node.win32-x64-msvc.node");
if(!fs.existsSync(nodePath)){
  console.error(`native addon missing after build: ${nodePath}`);
  process.exit(2);
}
const st=fs.statSync(nodePath);
console.log(`nativePath=${nodePath}`);
console.log(`nativeSize=${st.size}`);
console.log(`nativeMtime=${st.mtime.toISOString()}`);

const require=createRequire(import.meta.url);
delete require.cache[require.resolve(nodePath)];
const addon=require(nodePath);
const exports=Object.keys(addon).filter((k)=>/b14|warp|parallel|merge/i.test(k)).sort();
console.log(`nativeExports=${JSON.stringify(exports)}`);
const hasB14e=typeof addon.runBlackwellB14EWarpParallel==="function" ||
              typeof addon.runBlackwellB14eWarpParallel==="function";
if(!hasB14e){
  console.error("B14E export still missing after forced fresh build.");
  process.exit(3);
}
console.log("NATIVE_FRESH_BUILD_RESULT=B14E_EXPORT_PRESENT");
