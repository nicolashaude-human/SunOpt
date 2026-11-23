/*
 * Narrow, fail-closed A9 integration patch for the existing controller.
 * The watcher ZIP itself does NOT replace production files. This script is
 * explicitly invoked by the A9 acceptance runner and restores original.
 */
import {readFileSync,writeFileSync,existsSync,mkdirSync} from 'node:fs';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {findSunOptProjectRoot} from './approach-a-generic-typescript.mjs';
const root=findSunOptProjectRoot(import.meta.url)??process.cwd();
const dest=path.join(root,'src/simulation/tank-comparison-gpu-lockstep.ts');
const backup=path.join(root,'updates/outbox/approach-a9-bridge-original.ts');
const stamp=path.join(root,'updates/outbox/approach-a9-bridge-state.json');
const baseline=readFileSync(dest,'utf8');
const hash=x=>createHash('sha256').update(x).digest('hex');
const importText='import { wrapApproachA9Worker } from "./approach-a9-test-worker";\n';
const replacement=`    const originalWorker = new CudaCausalWorker(\n      exportPower,\n      demandEnergy,\n      deviceId,\n    );\n\n    if (causalPath === "A" && process.env.SUNOPT_APPROACH_A9_EXPERIMENT === "1") {\n      return wrapApproachA9Worker(originalWorker, exportPower, demandEnergy);\n    }\n\n    return originalWorker;`;
const original=`    return new CudaCausalWorker(\n      exportPower,\n      demandEnergy,\n      deviceId,\n    );`;
const mode=process.argv[2]??'--check';
if(mode==='--check'){
 const possible=baseline.includes(original)&&!baseline.includes('wrapApproachA9Worker');
 console.log(JSON.stringify({status:possible?'PATCHABLE':baseline.includes('wrapApproachA9Worker')?'ALREADY_PATCHED':'BLOCK',path:dest,hash:hash(baseline)}));
 if(!possible&&!baseline.includes('wrapApproachA9Worker'))process.exitCode=1;
}else if(mode==='--apply'){
 if(existsSync(stamp))throw Error('A9 bridge state already exists; restore before new test');
 if(!baseline.includes(original)||baseline.includes('wrapApproachA9Worker')||baseline.split(original).length!==2)
   throw Error('A9 controller insertion point missing/ambiguous, refusing to change source');
 if(baseline.includes(importText))throw Error('A9 import already present');
 const changed=importText+baseline.replace(original,replacement);
 if(!changed.includes(replacement)||!changed.includes(importText))throw Error('A9 patch validation failed');
 mkdirSync(path.dirname(backup),{recursive:true});
 writeFileSync(backup,baseline);
 writeFileSync(dest,changed);
 writeFileSync(stamp,JSON.stringify({originalSha256:hash(baseline),installedSha256:hash(changed),path:dest,backup},null,2));
 console.log('A9_BRIDGE_PATCH_APPLIED',dest);
}else if(mode==='--restore'){
 if(!existsSync(stamp)||!existsSync(backup))throw Error('A9 patch metadata/backup missing: refuse unsafe restore');
 const s=JSON.parse(readFileSync(stamp,'utf8'));
 if(hash(baseline)!==s.installedSha256)throw Error('A9 patched source changed since installation; refuse to overwrite user edits');
 const originalBytes=readFileSync(backup,'utf8');
 if(hash(originalBytes)!==s.originalSha256)throw Error('A9 backup integrity mismatch');
 writeFileSync(dest,originalBytes);
 // Deliberately retain backup/metadata as audit trail, remove state to allow next test.
 await import('node:fs').then(({unlinkSync})=>unlinkSync(stamp));
 console.log('A9_BRIDGE_PATCH_RESTORED',dest);
}else throw Error('Use --check, --apply or --restore');
