/*
 * Generic full TypeScript source runtime through the V8 JIT.
 * Handles the actual non-scalar SunOpt scope functions without any
 * specialized physics logic. CPU baseline only. Not the GPU backend.
 * Only execute trusted local source: vm is not a security boundary.
 */
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import vm from 'node:vm';
import path from 'node:path';
import { loadProjectTypeScript } from './approach-a-generic-typescript.mjs';
const ts = loadProjectTypeScript(import.meta.url);
const cache=new Map();
export function compileGenericFullSource(source, filename='trusted-source.ts') {
  const hash=createHash('sha256').update(source).digest('hex');
  const prior=cache.get(hash);
  if (prior)return prior;
  const file=ts.createSourceFile(filename,source,ts.ScriptTarget.ES2022,true);
  if(file.parseDiagnostics.length)throw new Error(`Invalid TS ${filename}`);
  if(file.statements.some(s=>ts.isImportDeclaration(s)||ts.isExportDeclaration(s)||ts.isExportAssignment(s)))
    throw new Error('Full-source runtime does not permit module dependencies');
  const output=ts.transpileModule(source,{fileName:filename,compilerOptions:{
    target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.CommonJS,
    strict:true,isolatedModules:true,
  },reportDiagnostics:true});
  if(output.diagnostics?.length)throw new Error(`TypeScript transpilation failed: ${output.diagnostics.map(d=>ts.flattenDiagnosticMessageText(d.messageText,' ')).join(' | ')}`);
  const module={exports:{}};
  const sandbox={exports:module.exports,module,console:{log(){},error(){}},
    Float64Array,Int32Array,Uint32Array,ArrayBuffer,Math,Number,Boolean};
  const context=vm.createContext(sandbox);
  // Source is trusted project code. Script timeout protects initial module load only.
  new vm.Script(output.outputText,{filename}).runInContext(context,{timeout:10000});
  const exports=Object.freeze({...module.exports});
  const handle=Object.freeze({hash,exports,run(name,...args){
    if(!Object.hasOwn(exports,name)||typeof exports[name]!=='function')throw new Error(`Unknown exported function ${name}`);
    return exports[name](...args);
  }});
  cache.set(hash,handle);
  return handle;
}
