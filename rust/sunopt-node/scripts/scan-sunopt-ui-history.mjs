#!/usr/bin/env node
/**
 * SunOpt UI/Docs scanner, read-only. Uses only Node built-ins.
 * Output: updates/outbox/sunopt-ui-scan-<UTC>.zip + .json + .md
 * Does not modify project sources, project configuration, Git, or watcher backups.
 */
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { deflateRawSync } from 'node:zlib';

const SCANNER_VERSION = '2026-11-17-ui-inventory-v1';
const SCRIPT_DIRECTORY = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_ROOT = path.resolve(SCRIPT_DIRECTORY, '../../..');
const PROJECT_ROOT = path.resolve(process.argv[2] || DEFAULT_ROOT);
const OUTBOX = path.join(PROJECT_ROOT, 'updates', 'outbox');
const MAX_FILE_BYTES = 750_000;
const MAX_TOTAL_BYTES = 38_000_000;
const MAX_FILES = 2200;
const MAX_GIT_SNAPSHOTS = 180;
const MAX_GIT_PER_PATH = 5;
const MAX_GIT_LOG_COMMITS = 250;
const SKIP_DIRS = new Set([
  'node_modules', '.git', '.next', '.turbo', '.cache', '.vercel', 'dist', 'build',
  'target', 'coverage', '.venv', 'venv', '__pycache__', '.idea', '.vscode',
  'outbox', '.pytest_cache', '.pnpm-store', 'vendor', '.storybook-cache',
]);
const CURRENT_DIRS = [
  'src/app', 'app', 'src/pages', 'pages',
  'src/components', 'components', 'src/styles', 'styles',
  'src/hooks', 'src/context', 'src/contexts', 'src/providers',
  'src/lib', 'src/ui', 'public', 'docs',
];
const WATCHER_HISTORY_ROOT = 'updates/history';
const SIMPLE_COLOR_RE = /#[a-fA-F0-9]{3,8}\b|(?:rgba?|hsla?|oklch|color-mix)\([^\n;]{1,130}\)|(?:var\(--[\w-]+\))/g;
const CSS_VAR_RE = /(--[\w-]+)\s*:\s*([^;\n}]{1,180})/g;
const TAILWIND_COLOR_RE = /\b(?:bg|text|border|ring|fill|stroke|from|via|to)-(?:\[[^\]\n]{1,70}\]|[a-z][\w-]{2,45})\b/g;

if (!fs.existsSync(path.join(PROJECT_ROOT, 'package.json')) ||
    !(['src/app', 'app', 'pages', 'src/pages'].some(p => fs.existsSync(path.join(PROJECT_ROOT,p))))) {
  console.error('SunOpt UI Scan: invalid project root:', PROJECT_ROOT);
  console.error('Expected package.json and an app/pages folder. No files written.');
  process.exit(2);
}

const entries = [];
const records = [];
const skipped = [];
const errors = [];
const pathSeen = new Set();
const hashToFile = new Map();
let totalBytes = 0;

function safeRelative(p) {
  return String(p).replace(/\\/g,'/').replace(/^\/+/, '');
}
function sha256(bytes) { return createHash('sha256').update(bytes).digest('hex'); }
function isHiddenSensitive(name) {
  return /(^|\/)(?:\.env(?:\.|$)|id_rsa|id_ed25519|credentials|private[-_]?key|\.npmrc|\.pypirc|\.ssh|token[s]?\.json|secret[s]?\.json)(?:\/|$)/i.test(name);
}
function isTextFile(filename) { return /\.(?:tsx?|jsx?|mjs|cjs|css|scss|sass|less|html?|mdx?|json|svg|yml|yaml)$/i.test(filename); }
function isSourceCode(filename) { return /\.(?:tsx?|jsx?|mjs|cjs|css|scss|sass|less|html?|mdx?)$/i.test(filename); }
function isUiFile(relativePath, allowArchiveName=false) {
  const p = safeRelative(relativePath); const low = p.toLowerCase();
  if (isHiddenSensitive(p) || /(^|\/)(?:api|__tests__|fixtures|testdata)(\/|$)/i.test(p)) return false;
  if (/(?:^|\/)(?:package-lock\.json|pnpm-lock\.yaml|yarn\.lock)$/i.test(p)) return false;
  const base = p.split('/').at(-1) || '';
  const ext = path.extname(base).toLowerCase();
  const archived = allowArchiveName && /\.(?:bak|backup|old|orig|save|previous|legacy)(?:\.|$)|~$/i.test(base);
  if (archived) return /\.(?:tsx?|jsx?|css|scss|mdx?|html?)(?:\.|$)/i.test(base);
  if (low.startsWith('src/app/') || low.startsWith('app/') || low.startsWith('src/pages/') || low.startsWith('pages/')) {
    return /\.(?:tsx?|jsx?|css|scss|sass|less|html?|mdx?)$/i.test(base);
  }
  if (/^(?:src\/)?(?:components|styles|hooks|context|contexts|providers|ui)\//i.test(p)) {
    return /\.(?:tsx?|jsx?|css|scss|sass|less|mdx?|html?)$/i.test(base);
  }
  if (low.startsWith('src/lib/')) {
    return /(?:nav|menu|ui|design|theme|style|color|palette|layout|component|sidebar|scroll|animation|icon)/i.test(p)
      && /\.(?:tsx?|jsx?|css|scss)$/i.test(base);
  }
  if (low.startsWith('docs/')) {
    return /\.(?:mdx?|html?|css|scss)$/i.test(base);
  }
  if (low.startsWith('public/')) {
    return /\.(?:svg|png|jpe?g|webp|gif|ico|html?|css)$/i.test(base);
  }
  if (/^(?:package\.json|tailwind\.config\.[\w]+|postcss\.config\.[\w]+|next\.config\.[\w]+|tsconfig\.json|components\.json|eslint\.config\.[\w]+)$/i.test(p)) return true;
  return false;
}
function fileCategory(p) {
  const low=safeRelative(p).toLowerCase();
  if (/(?:^|\/)(?:page|index)(?:(?:[-.]old|[-.]bak|[-.]backup|[-.]legacy))?\.(?:tsx|jsx|ts|js|mdx)(?:\.(?:bak|old|backup|orig))?$/.test(low) && /(^|\/)(?:app|pages)\//.test(low)) return 'page';
  if (/^(?:src\/)?pages\/.+\.(?:tsx|jsx|ts|js|mdx)$/.test(low)) return 'page';
  if (/(?:^|\/)(?:layout|template|loading|error|not-found|default)\.(?:tsx|jsx|ts|js|mdx)$/.test(low)) return 'layout';
  if (low.includes('component') || low.startsWith('src/ui/')) return 'component';
  if (/\.(css|scss|sass|less)$/.test(low) || low.startsWith('src/styles/')) return 'style';
  if (low.startsWith('public/')) return 'asset';
  if (low.startsWith('docs/')) return 'documentation';
  return 'support';
}
function walk(start, onFile, maxDepth=32) {
  if (!fs.existsSync(start)) return;
  const todo=[{dir:start,depth:0}];
  while(todo.length) {
    const {dir,depth}=todo.pop();
    let nodes;
    try { nodes=fs.readdirSync(dir,{withFileTypes:true}).sort((a,b)=>a.name.localeCompare(b.name)); }
    catch(e){ errors.push({path:safeRelative(path.relative(PROJECT_ROOT,dir)),error:e.message});continue; }
    for(const node of nodes) {
      if (node.isSymbolicLink()) continue;
      const full=path.join(dir,node.name);
      if(node.isDirectory()) {
        if(depth<maxDepth && !SKIP_DIRS.has(node.name) && !node.name.startsWith('.git')) todo.push({dir:full,depth:depth+1});
      } else if(node.isFile()) onFile(full);
    }
  }
}
function analyzeText(txt) {
  const colors=[...new Set(txt.match(SIMPLE_COLOR_RE) || [])].slice(0,100);
  const tailwind=[...new Set(txt.match(TAILWIND_COLOR_RE) || [])].slice(0,80);
  const variables=[...txt.matchAll(CSS_VAR_RE)].slice(0,100).map(m=>({name:m[1],value:m[2].trim()}));
  const signals=[];
  for(const [label,pattern] of [
    ['navigation',/onMouseEnter|onMouseLeave|mouseenter|mouseleave|onPointerEnter|<nav\b|navigation|sidebar|submenu|menubar/i],
    ['drag-scroll',/pointermove|pointerdown|mousedown|onMouseDown|scrollLeft|overflow-x|touch-action|user-select/i],
    ['green-box',/green|emerald|#0[0-9a-f]{5}|#1[0-9a-f]{5}|border-green|ring-green/i],
    ['ui-usage',/className=|styled\.|<Card|<Button|<Box|<Dialog/i],
  ]) if(pattern.test(txt)) signals.push(label);
  return {colors,tailwind,cssVariables:variables,signals};
}
function insertFile(sourcePath, zipPath, sourceKind, historyInfo={}) {
  const zipName = safeRelative(zipPath);
  if (pathSeen.has(zipName)) return;
  if (entries.length >= MAX_FILES) { skipped.push({path:zipName,reason:'file-count-cap'});return; }
  let stat;
  try { stat=fs.statSync(sourcePath); } catch(e) { errors.push({path:sourcePath,error:e.message});return; }
  if(!stat.isFile()) return;
  if(stat.size > MAX_FILE_BYTES) { skipped.push({path:zipName,reason:'individual-size-cap',bytes:stat.size});return; }
  if(totalBytes+stat.size>MAX_TOTAL_BYTES) { skipped.push({path:zipName,reason:'aggregate-size-cap',bytes:stat.size});return; }
  let bytes;
  try { bytes=fs.readFileSync(sourcePath); }catch(e){errors.push({path:sourcePath,error:e.message});return;}
  insertBytes(bytes, zipName, sourceKind, historyInfo);
}
function insertBytes(bytes, zipPath, sourceKind, historyInfo={}) {
  const zipName=safeRelative(zipPath);
  if(pathSeen.has(zipName))return;
  if(bytes.length > MAX_FILE_BYTES || totalBytes+bytes.length>MAX_TOTAL_BYTES || entries.length>=MAX_FILES) {
    skipped.push({path:zipName,reason:'size-or-count-cap',bytes:bytes.length});return;
  }
  pathSeen.add(zipName);
  const hash=sha256(bytes);
  const duplicateOf=hashToFile.get(hash) || null;
  const originalPath=historyInfo.originalPath || zipName.replace(/^files\/(current|watcher-history|git|archive)\//,'');
  const category=fileCategory(originalPath);
  let annotation={colors:[],tailwind:[],cssVariables:[],signals:[]};
  if(isTextFile(zipName) && !/\.(?:svg)$/i.test(zipName)) {
    annotation=analyzeText(bytes.toString('utf8'));
  }
  const record={archivePath:zipName,originalPath,category,sourceKind,bytes:bytes.length,sha256:hash,
    duplicateOf,...historyInfo,...annotation};
  records.push(record);
  if(!duplicateOf) { entries.push({name:zipName,data:bytes});totalBytes+=bytes.length;hashToFile.set(hash,zipName);}
}

// 1. Current UI and explicit top-level design configuration.
for(const root of CURRENT_DIRS) {
  walk(path.join(PROJECT_ROOT,root),full=>{
    const rel=safeRelative(path.relative(PROJECT_ROOT,full));
    if(isUiFile(rel,true)) insertFile(full,`files/current/${rel}`,'current',{originalPath:rel});
  });
}
for(const name of ['package.json','tailwind.config.js','tailwind.config.ts','postcss.config.js','postcss.config.mjs',
                   'next.config.ts','next.config.js','tsconfig.json','components.json']) {
  const full=path.join(PROJECT_ROOT,name);
  if(fs.existsSync(full)) insertFile(full,`files/current/${name}`,'current',{originalPath:name});
}

// 2. Watcher backups: recovered earlier versions of UI and docs, without touching them.
walk(path.join(PROJECT_ROOT,WATCHER_HISTORY_ROOT),full=>{
  const rel=safeRelative(path.relative(PROJECT_ROOT,full));
  const prefix=`${WATCHER_HISTORY_ROOT}/`;
  const rest=rel.slice(prefix.length);
  const m=rest.match(/^([^/]+)\/(.+)$/);
  if(!m) return;
  const [,snapshot,sourcePath]=m;
  if(isUiFile(sourcePath,true)) insertFile(full,`files/watcher-history/${snapshot}/${sourcePath}`,'watcher-history',{
    originalPath:sourcePath,snapshot
  });
});

// 3. Alternate local copies in common project backup folders.
for(const name of ['backup','backups','archive','archives','old','legacy','_old','_backup']) {
  walk(path.join(PROJECT_ROOT,name),full=>{
    const rel=safeRelative(path.relative(PROJECT_ROOT,full));
    const actual=rel.slice(name.length+1);
    if(isUiFile(actual,true)) insertFile(full,`files/archive/${rel}`,'local-archive',{originalPath:actual});
  },20);
}

// 4. Historical committed pages, layouts, CSS, navigation and docs from Git.
let gitAvailable=false;
try {
  const gitRoot=execFileSync('git',['rev-parse','--show-toplevel'],{cwd:PROJECT_ROOT,encoding:'utf8',timeout:3000,stdio:['ignore','pipe','ignore']}).trim();
  gitAvailable=path.resolve(gitRoot).toLowerCase()===PROJECT_ROOT.toLowerCase();
} catch { /* Git may be unavailable or this may be an unversioned folder. */ }
const gitSelectedPaths=new Map();
if(gitAvailable) {
  try {
    const log=execFileSync('git',[
      'log','--all','--no-renames',`--max-count=${MAX_GIT_LOG_COMMITS}`,'--format=COMMIT:%H','--name-only','--',
      'src/app','app','src/pages','pages','src/components','components','src/styles','styles','docs',
    ],{cwd:PROJECT_ROOT,encoding:'utf8',maxBuffer:5_000_000,timeout:25_000});
    let commit=null;
    for(const line of log.split(/\r?\n/)) {
      const s=line.trim();
      if(s.startsWith('COMMIT:')) {commit=s.slice(7);continue;}
      if(!commit || !s || !isUiFile(s,true))continue;
      // Strong preference for old pages, layouts, relevant stylesheet and nav sources.
      const cat=fileCategory(s);
      const nav=/(nav|menu|sidebar|header|scroll|theme|color|token|palette)/i.test(s);
      if(!['page','layout','style','documentation'].includes(cat) && !nav)continue;
      const arr=gitSelectedPaths.get(s) || [];
      if(arr.length<MAX_GIT_PER_PATH && !arr.includes(commit)) {arr.push(commit);gitSelectedPaths.set(s,arr);}
    }
    let historicalCount=0;
    for(const [originalPath, commits] of gitSelectedPaths) {
      for(const commit of commits) {
        if(historicalCount>=MAX_GIT_SNAPSHOTS)break;
        try {
          const bytes=execFileSync('git',['show',`${commit}:${originalPath}`],{
            cwd:PROJECT_ROOT,maxBuffer:MAX_FILE_BYTES+4096,timeout:3500,stdio:['ignore','pipe','ignore']});
          const before=records.length;
          insertBytes(bytes,`files/git/${commit.slice(0,12)}/${originalPath}`,'git-history',{
            originalPath,commit,
          });
          if(records.length>before)historicalCount++;
        }catch{/* Source path may be absent at this revision; history can include deletions. */}
      }
    }
  } catch(e) {errors.push({source:'git-log',error:e.message});}
}

function routeOf(originalPath) {
  const p=safeRelative(originalPath);
  const appPrefix=p.match(/^(?:src\/)?app\//);
  const pagesPrefix=p.match(/^(?:src\/)?pages\//);
  if(!appPrefix && !pagesPrefix) return null;
  let rel=p.slice((appPrefix||pagesPrefix)[0].length);
  if(/^api\//i.test(rel)) return null;
  if(appPrefix) {
    if(!/(?:^|\/)page\.(?:tsx|jsx|ts|js|mdx)$/i.test(rel)) return null;
    rel=rel.replace(/(?:^|\/)page\.(?:tsx|jsx|ts|js|mdx)$/i,'');
    rel=rel.split('/').filter(part=>part&&!/^\([^)]+\)$/.test(part)).join('/');
  } else {
    if(!/\.(?:tsx|jsx|ts|js|mdx)$/i.test(rel)) return null;
    rel=rel.replace(/\.(?:tsx|jsx|ts|js|mdx)$/i,'');
    rel=rel.replace(/(?:^|\/)index$/i,'');
  }
  return rel ? `/${rel.replace(/^\/+|\/+$/g,'')}` : '/';
}

const current=records.filter(r=>r.sourceKind==='current');
const history=records.filter(r=>r.sourceKind!=='current');
const pages=current.filter(r=>r.category==='page').map(r=>({route:routeOf(r.originalPath),file:r.originalPath})).filter(r=>r.route!==null).sort((a,b)=>a.route.localeCompare(b.route));
const colors=new Map();
for(const r of current) for(const c of r.colors) {
  const prev=colors.get(c) || {count:0,files:[]};prev.count++;
  if(prev.files.length<12)prev.files.push(r.originalPath);
  colors.set(c,prev);
}
const navFiles=current.filter(r=>r.signals.includes('navigation')).map(r=>r.originalPath);
const dragFiles=current.filter(r=>r.signals.includes('drag-scroll')).map(r=>r.originalPath);
const greenFiles=current.filter(r=>r.signals.includes('green-box')).map(r=>r.originalPath);
const oldPages=history.filter(r=>r.category==='page').map(r=>({path:r.originalPath,source:r.sourceKind,snapshot:r.snapshot||null,commit:r.commit||null,archivePath:r.archivePath,duplicateOf:r.duplicateOf}));
const report={
  scanner:SCANNER_VERSION,createdAt:new Date().toISOString(),root:PROJECT_ROOT,
  safety:{readOnly:true,productionCodeChanged:false,gitChanged:false,watcherBackupsChanged:false,
    excluded:['.env','credential/private key files','node_modules','.next','native binaries','API routes','updates/outbox'],
    maxFileBytes:MAX_FILE_BYTES,maxTotalBytes:MAX_TOTAL_BYTES,maxFiles:MAX_FILES},
  counts:{current:current.length,historical:history.length,archivedUniqueContent:entries.length,
    totalSourceBytes:totalBytes,pages:pages.length,components:current.filter(r=>r.category==='component').length,
    styleFiles:current.filter(r=>r.category==='style').length,docs:current.filter(r=>r.category==='documentation').length,
    oldPageCandidates:oldPages.length,gitAvailable,gitPathsSearched:gitSelectedPaths.size,
    skipped:skipped.length,errors:errors.length},
  routes:pages,oldPages,
  navigationCandidates:navFiles,dragScrollCandidates:dragFiles,greenBoxCandidates:greenFiles,
  colorOccurrences:[...colors.entries()].sort((a,b)=>b[1].count-a[1].count).map(([color,data])=>({color,...data})),
  files:records,skipped,errors,
};
const md=[];
md.push('# SunOpt – UI, Pages, Components und ältere Versionen');
md.push('',`**Scan:** ${report.createdAt}`,`**Projekt:** ${PROJECT_ROOT}`,
  '**Modus:** nur lesen, keine Änderung an A/B, JIT, Simulation oder UI', '',
  '## Umfang', '',
  `- Aktuelle UI-/Dokumentationsdateien: ${current.length}`,
  `- Historische Fundstellen: ${history.length} (Git, Watcher-History, lokale Backups)`,
  `- Pages: ${pages.length}; Components: ${report.counts.components}; Styles: ${report.counts.styleFiles}`,
  `- Frühere Page-Kandidaten: ${oldPages.length}`,
  `- Gespeicherte eindeutige Quelldateien: ${entries.length}`,
  `- Git auswertbar: ${gitAvailable ? 'ja' : 'nein'}`, '',
  '## Aktuelle Pages', '', '| Route | Datei |', '|---|---|',
  ...pages.map(p=>`| \`${p.route}\` | \`${p.file}\` |`),'',
  '## Mögliche alte Pages', '',
  ...(oldPages.length?oldPages.slice(0,100).map(p=>`- **${p.source}**: \`${p.path}\` (${p.snapshot||p.commit?.slice(0,12)||'lokal'})`):['Keine früheren Page-Versionen in den verfügbaren Backups oder Git-Commits gefunden.']),
  '', '## Navigation und Drag-Scroll – Fundstellen', '',
  ...[...new Set([...navFiles,...dragFiles])].map(p=>`- \`${p}\``),
  '', '## Grüne Boxen – mögliche Fundstellen', '',
  ...(greenFiles.length?greenFiles.map(p=>`- \`${p}\``):['Keine eindeutigen grünen Styles erkannt.']),
  '', '## Häufige Farben / CSS-Variablen', '',
  '| Wert | Fundstellen |', '|---|---:|',
  ...report.colorOccurrences.slice(0,60).map(c=>`| \`${c.color.replace(/\|/g,'\\|')}\` | ${c.count} |`),
  '', '## Hinweise', '',
  '- Dies ist eine Quellcode-/Dateiinventur, keine automatisierte visuelle Browser-Prüfung.',
  '- Alte Dateien werden nur gefunden, soweit Git-Historie oder lokale Sicherungen vorhanden sind.',
  '- Identische Inhalte werden im ZIP nur einmal gespeichert; Duplikate sind in `inventory.json` verlinkt.',
  '- Prüfe das ZIP vor dem Weitergeben auf projektinterne Informationen.',
  `- Übersprungene Dateien: ${skipped.length}; Lesefehler: ${errors.length}.`, '',
);
const stamp=new Date().toISOString().replace(/[-:.]/g,'').replace('T','-').replace('Z','Z');
const name=`sunopt-ui-scan-${stamp}`;
const json=JSON.stringify(report,null,2)+'\n';
const markdown=md.join('\n')+'\n';
entries.push({name:'inventory.json',data:Buffer.from(json,'utf8')});
entries.push({name:'UI_SCAN_REPORT.md',data:Buffer.from(markdown,'utf8')});

// Minimal standard Zip writer (DEFLATE), no external dependencies.
const crcTable=Array.from({length:256},(_,i)=>{let c=i;for(let k=0;k<8;k++)c=(c&1)?(0xedb88320^(c>>>1)):(c>>>1);return c>>>0;});
function crc32(data) {let c=0xffffffff;for(const b of data)c=crcTable[(c^b)&255]^(c>>>8);return (c^0xffffffff)>>>0;}
function createZip(files) {
  const chunks=[],directory=[]; let offset=0;
  const now=new Date();
  const dosTime=((now.getHours()&31)<<11)|((now.getMinutes()&63)<<5)|((Math.floor(now.getSeconds()/2))&31);
  const dosDate=(((Math.max(1980,now.getFullYear())-1980)&127)<<9)|(((now.getMonth()+1)&15)<<5)|(now.getDate()&31);
  for(const {name,data} of files) {
    const label=Buffer.from(name,'utf8'),deflated=deflateRawSync(data,{level:6}),crc=crc32(data);
    const head=Buffer.alloc(30);
    head.writeUInt32LE(0x04034b50,0); head.writeUInt16LE(20,4);head.writeUInt16LE(0x0800,6);
    head.writeUInt16LE(8,8);head.writeUInt16LE(dosTime,10);head.writeUInt16LE(dosDate,12);
    head.writeUInt32LE(crc,14);head.writeUInt32LE(deflated.length,18);head.writeUInt32LE(data.length,22);
    head.writeUInt16LE(label.length,26);head.writeUInt16LE(0,28);
    chunks.push(head,label,deflated);
    const central=Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50,0);central.writeUInt16LE(20,4);central.writeUInt16LE(20,6);
    central.writeUInt16LE(0x0800,8);central.writeUInt16LE(8,10);
    central.writeUInt16LE(dosTime,12);central.writeUInt16LE(dosDate,14);
    central.writeUInt32LE(crc,16);central.writeUInt32LE(deflated.length,20);central.writeUInt32LE(data.length,24);
    central.writeUInt16LE(label.length,28);central.writeUInt32LE(0,38);central.writeUInt32LE(offset,42);
    directory.push(central,label);offset+=head.length+label.length+deflated.length;
  }
  const centralSize=directory.reduce((s,b)=>s+b.length,0);
  const end=Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50,0);
  end.writeUInt16LE(files.length,8);end.writeUInt16LE(files.length,10);
  end.writeUInt32LE(centralSize,12);end.writeUInt32LE(offset,16);
  return Buffer.concat([...chunks,...directory,end]);
}
fs.mkdirSync(OUTBOX,{recursive:true});
const targetZip=path.join(OUTBOX,`${name}.zip`),targetJson=path.join(OUTBOX,`${name}.json`),targetMd=path.join(OUTBOX,`${name}.md`);
fs.writeFileSync(targetZip,createZip(entries));
fs.writeFileSync(targetJson,json);
fs.writeFileSync(targetMd,markdown);
console.log('SunOpt UI scan complete (READ-ONLY)');
console.log(JSON.stringify({zip:targetZip,report:targetJson,markdown:targetMd,counts:report.counts},null,2));
