// Structural check – requires no GPU, and never runs the annual simulation.
import fs from 'node:fs';
import path from 'node:path';
const root = process.cwd();
const read = p=>fs.readFileSync(path.join(root,p),'utf8');
const checks = {
 staticPage: read('public/presentation/index.html').includes('annual-static-slide.js'),
 staticCss: read('public/presentation/index.html').includes('annual-static-slide.css'),
 noDynamicSimulationOnSlide: !/\/api\/debug\/hot-water|\/api\/tank-comparison/.test(read('public/presentation/annual-static-slide.js')),
 staticSnapshotOnly: read('public/presentation/annual-static-slide.js').includes('/presentation/annual-simulation-static.json'),
 oneTimeCapture: read('rust/sunopt-node/scripts/capture-sunopt-annual-slide.ts').includes('POST') && read('rust/sunopt-node/scripts/capture-sunopt-annual-slide.ts').includes('/api/debug/hot-water'),
 captureFromActualApi: read('rust/sunopt-node/scripts/capture-sunopt-annual-slide.ts').includes('simulationDurationMs'),
 captureUsesProjectConfig: read('rust/sunopt-node/scripts/capture-sunopt-annual-slide.ts').includes('HOT_WATER_TEST_CONFIG'),
 guardSetupMeasured: read('src/app/api/tank-comparison/route.ts').includes('gpuGuardPreparationDurationMs'),
 guardRestoreMeasured: read('src/app/api/tank-comparison/route.ts').includes('gpuGuardRestoreDurationMs'),
};
const report={status:Object.values(checks).every(Boolean)?'PASS':'FAIL',checks, note:'Structural verification only. Annual data must be captured once on the local SunOpt application.'};
const outbox=path.join(root,'updates','outbox');fs.mkdirSync(outbox,{recursive:true});
fs.writeFileSync(path.join(outbox,'sunopt-annual-slide-verify.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
if(report.status!=='PASS')process.exitCode=1;
