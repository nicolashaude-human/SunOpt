// rust/sunopt-node/scripts/build-sunopt-standalone-presentation.mjs
// Rebuild public/presentation/index.html as ONE self-contained HTML5 file.
// Reads local recorded results only; never calls a simulation or a server.
import fs from 'node:fs';
import path from 'node:path';
import { validateTankRun } from './sunopt-tank-slide-data.mjs';
import { analyseSolarSlide } from './sunopt-solar-slide-data.mjs';
const root=path.resolve(import.meta.dirname,'../../..');
const ppt=path.join(root,'public','presentation');
const template=path.join(ppt,'standalone-template.html');
const snapshotFile=path.join(ppt,'annual-simulation-static.json');
const seriesFile=path.join(ppt,'annual-simulation-static-series.json');
const outputFile=path.join(ppt,'index.html');
const outbox=path.join(root,'updates','outbox');
fs.mkdirSync(outbox,{recursive:true});
const snapshot=JSON.parse(fs.readFileSync(snapshotFile,'utf8'));
if(snapshot.schema!=='sunopt-presentation-year-static-v1'||snapshot.months?.length!==12||snapshot.source?.dataPoints!==35040)throw new Error('Ungueltiger Jahres-Snapshot.');
const solarCache=path.join(root,'data','cache','weather','reference-8001-2023-az0-inc30-loss14.json');
const rainFile=path.join(root,'data','cache','weather','rain-8001-2023-era5.json');
const solar=analyseSolarSlide(
 JSON.parse(fs.readFileSync(solarCache,'utf8')),
 JSON.parse(fs.readFileSync(rainFile,'utf8')),
);
const actualSeries=fs.existsSync(seriesFile)?JSON.parse(fs.readFileSync(seriesFile,'utf8')):null;
if(actualSeries){
 if(actualSeries.schema!=='sunopt-presentation-year-series-v1'||actualSeries.rows?.length!==snapshot.source.dataPoints||actualSeries.timestamps?.length!==actualSeries.rows.length)throw new Error('Unvollstaendige Zeitreihen.');
 if(actualSeries.sourceResponseSha256!==snapshot.source.responseSha256)throw new Error('Zeitreihen stammen aus einem anderen API-Lauf. Erneut capturen.');
}
function embed(obj){return JSON.stringify(obj).replace(/</g,'\\u003c').replace(/>/g,'\\u003e').replace(/&/g,'\\u0026').replace(/\u2028/g,'\\u2028').replace(/\u2029/g,'\\u2029')}
const tankFile=path.join(ppt,'tank-comparison-static.json');
const tankRuns=fs.existsSync(tankFile)?JSON.parse(fs.readFileSync(tankFile,'utf8')):{};
for(const [mode,run] of Object.entries(tankRuns))validateTankRun(run,mode);
if(tankRuns.A&&tankRuns.B&&['year','pvPeakPower','heatPumpThermalPower','gridPriceChfPerKWh','feedInPriceChfPerKWh']
 .some(key=>tankRuns.A[key]!==tankRuns.B[key]))throw new Error('Die Tankläufe A und B haben unterschiedliche Parameter.');
const html=fs.readFileSync(template,'utf8')
 .replace('__SUNOPT_EMBED_SNAPSHOT__',()=>embed(snapshot))
 .replace('__SUNOPT_EMBED_SERIES__',()=>embed(actualSeries))
 .replace('__SUNOPT_EMBED_TANKS__',()=>embed(tankRuns))
 .replace('__SUNOPT_EMBED_SOLAR__',()=>embed(solar));
if(/__SUNOPT_EMBED_/.test(html)||/<script[^>]+\bsrc\s*=|<link[^>]+\bhref\s*=|\bfetch\s*\(/i.test(html))throw new Error('Standalone-Pruefung: Externe Abhaengigkeit oder Platzhalter gefunden.');
const tmp=outputFile+'.tmp-'+process.pid;fs.writeFileSync(tmp,html);fs.renameSync(tmp,outputFile);
const report={status:'PASS',year:snapshot.source.year,modelSet:snapshot.source.modelSet,dataPoints:snapshot.source.dataPoints,hasDetailedSeries:!!actualSeries,seriesPoints:actualSeries?.rows.length??0,standaloneHtml:outputFile,bytes:Buffer.byteLength(html),noExternalAssets:true,noFetch:true,sourceResponseSha256:snapshot.source.responseSha256};
fs.writeFileSync(path.join(outbox,'sunopt-standalone-presentation-build.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
