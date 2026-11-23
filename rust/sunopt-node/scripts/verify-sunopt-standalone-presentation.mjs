// rust/sunopt-node/scripts/verify-sunopt-standalone-presentation.mjs
// Structural / data-integrity check only. Does not execute the simulation.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { Script } from 'node:vm';
import { validateTankRun } from './sunopt-tank-slide-data.mjs';
import { analyseSolarSlide } from './sunopt-solar-slide-data.mjs';
const root=path.resolve(import.meta.dirname,'../../..');
const loc=p=>path.join(root,p);
const html=fs.readFileSync(loc('public/presentation/index.html'),'utf8');
for(const [,script] of html.matchAll(/<script>([\s\S]*?)<\/script>/g))new Script(script);
const snapshot=JSON.parse(fs.readFileSync(loc('public/presentation/annual-simulation-static.json'),'utf8'));
const template=fs.readFileSync(loc('public/presentation/standalone-template.html'),'utf8');
const capture=fs.readFileSync(loc('rust/sunopt-node/scripts/capture-sunopt-annual-slide.ts'),'utf8');
const embedded=(id)=>{
 const match=html.match(new RegExp(`<script id="${id}" type="application/json">([\\s\\S]*?)<\\/script>`));
 return match?JSON.parse(match[1]):null;
};
const snapInHtml=embedded('year-snapshot'),seriesInHtml=embedded('year-series');
const solarInHtml=embedded('solar-analysis');
const solarCache=JSON.parse(fs.readFileSync(loc('data/cache/weather/reference-8001-2023-az0-inc30-loss14.json'),'utf8'));
const rainCache=JSON.parse(fs.readFileSync(loc('data/cache/weather/rain-8001-2023-era5.json'),'utf8'));
assert.deepEqual(solarInHtml,analyseSolarSlide(solarCache,rainCache));
const tanksInHtml=embedded('tank-runs');
const tankSnapshot=JSON.parse(fs.readFileSync(loc('public/presentation/tank-comparison-static.json'),'utf8'));
assert.deepEqual(tanksInHtml,tankSnapshot);
for(const mode of ['A','B'])validateTankRun(tanksInHtml[mode],mode);
const tankRun=tanksInHtml.B;
const reserve=tankRun.thermalReserve;
const pvEnergy=point=>point.feedIn+point.selfConsumption;
const monthSumGood= ['existing','quantumGravity','plasmaWave'].every(m=>['gridImportEnergy','pvUsedByHeatPumpEnergy','heatPumpEnergy','storageLossEnergy','unmetThermalDemand'].every(k=>{
 const total=snapshot.months.reduce((s,x)=>s+x[m][k],0);
 return Math.abs(total-snapshot.annual[m][k])<1e-5;
}));
const checks={
 oneHtmlFile:html.startsWith('<!doctype html>'),
 noExternalStyles:/<link\b/i.test(html)===false,
 noExternalScripts:/<script\b[^>]+\bsrc\s*=/i.test(html)===false,
 noExternalImages:/<img\b[^>]+\bsrc\s*=/i.test(html)===false,
 noIframe:/<iframe\b/i.test(html)===false,
 noFetch:/\bfetch\s*\(/i.test(html)===false,
 noEmbeddedFilePaths:!html.includes('file:///')&&!html.includes('src="/'),
 embeddedSnapshotSource:!!snapInHtml&&snapInHtml.source.responseSha256===snapshot.source.responseSha256,
 dataCount:snapInHtml?.source.dataPoints===35040,
 tankPvBalance:tankRun.points.every(point=>
  Math.abs(pvEnergy(point)-pvEnergy(tankRun.points[0]))<1e-6&&
  Math.abs(point.feedInRevenue-point.feedIn*tankRun.feedInPriceChfPerKWh)<1e-6),
 tankThermalReserve:reserve?.horizonHours===168&&reserve.maxHotWaterDemandKWh>0&&
  reserve.usableCapacityKWhPerLitre>0&&reserve.spaceHeating?.peakWeekShare>0&&
  reserve.spaceHeating.peakWeekShare<1,
 solarShares:Math.abs(solarInHtml.nightPercent+solarInHtml.seasonalPercent+
  solarInHtml.weakPercent+solarInHtml.productivePercent-100)<1e-8,
 solarDryDayProfiles:['summer','winter'].every(season=>{
  const summer=season==='summer';
  const entries=Object.entries(rainCache.rainByDate).filter(([date])=>{
   const month=Number(date.slice(5,7));
   return (month>=4&&month<=9)===summer;
  });
  const profile=solarInHtml.dryDayProfiles[season];
  return profile.values.length===24&&profile.values.every(Number.isFinite)&&
   profile.totalDays===entries.length&&profile.dryDays===entries.filter(([,mm])=>mm===0).length;
 }),
 solarAnnualBaseline:Math.abs(solarInHtml.annualMeanPower-
  solarInHtml.annualEnergy/solarInHtml.pointCount)<1e-8,
 monthlyAnnualParity:monthSumGood,
 threeModelResults:['existing','quantumGravity','plasmaWave'].every(m=>!!snapInHtml?.annual?.[m]),
 threeDetailedCharts:template.includes('plot-control')&&template.includes('plot-temperature')&&template.includes('plot-scheduling'),
 captureFullYearSeries:capture.includes('buildChartSeries')&&capture.includes('combinedPlannedHpTopPower'),
 captureRebuildsOfflineHtml:capture.includes('build-sunopt-standalone-presentation.mjs'),
 embeddedSeriesCompatible:!seriesInHtml||(seriesInHtml.sourceResponseSha256===snapInHtml.source.responseSha256&&seriesInHtml.rows.length===35040),
};
const report={status:Object.values(checks).every(Boolean)?'PASS':'FAIL',checks,sourceSha256:snapshot.source.responseSha256,hasDetailedSeries:!!seriesInHtml,warning:seriesInHtml?null:'Die hochgeladene alte JSON enthält nur Monatsaggregate. Für drei echte 15-Minuten-Diagramme einmalig CAPTURE_SUNOPT_ANNUAL_SLIDE.cmd --replace ausführen.'};
const outbox=loc('updates/outbox');fs.mkdirSync(outbox,{recursive:true});
fs.writeFileSync(path.join(outbox,'sunopt-standalone-presentation-verify.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
if(report.status!=='PASS')process.exitCode=1;
