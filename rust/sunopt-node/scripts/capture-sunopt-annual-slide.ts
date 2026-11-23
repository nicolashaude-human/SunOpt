// rust/sunopt-node/scripts/capture-sunopt-annual-slide.ts
// Einmalige echte Jahressimulation via der bestehenden Next.js-API.
// Danach wird nur ein unveraenderliches, statisches HTML/JSON erzeugt.
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { spawnSync } from "node:child_process";
import { HOT_WATER_TEST_CONFIG } from "../../../src/config/config";

const root = process.cwd(); // CAPTURE_SUNOPT_ANNUAL_SLIDE.cmd wechselt zuerst ins Projektverzeichnis.
if (!fs.existsSync(path.join(root, "src", "config", "config.ts"))) throw new Error("SunOpt-Projektverzeichnis nicht gefunden: " + root);
const outbox = path.join(root, "updates", "outbox");
const presentation = path.join(root, "public", "presentation");
fs.mkdirSync(outbox, { recursive: true });
fs.mkdirSync(presentation, { recursive: true });
const failedRun = path.join(outbox, "failed-run.txt");
const snapshotJson = path.join(presentation, "annual-simulation-static.json");
const snapshotHtml = path.join(presentation, "annual-simulation-static.html");
const detailFile = path.join(presentation, "annual-simulation-static-series.json");
const builderFile = path.join(root, "rust", "sunopt-node", "scripts", "build-sunopt-standalone-presentation.mjs");
const force = process.argv.includes("--replace");
const host = process.env.SUNOPT_BASE_URL ?? "http://127.0.0.1:3000";
const year = 2023;
const volume = 3000;
const models = ["existing", "quantumGravity", "plasmaWave"] as const;
const modelLabels: Record<string, string> = {
  existing: "Existing", quantumGravity: "Quantum Gravity", plasmaWave: "Plasma Wave",
};
const metricKeys = ["gridImportEnergy", "pvUsedByHeatPumpEnergy", "heatPumpEnergy", "storageLossEnergy", "unmetThermalDemand"] as const;
const colors = ["#FFB300", "#803E75", "#FF6800"];
const config = {
  postalCode: "8001", year,
  pvSystem: { peakPower: 10, azimuth: 0, inclination: 30, systemLoss: 0.14 },
  hotWaterDemand: { persons: 2, dailyConsumptionPerPerson: 50, coldWaterTemperature: 10, deliveryTemperature: 60 },
  heatPump: { thermalPower: 5, maxFlowTemperature: 75, copModel: { type: "constant", cop: 3 } },
  storage: { volume, minTemperature: 60, maxTemperature: 75, initialTemperature: 60,
    insulation: { thermalConductivity: 0.030, thickness: 0.10, cylinderHeight: 1.70, ambientTemperature: 20 } },
  control: { minimumReserveHours: 2, forecastHorizonHours: 72, pvTargetTemperature: 75,
    householdReservePower: HOT_WATER_TEST_CONFIG.householdReservePower,
    // Genau wie auf der Jahressimulation-Page: Prozent auf ganze Zahl gerundet.
    forecastSafetyFactor: 1 + Math.max(0, Math.round((HOT_WATER_TEST_CONFIG.forecastSafetyFactor - 1) * 100)) / 100 },
};

type Model = Record<string, number | null | undefined>;
type Point = { timestamp: string; existing: Model; quantumGravity: Model; plasmaWave: Model; required48hEnergy?:number; userForecastWaterlinePower?:number|null; noSunshineWaterlinePower?:number|null; effectiveWaterlinePower?:number|null };
type ApiResult = { year: number; storageVolume: number; simulationDurationMs: number;
  timeStepMinutes: number; dataPoints: number; data: Point[]; [key: string]: unknown };
type Metric = typeof metricKeys[number];
type Totals = Record<Metric, number> & { temperatureSum: number; temperatureSamples: number };
function empty(): Totals { return {gridImportEnergy:0,pvUsedByHeatPumpEnergy:0,heatPumpEnergy:0,storageLossEnergy:0,unmetThermalDemand:0,temperatureSum:0,temperatureSamples:0}; }
function accumulate(total: Totals, model: Model) {
  for (const k of metricKeys) {
    const v = model[k];
    if (typeof v !== "number" || !Number.isFinite(v)) throw new Error(`Ungueltiger Simulationswert: ${k}=${v}`);
    total[k] += v;
  }
  const temp = model.storageTemperature;
  if (typeof temp === "number" && Number.isFinite(temp)) {total.temperatureSum += temp;total.temperatureSamples++;}
}
const n = (value: number, digits=1) => value.toLocaleString("de-CH", {minimumFractionDigits:digits,maximumFractionDigits:digits});
const esc=(v:unknown)=>String(v).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c] as string));
function monthlySvg(months: Array<Record<string, Totals>>) {
  const w=1110,h=315,l=65,r=25,t=28,b=50;
  const plotH=h-t-b,plotW=w-l-r;
  const maximum=Math.max(1,...months.flatMap(m=>models.map(id=>m[id].gridImportEnergy)));
  const top=maximum*1.10;
  let bars="",ticks="";
  for (let i=0;i<=4;i++){
    const y=t+plotH*(1-i/4);
    ticks+=`<line x1="${l}" y1="${y.toFixed(1)}" x2="${w-r}" y2="${y.toFixed(1)}" stroke="#e4d6bf"/><text x="${l-9}" y="${(y+4).toFixed(1)}" text-anchor="end" font-size="12" fill="#66543d">${esc(n(top*i/4,0))}</text>`;
  }
  months.forEach((month,i)=>{
    const gw=plotW/12, bw=gw*0.24, gx=l+gw*i+gw*0.11;
    models.forEach((id,j)=>{
      const v=month[id].gridImportEnergy;
      const height=Math.max(0,v/top*plotH), x=gx+j*(bw+2), y=t+plotH-height;
      bars+=`<rect x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${bw.toFixed(2)}" height="${height.toFixed(2)}" fill="${colors[j]}" rx="2"><title>${esc(modelLabels[id])}, ${i+1}: ${esc(n(v,2))} kWh</title></rect>`;
    });
    bars+=`<text x="${(l+(i+.5)*gw).toFixed(1)}" y="${h-20}" text-anchor="middle" font-size="14" fill="#4d3b2e">${["Jan","Feb","Mär","Apr","Mai","Jun","Jul","Aug","Sep","Okt","Nov","Dez"][i]}</text>`;
  });
  return `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Monatlicher Netzbezug der drei Simulationsmodelle in Kilowattstunden">${ticks}${bars}<text x="18" y="${t+plotH/2}" transform="rotate(-90 18 ${t+plotH/2})" text-anchor="middle" fill="#66543d" font-size="13">Netzbezug (kWh)</text></svg>`;
}
function renderHtml(snapshot: ReturnType<typeof buildSnapshot>): string {
 const {annual,months,source,parameters}=snapshot;
 const svg=monthlySvg(months);
 const tr = models.map((id,i)=>{
   const a=annual[id];
   return `<tr><td><span class="key" style="background:${colors[i]}"></span>${esc(modelLabels[id])}</td><td>${esc(n(a.gridImportEnergy,2))}</td><td>${esc(n(a.pvUsedByHeatPumpEnergy,2))}</td><td>${esc(n(a.heatPumpEnergy,2))}</td><td>${esc(n(a.storageLossEnergy,2))}</td></tr>`;
 }).join("");
 return `<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SunOpt Jahressimulation ${year} – statischer Snapshot</title><style>
 :root{font-family:Arial,sans-serif;color:#2f281e;background:#fffaf2}*{box-sizing:border-box}body{margin:0;padding:20px 24px}h1{margin:4px 0 5px;font-size:25px}h2{font-size:17px;margin:0 0 10px}.meta{color:#675640;font-size:13px;line-height:1.5}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:18px 0}.box,.chart{border:1px solid #d8c5a5;border-radius:12px;background:#fff;padding:12px}.box strong{display:block;font-size:21px;margin-top:5px}.box span{font-size:12px;color:#66543d}.chart svg{display:block;width:100%;height:auto;max-height:315px}.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:13px;margin:8px 0}.key{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:7px 10px;text-align:right;border-bottom:1px solid #eadfcf}td:first-child,th:first-child{text-align:left}.tiny{color:#675640;font-size:11px;margin-top:9px;line-height:1.5}footer{padding-top:12px;font-size:11px;color:#675640} @media(max-width:700px){body{padding:12px}.grid{grid-template-columns:1fr 1fr}th,td{font-size:11px;padding:5px}}
 </style></head><body><header><div class="meta">Folie 2 · Einmalig berechnete Ergebnisse · Keine Laufzeitberechnung in der Präsentation</div><h1>Jahressimulation ${year}</h1><div class="meta">PLZ ${esc(parameters.postalCode)} · PV ${parameters.pvSystem.peakPower} kWp · Wärmepumpe ${parameters.heatPump.thermalPower} kW<sub>th</sub> · Speicher ${parameters.storage.volume} l · ${source.dataPoints} Messpunkte (${esc(source.timeStepMinutes)} min)</div></header>
 <div class="grid"><div class="box"><span>Netzbezug · Existing</span><strong>${esc(n(annual.existing.gridImportEnergy,1))} kWh</strong></div><div class="box"><span>PV → Wärmepumpe · Existing</span><strong>${esc(n(annual.existing.pvUsedByHeatPumpEnergy,1))} kWh</strong></div><div class="box"><span>Strombedarf Wärmepumpe · Existing</span><strong>${esc(n(annual.existing.heatPumpEnergy,1))} kWh</strong></div><div class="box"><span>Rechenzeit des einmaligen Jahreslaufs</span><strong>${esc(n(source.simulationDurationMs/1000,2))} s</strong></div></div>
 <div class="chart"><h2>Netzbezug im Jahresverlauf</h2><div class="legend">${models.map((id,i)=>`<span><span class="key" style="background:${colors[i]}"></span>${esc(modelLabels[id])}</span>`).join("")}</div>${svg}</div><div class="chart" style="margin-top:12px"><h2>Jahresbilanz der Modelle · kWh</h2><table><thead><tr><th>Modell</th><th>Netzbezug el.</th><th>PV → WP el.</th><th>WP el.</th><th>Verluste th.</th></tr></thead><tbody>${tr}</tbody></table><p class="tiny">Die Werte wurden einmalig über /api/debug/hot-water berechnet und als statische Datei eingefroren. Die Darstellung führt weder eine neue Simulation noch eine Optimierung aus.</p></div><footer>Erstellt ${esc(source.createdAt)} · Quellantwort SHA-256 ${esc(source.responseSha256)} · Datenbasis ${esc(source.api)} · Modell: ${esc(source.modelSet)}</footer></body></html>`;
}
function buildSnapshot(data: ApiResult, rawHash: string) {
 const months: Array<Record<string, Totals>> = Array.from({length:12},()=>Object.fromEntries(models.map(id=>[id,empty()])) as Record<string,Totals>);
 const annual = Object.fromEntries(models.map(id=>[id,empty()])) as Record<typeof models[number],Totals>;
 for (const entry of data.data) {
   if (typeof entry.timestamp !== "string" || !/^2023-\d{2}-\d{2}/.test(entry.timestamp)) throw new Error("Ungültiges Datum oder nicht 2023: " + String(entry.timestamp));
   const idx=Number(entry.timestamp.slice(5,7))-1;
   if(!Number.isInteger(idx)||idx<0||idx>11)throw new Error("Monat fehlerhaft: "+entry.timestamp);
   for(const id of models){
     const model=entry[id]; if(!model || typeof model!=="object")throw new Error("Fehlende Modellergebnisse: "+id);
     accumulate(annual[id],model);accumulate(months[idx][id],model);
   }
 }
 const source={createdAt:new Date().toISOString(),api:"POST /api/debug/hot-water",responseSha256:rawHash,year:data.year,storageVolume:data.storageVolume,dataPoints:data.data.length,timeStepMinutes:data.timeStepMinutes,simulationDurationMs:data.simulationDurationMs,modelSet:models.join(", ")};
 return { schema:"sunopt-presentation-year-static-v1",source,parameters:config,annual,months };
}

// Fields mirror the THREE TypeScript Jahressimulation diagrams:
// 1) Control signal and energy, 2) Storage temperature,
// 3) PV export and HP scheduling. Full 15-minute source values,
// without rounding, approximation or numerical resimulation.
function buildChartSeries(data: ApiResult, rawHash: string) {
  const shared = [
    "required48hEnergy", "userForecastWaterlinePower",
    "noSunshineWaterlinePower", "effectiveWaterlinePower",
    "combinedPlannedHpTopPower",
  ];
  const fields = [
    "controlSignal", "heatPumpEnergy", "storageEnergy", "pvEnergy",
    "storageTemperature", "exportBeforeHpPower", "exportAfterHpPower",
    "heatPumpElectricalPower", "waterlineHeatPumpElectricalPower",
  ];
  const columns = [
    ...shared,
    ...models.flatMap(id => fields.map(field => id + "." + field)),
  ];
  const timestamps: string[] = [];
  const rows: Array<Array<number | null>> = [];
  for (const point of data.data) {
    timestamps.push(point.timestamp);
    const plasma = point.plasmaWave as Model;
    const waterline = point.effectiveWaterlinePower;
    const scheduled = plasma.waterlineHeatPumpElectricalPower;
    const toNumber = (value: unknown) =>
      typeof value === "number" && Number.isFinite(value) ? value : null;
    const sharedVals = [
      toNumber(point.required48hEnergy),
      toNumber(point.userForecastWaterlinePower),
      toNumber(point.noSunshineWaterlinePower),
      toNumber(waterline),
      toNumber(waterline) === null || toNumber(scheduled) === null
        ? null : Number(waterline) + Number(scheduled),
    ];
    const vals = [...sharedVals];
    for (const id of models) {
      for (const field of fields) {
        vals.push(toNumber(point[id][field]));
      }
    }
    rows.push(vals);
  }
  return {
    schema: "sunopt-presentation-year-series-v1",
    sourceResponseSha256: rawHash,
    columns,
    timestamps,
    rows,
    note: "Unmodified full-year TypeScript simulation results; chart rendering only, no simulation",
  };
}

async function main() {
 if ((fs.existsSync(snapshotJson)||fs.existsSync(snapshotHtml))&&!force) throw new Error("Ein statischer Jahres-Snapshot existiert bereits. Für bewusstes Überschreiben --replace angeben.");
 const url=new URL("/api/debug/hot-water",host);
 console.log("[SunOpt] Einmalige echte Jahresberechnung läuft: " + url + " (2023, 3000 l, PLZ 8001)");
 console.log("[SunOpt] Folie 2 führt später keine Berechnung aus.");
 const started=performance.now();
 const res=await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(config)});
 const raw=await res.text();
 if(!res.ok)throw new Error("API HTTP "+res.status+": "+raw.slice(0,1500));
 const data=JSON.parse(raw) as ApiResult;
 if(data.year!==year||data.storageVolume!==volume||!Array.isArray(data.data)||data.data.length<30000)throw new Error("Jahresresultat unvollständig oder Parameter abweichend");
 if(!Number.isFinite(data.simulationDurationMs))throw new Error("Simulationszeit fehlt");
 const hash=crypto.createHash("sha256").update(raw).digest("hex");
 const snapshot=buildSnapshot(data,hash);
 const series=buildChartSeries(data,hash);
 const html=renderHtml(snapshot);
 // Einmalige, atomare Ablage; Zwischenstand bei Fehler nicht anzeigen.
 for(const [dest,text] of [[snapshotHtml,html],[snapshotJson,JSON.stringify(snapshot,null,2)+"\n"],[detailFile,JSON.stringify(series)+"\n"]] as const){
   const temp=dest+".tmp-"+process.pid;fs.writeFileSync(temp,text,"utf8");fs.renameSync(temp,dest);
 }
 const build=spawnSync(process.execPath,[builderFile],{cwd:root,encoding:"utf8",windowsHide:true});
 if (build.status !== 0) throw new Error("Standalone HTML5-Build fehlgeschlagen: "+(build.stderr||build.stdout));
 const report={status:"PASS",captureWallMs:performance.now()-started,simulationDurationMs:data.simulationDurationMs,dataPoints:data.data.length,year,volume,staticJson:snapshotJson,staticHtml:snapshotHtml,staticSeries:detailFile,standaloneHtml:path.join(presentation,"index.html"),sha256:hash};
 const reportFile=path.join(outbox,"sunopt-presentation-annual-snapshot.json");
 fs.writeFileSync(reportFile,JSON.stringify(report,null,2)+"\n");
 console.log(JSON.stringify(report,null,2));
 console.log("[SunOpt] Fertig: public/presentation/index.html ist EINE standalone HTML5-Datei mit eingebetteten Zeitreihen.");
}
main().catch(error=>{const message=String(error instanceof Error?error.stack:error);console.error(message);fs.writeFileSync(failedRun,"Annual slide capture failed: "+message+"\n");process.exitCode=1;});
