import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { validateTankRun } from './sunopt-tank-slide-data.mjs';

const folder = path.resolve(import.meta.dirname, '../../../public/presentation');
const snapshotFile = path.join(folder, 'tank-comparison-static.json');
const base = process.env.SUNOPT_BASE_URL ?? 'http://127.0.0.1:3000';
const runs = fs.existsSync(snapshotFile) ? JSON.parse(fs.readFileSync(snapshotFile, 'utf8')) : {};
const request = {
  heatPumpThermalPower: 7,
  pvPeakPower: 10,
  gridPriceChfPerKWh: 0.15,
  feedInPriceChfPerKWh: 0.05,
};

const args = process.argv.slice(2);
const selected = args.filter(arg => arg !== '--replace');
if (selected.some(arg => !['A', 'B'].includes(arg))) {
  throw new Error('Aufruf: node capture-sunopt-tank-slide.mjs [A] [B] [--replace]');
}

for (const causalPath of new Set(selected.length ? selected : ['A', 'B'])) {
  if (runs[causalPath] && !args.includes('--replace')) {
    validateTankRun(runs[causalPath], causalPath);
    continue;
  }
  const response = await fetch(new URL('/api/tank-comparison', base), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...request, causalPath }),
    signal: AbortSignal.timeout(300_000),
  });
  const raw = await response.text();
  if (!response.ok) throw new Error(`Tankvergleich ${causalPath} (HTTP ${response.status}): ${raw.slice(0, 500)}`);
  const run = JSON.parse(raw);
  if (!run.thermalReserve?.spaceHeating) {
    throw new Error('Die Tankvergleichs-API hat keine thermische Reserve geliefert. Läuft der aktuelle Serverstand?');
  }
  run.source = {
    api: '/api/tank-comparison',
    capturedAt: new Date().toISOString(),
    responseSha256: createHash('sha256').update(raw).digest('hex'),
  };
  validateTankRun(run, causalPath);
  runs[causalPath] = run;
  const temp = `${snapshotFile}.${process.pid}.tmp`;
  fs.writeFileSync(temp, JSON.stringify(runs, null, 2) + '\n');
  fs.renameSync(temp, snapshotFile);
  console.log(`Ansatz ${causalPath}: ${run.points.length} Tankgrössen gespeichert.`);
}

const build = spawnSync(process.execPath, [
  path.join(import.meta.dirname, 'build-sunopt-standalone-presentation.mjs'),
], { stdio: 'inherit' });
if (build.error) throw build.error;
process.exitCode = build.status ?? 1;
