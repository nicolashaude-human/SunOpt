const metrics = ['gridImport', 'feedIn', 'selfConsumption', 'storageLosses', 'netEnergyCost', 'simulationDurationMs'];

export function validateTankRun(run, causalPath) {
  const fields = ['year', 'heatPumpThermalPower', 'pvPeakPower', 'gridPriceChfPerKWh',
    'feedInPriceChfPerKWh', 'sweepDurationMs', 'totalDurationMs'];
  if (run?.fake !== false || run.causalPath !== causalPath ||
      run.source?.api !== '/api/tank-comparison' ||
      !/^[0-9a-f]{64}$/.test(run.source.responseSha256) ||
      !fields.every(field => Number.isFinite(run[field])) ||
      run.points?.length !== 28 ||
      !run.points.every((point, index) => point.storageVolume === 300 + index * 100 &&
        metrics.every(metric => Number.isFinite(point[metric])))) {
    throw new Error(`Ungültiger Snapshot für Tankvergleich ${causalPath}.`);
  }
}
