// src/types/tank-comparison.ts

export interface TankComparisonPoint {
  storageVolume: number;

  gridImport: number;
  feedIn: number;
  selfConsumption: number;

  storageLosses: number;
  unmetThermalDemand: number;

  gridCost: number;
  feedInRevenue: number;
  netEnergyCost: number;

  /*
   * Plasma-Wave controller construction +
   * complete annual simulation for this tank.
   */
  simulationDurationMs: number;
}


export interface TankComparisonResponse {
  fake: boolean;

  year: number;

  heatPumpThermalPower: number;
  pvPeakPower: number;

  gridPriceChfPerKWh: number;
  feedInPriceChfPerKWh: number;

  /*
   * Weather + DHW generation + 15-minute resampling.
   */
  preparationDurationMs: number;

  /*
   * Wall-clock duration of the complete 28-tank sweep.
   */
  sweepDurationMs: number;

  /*
   * Complete API calculation.
   */
  totalDurationMs: number;

  averageSimulationDurationMs: number;
  minimumSimulationDurationMs: number;
  maximumSimulationDurationMs: number;

  points: TankComparisonPoint[];
}