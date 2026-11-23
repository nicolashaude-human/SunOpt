// src/types/tank-comparison.ts

export type TankComparisonCausalPath =
  | "A"
  | "B";


export interface TankComparisonRequest {
  heatPumpThermalPower: number;
  pvPeakPower: number;

  gridPriceChfPerKWh: number;
  feedInPriceChfPerKWh: number;

  causalPath?:
    TankComparisonCausalPath;
}


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


export interface TankComparisonThermalReserve {
  horizonHours: number;
  maxHotWaterDemandKWh: number;
  usableCapacityKWhPerLitre: number;
  minimumTemperatureC: number;
  maximumTemperatureC: number;
  spaceHeating?: {
    peakWeekShare: number;
    indoorTemperatureC: number;
    heatingLimitTemperatureC: number;
  };
}

export interface TankComparisonResponse {
  fake: boolean;

  /*
   * Causal solver path actually selected for this result.
   *
   * A = exact FP64 reference path
   * B = B14P FP32 production path with FP64 fallback
   */
  causalPath:
    TankComparisonCausalPath;

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
  thermalReserve?: TankComparisonThermalReserve;
}
