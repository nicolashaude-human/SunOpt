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


export interface TankComparisonResponse {
  fake: boolean;

  /*
   * Causal solver path actually selected for this result.
   *
   * A = consolidated A1+A2 Rust/Cranelift native controller (CPU workers)
   * B = B14P FP32 CUDA production path with FP64 fallback
   * The old exact FP64 CUDA reference remains available in the
   * native GPU lockstep engine and dedicated GPU endpoint.
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

  approachA?: {
    backend: string;
    workerCount: number;
    nativeScopes: Array<{
      scopeId: string;
      calls: number;
      nativeCalls: number;
      fallbackCalls: number;
      failures: number;
    }>;
  };
}