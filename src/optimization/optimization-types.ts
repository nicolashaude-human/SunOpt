// src/optimization/optimization-types.ts

export type OptimizationTarget =
  | "grid-import"
  | "pv-self-consumption";


export interface StorageOptimizationRange {
  minVolume: number;
  maxVolume: number;
  step: number;
}


export interface StorageOptimizationVariant {
  storageVolume: number;

  gridImport: number;

  pvUsedByHeatPump: number;

  pvSurplus: number;

  heatPumpConsumption: number;

  storageLosses: number;

  unmetThermalDemand: number;

  pvSelfConsumptionRate: number;
}


export interface StorageOptimizationResult {
  target: OptimizationTarget;

  optimalStorageVolume: number;

  optimalVariant:
    StorageOptimizationVariant;

  variants:
    StorageOptimizationVariant[];
}