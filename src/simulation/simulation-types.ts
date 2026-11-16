// src/simulation/simulation-types.ts

export interface EnergyBalance {
  pvProduction: number;       // kWh_el
  pvUsedByHeatPump: number;   // kWh_el
  gridImport: number;         // kWh_el
  pvSurplus: number;          // kWh_el

  heatPumpConsumption: number; // kWh_el
  thermalProduction: number;   // kWh_th

  thermalDemand: number;       // kWh_th
  unmetThermalDemand: number;  // kWh_th

  storageLosses: number;       // kWh_th
}