// src/domain/combined-heat-demand.ts

export interface CombinedHeatDemandPoint {
  timestamp: string;

  hotWaterDemand: number;    // kWh_th
  spaceHeatingDemand: number; // kWh_th

  totalThermalDemand: number; // kWh_th
}

export type CombinedHeatDemand =
  CombinedHeatDemandPoint[];