// src/domain/space-heating.ts

export interface SpaceHeatingDataPoint {
  timestamp: string;

  outdoorTemperature: number; // °C

  thermalEnergy: number; // kWh_th
}

export type SpaceHeatingData =
  SpaceHeatingDataPoint[];