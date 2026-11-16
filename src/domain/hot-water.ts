// src/domain/hot-water.ts

export interface HotWaterDemand {
  persons: number;

  dailyConsumptionPerPerson: number;
  // L/(person day) bei deliveryTemperature

  coldWaterTemperature: number;
  // °C

  deliveryTemperature: number;
  // °C, gewünschte Brauchwassertemperatur,
  // z.B. 60 °C
}


export interface HotWaterProfile {
  hourlyFractions: number[];
}


export interface DailyHotWaterDataPoint {
  hour: number;

  waterVolume: number;
  // L bei deliveryTemperature

  thermalEnergy: number;
  // kWh_th relativ zu coldWaterTemperature
}


export type DailyHotWaterData =
  DailyHotWaterDataPoint[];


export interface HotWaterDataPoint {
  timestamp: string;

  hour: number;

  waterVolume: number;

  thermalEnergy: number;
}


export type HotWaterData =
  HotWaterDataPoint[];