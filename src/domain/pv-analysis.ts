// src/domain/pv-analysis.ts

import type { WeatherData } from "./weather";

export interface PVAnalysis {
  annualEnergy: number;   // kWh
  specificYield: number; // kWh/kWp
  peakPowerObserved: number; // kW
}

export function analyzePVProduction(
  weatherData: WeatherData,
  installedPeakPower: number,
): PVAnalysis {

  if (installedPeakPower <= 0) {
    throw new Error(
      "Installed peak power must be greater than zero.",
    );
  }

  const annualEnergy = weatherData.reduce(
    (sum, point) => sum + point.pvPower,
    0,
  );

  const peakPowerObserved = weatherData.reduce(
    (max, point) =>
      Math.max(max, point.pvPower),
    0,
  );

  const specificYield =
    annualEnergy / installedPeakPower;

  return {
    annualEnergy,
    specificYield,
    peakPowerObserved,
  };
}