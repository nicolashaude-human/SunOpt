// src/simulation/plasma-wave-historical-series-cache.ts

import type {
  HeatPump,
  HotWaterData,
  ThermalStorage,
  WeatherData,
} from "@/domain";

import {
  calculateHeatPumpStep,
} from "@/domain";

import {
  calculatePlasmaWave15MinuteModel,
} from "./plasma-wave-15min-model";


export interface PlasmaWaveHistoricalSeries {
  length: number;

  timeStepHours: number;
  householdReservePower: number;
  safetyFactor: number;

  cop: number;

  pvPowerByIndex: Float64Array;
  exportPowerByIndex: Float64Array;
  loadEnergyByIndex: Float64Array;
  pvAfterHouseholdEnergyByIndex: Float64Array;
  usablePvThermalEnergyByIndex: Float64Array;

  pvAvailableByIndex: Uint8Array;
  pvStartByIndex: Uint8Array;
  pvEndByIndex: Uint8Array;
}


export interface PlasmaWaveHistoricalSeriesInput {
  weatherData: WeatherData;
  hotWaterData: HotWaterData;

  heatPump: HeatPump;

  /*
   * Only required on a CACHE MISS because
   * calculatePlasmaWave15MinuteModel currently also calculates
   * tank-dependent storageLossEnergy.
   *
   * storage is deliberately NOT part of the cache identity.
   */
  storage: ThermalStorage;

  timeStepHours: number;
  householdReservePower: number;
  safetyFactor: number;

  lossReferenceTemperature: number;
}


interface CacheEntry {
  hotWaterData: HotWaterData;
  heatPump: HeatPump;

  timeStepHours: number;
  householdReservePower: number;
  safetyFactor: number;

  series: PlasmaWaveHistoricalSeries;
}


/*
 * weatherData is the first identity key.
 *
 * For one tank sweep SunOpt passes the same historical arrays to all
 * tank simulations, so all tanks hit the same cache entry.
 */
let cacheByWeatherData =
  new WeakMap<object, CacheEntry[]>();


let hits = 0;
let misses = 0;


export function getPlasmaWaveHistoricalSeries(
  input: PlasmaWaveHistoricalSeriesInput,
): PlasmaWaveHistoricalSeries {

  const {
    weatherData,
    hotWaterData,
    heatPump,
    storage,
    timeStepHours,
    householdReservePower,
    safetyFactor,
    lossReferenceTemperature,
  } = input;


  if (weatherData.length !== hotWaterData.length) {
    throw new Error(
      "Weather data and hot-water data must have the same length.",
    );
  }


  const weatherKey = weatherData as unknown as object;

  let entries = cacheByWeatherData.get(weatherKey);

  if (!entries) {
    entries = [];
    cacheByWeatherData.set(weatherKey, entries);
  }


  const cached = entries.find(
    (entry) =>
      entry.hotWaterData === hotWaterData &&
      entry.heatPump === heatPump &&
      entry.timeStepHours === timeStepHours &&
      entry.householdReservePower === householdReservePower &&
      entry.safetyFactor === safetyFactor,
  );


  if (cached) {
    hits += 1;
    return cached.series;
  }


  misses += 1;


  /*
   * CACHE MISS
   *
   * Build the physical 15-minute model exactly once.
   *
   * We only retain the tank-independent arrays below.
   * storageLossEnergy remains outside this cache.
   */
  const segmentModel =
    calculatePlasmaWave15MinuteModel({
      weatherData,
      hotWaterData,
      heatPump,
      storage,
      timeStepHours,
      householdReservePower,
      safetyFactor,
      lossReferenceTemperature,
    });


  const referenceHeatPumpStep =
    calculateHeatPumpStep(
      heatPump,
      heatPump.thermalPower * timeStepHours,
      timeStepHours,
    );


  const cop =
    referenceHeatPumpStep.cop;


  if (cop <= 0) {
    throw new Error(
      "Heat-pump COP must be greater than zero.",
    );
  }


  const pvAvailableByIndex =
    Uint8Array.from(
      segmentModel.points,
      (point) => point.pvAvailable ? 1 : 0,
    );


  const pvStartByIndex =
    Uint8Array.from(
      segmentModel.points,
      (point) => point.pvStart ? 1 : 0,
    );


  const pvEndByIndex =
    Uint8Array.from(
      segmentModel.points,
      (point) => point.pvEnd ? 1 : 0,
    );


  const series: PlasmaWaveHistoricalSeries = {
    length: weatherData.length,

    timeStepHours,
    householdReservePower,
    safetyFactor,

    cop,

    /*
     * These Float64Arrays are the arrays you already added in step 1.
     *
     * We retain the same array objects.
     * No Float64Array.from(...) copy here.
     */
    pvPowerByIndex:
      segmentModel.pvPowerByIndex,

    exportPowerByIndex:
      segmentModel.exportPowerByIndex,

    loadEnergyByIndex:
      segmentModel.loadEnergyByIndex,

    pvAfterHouseholdEnergyByIndex:
      segmentModel.pvAfterHouseholdEnergyByIndex,

    usablePvThermalEnergyByIndex:
      segmentModel.usablePvThermalEnergyByIndex,

    pvAvailableByIndex,
    pvStartByIndex,
    pvEndByIndex,
  };


  entries.push({
    hotWaterData,
    heatPump,
    timeStepHours,
    householdReservePower,
    safetyFactor,
    series,
  });


  return series;
}


export function getPlasmaWaveHistoricalSeriesCacheStats() {
  return {
    hits,
    misses,
    entries:
      hits + misses,
  };
}


export function clearPlasmaWaveHistoricalSeriesCache(): void {
  cacheByWeatherData =
    new WeakMap<object, CacheEntry[]>();

  hits = 0;
  misses = 0;
}