// src/simulation/plasma-wave-15min-model.ts

import type {
  HeatPump,
  HotWaterData,
  ThermalStorage,
  WeatherData,
} from "@/domain";

import {
  calculateHeatPumpStep,
  calculateStorageHeatLossAtTemperature,
} from "@/domain";


export interface PlasmaWave15MinutePoint {
  timestamp: string;
  hotWaterEnergy: number;
  protectedHotWaterEnergy: number;
  storageLossEnergy: number;
  loadEnergy: number;
  pvEnergy: number;
  pvAfterHouseholdEnergy: number;
  usablePvThermalEnergy: number;
  netEnergy: number;
  pvAvailable: boolean;
  pvStart: boolean;
  pvEnd: boolean;
}


export interface PlasmaWave15MinuteModel {
  timeStepHours: number;

  points: PlasmaWave15MinutePoint[];

  /*
   * Shared numeric forecast series.
   *
   * These arrays belong to exactly the same annual indices as points.
   * They are created ONCE and reused by both 48 h and 7-day kernels.
   */
  pvPowerByIndex: Float64Array;
  exportPowerByIndex: Float64Array;
  loadEnergyByIndex: Float64Array;
  pvAfterHouseholdEnergyByIndex: Float64Array;
  usablePvThermalEnergyByIndex: Float64Array;
}


export interface PlasmaWave15MinuteInput {
  weatherData: WeatherData;
  hotWaterData: HotWaterData;
  heatPump: HeatPump;
  storage: ThermalStorage;
  timeStepHours: number;
  householdReservePower: number;
  safetyFactor: number;
  lossReferenceTemperature: number;
}


export function calculatePlasmaWave15MinuteModel(
  input: PlasmaWave15MinuteInput,
): PlasmaWave15MinuteModel {

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


  if (timeStepHours <= 0) {
    throw new Error(
      "timeStepHours must be greater than zero.",
    );
  }


  const referenceHeatPumpStep = calculateHeatPumpStep(
    heatPump,
    heatPump.thermalPower * timeStepHours,
    timeStepHours,
  );


  const cop = referenceHeatPumpStep.cop;


  const storageLossEnergy = calculateStorageHeatLossAtTemperature(
    storage,
    lossReferenceTemperature,
    timeStepHours,
  );


  const maximumHeatPumpThermalEnergy = heatPump.thermalPower * timeStepHours;


  /*
   * ================================================================
   * SHARED NUMERIC FORECAST ARRAYS
   * ================================================================
   *
   * These arrays are filled during the SAME pass that creates points.
   *
   * No later adapter has to map/copy weather or demand data again.
   */
  const pvPowerByIndex = new Float64Array(weatherData.length);
  const exportPowerByIndex = new Float64Array(weatherData.length);
  const loadEnergyByIndex = new Float64Array(weatherData.length);
  const pvAfterHouseholdEnergyByIndex = new Float64Array(weatherData.length);
  const usablePvThermalEnergyByIndex = new Float64Array(weatherData.length);


  const points = weatherData.map(
    (weatherPoint, index): PlasmaWave15MinutePoint => {

      const hotWaterPoint = hotWaterData[index];

      if (!hotWaterPoint) {
        throw new Error(
          `Hot-water point missing at index ${index}.`,
        );
      }


      const hotWaterEnergy = hotWaterPoint.thermalEnergy;
      const protectedHotWaterEnergy = hotWaterEnergy * safetyFactor;
      const loadEnergy = protectedHotWaterEnergy;

      /*
       * pvPowerByIndex deliberately preserves the original weather
       * value because the 7-day kernel previously received exactly
       * weatherPoint.pvPower.
       */
      const rawPvPower = weatherPoint.pvPower;
      const pvPower = Math.max(0, rawPvPower);

      const pvEnergy = pvPower * timeStepHours;

      const pvAfterHouseholdPower = Math.max(
        0,
        rawPvPower - householdReservePower,
      );

      const pvAfterHouseholdEnergy =
        pvAfterHouseholdPower * timeStepHours;

      const theoreticalPvThermalEnergy =
        pvAfterHouseholdEnergy * cop;

      const usablePvThermalEnergy = Math.min(
        theoreticalPvThermalEnergy,
        maximumHeatPumpThermalEnergy,
      );

      const netEnergy = loadEnergy - usablePvThermalEnergy;

      const pvAvailable = pvAfterHouseholdEnergy > 0.01;


      /*
       * Fill shared typed arrays while all values are already hot.
       */
      pvPowerByIndex[index] = rawPvPower;
      exportPowerByIndex[index] = pvAfterHouseholdPower;
      loadEnergyByIndex[index] = loadEnergy;
      pvAfterHouseholdEnergyByIndex[index] = pvAfterHouseholdEnergy;
      usablePvThermalEnergyByIndex[index] = usablePvThermalEnergy;


      return {
        timestamp: weatherPoint.timestamp,
        hotWaterEnergy,
        protectedHotWaterEnergy,
        storageLossEnergy,
        loadEnergy,
        pvEnergy,
        pvAfterHouseholdEnergy,
        usablePvThermalEnergy,
        netEnergy,
        pvAvailable,
        pvStart: false,
        pvEnd: false,
      };
    },
  );


  /*
   * Mark PV-window transitions.
   */
  for (let index = 0; index < points.length; index++) {

    const current = points[index];

    if (!current) continue;

    const previous = index > 0 ? points[index - 1] : undefined;
    const next = index < points.length - 1 ? points[index + 1] : undefined;

    current.pvStart = current.pvAvailable && !previous?.pvAvailable;
    current.pvEnd = current.pvAvailable && !next?.pvAvailable;
  }


  return {
    timeStepHours,
    points,
    pvPowerByIndex,
    exportPowerByIndex,
    loadEnergyByIndex,
    pvAfterHouseholdEnergyByIndex,
    usablePvThermalEnergyByIndex,
  };
}