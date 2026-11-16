// src/simulation/hot-water-demand-forecast.ts

/**
 * this is only a Debug plot Simulation to calculate 
 * Kurve "Bedarf Δt + 25 %" constant per Tank
 */

import type {
  HotWaterData,
} from "../domain/hot-water";

import type {
  ThermalStorage,
} from "../domain/thermal-storage";

import {
  calculatePassiveCoolingTimeHours,
  calculateStorageHeatLossAtTemperature,
} from "../domain/thermal-storage-calculation";


export interface RollingHotWaterDemandForecastPoint {

  /**
   * Raw hot-water demand inside the complete
   * moving forecast window.
   *
   * kWh_th
   */
  hotWaterDemandEnergy: number;


  /**
   * Estimated storage losses inside the complete
   * moving forecast window.
   *
   * kWh_th
   */
  storageLossEnergy: number;


  /**
   * Hot-water demand + storage losses
   * before safety margin.
   *
   * kWh_th
   */
  baseForecastEnergy: number;


  /**
   * Complete forecast including safety margin.
   *
   * kWh_th
   */
  forecastDemandEnergy: number;
}


export interface RollingHotWaterDemandForecast {

  /**
   * Physical passive cooling time
   * from targetTemperature to minTemperature.
   */
  passiveCoolingTimeHours: number;


  /**
   * Effective forecast horizon.
   *
   * Limited to maximumForecastHours.
   */
  planningHorizonHours: number;


  /**
   * Number of simulation intervals inside
   * the moving window.
   */
  forecastSteps: number;


  /**
   * Forecast result for every simulation point.
   *
   * Same array length as hotWaterData.
   */
  points:
    RollingHotWaterDemandForecastPoint[];
}


export interface RollingHotWaterDemandForecastInput {

  hotWaterData:
    HotWaterData;

  storage:
    ThermalStorage;

  timeStepHours:
    number;


  /**
   * Normally 75 °C.
   */
  targetTemperature?:
    number;


  /**
   * Normally 60 °C.
   */
  minimumTemperature?:
    number;


  /**
   * 1.25 = +25 %.
   */
  safetyFactor?:
    number;


  /**
   * Meteorological upper limit.
   *
   * Default:
   * 168 h = 7 days.
   */
  maximumForecastHours?:
    number;
}


export function calculateRollingHotWaterDemandForecast(
  input: RollingHotWaterDemandForecastInput,
): RollingHotWaterDemandForecast {

  const {
    hotWaterData,
    storage,
    timeStepHours,
  } = input;


  const targetTemperature =
    input.targetTemperature ??
    75;


  const minimumTemperature =
    input.minimumTemperature ??
    60;


  const safetyFactor =
    input.safetyFactor ??
    1.25;


  const maximumForecastHours =
    input.maximumForecastHours ??
    168;


  validateInput(
    hotWaterData,
    storage,
    timeStepHours,
    targetTemperature,
    minimumTemperature,
    safetyFactor,
    maximumForecastHours,
  );


  /*
   * ==================================================
   * 1. PHYSICAL STORAGE HORIZON
   * ==================================================
   *
   * Tank expiry time:
   *
   * 75 °C -> 60 °C
   */

  const passiveCoolingTimeHours =
    calculatePassiveCoolingTimeHours(
      storage,
      targetTemperature,
      minimumTemperature,
    );


  /*
   * Meteorological limit:
   *
   * Maximum 7 days by default.
   */

  const requestedPlanningHorizonHours =
    Math.min(
      passiveCoolingTimeHours,
      maximumForecastHours,
    );


  /*
   * Convert horizon into complete simulation steps.
   *
   * Example:
   *
   * 15 min:
   * timeStepHours = 0.25
   */

  const forecastSteps =
    Math.max(
      1,

      Math.min(
        hotWaterData.length,

        Math.floor(
          requestedPlanningHorizonHours /
          timeStepHours,
        ),
      ),
    );


  const planningHorizonHours =
    forecastSteps *
    timeStepHours;


  /*
   * ==================================================
   * 2. STORAGE LOSS PER INTERVAL
   * ==================================================
   *
   * For this first diagnostic implementation
   * we calculate conservatively at 75 °C.
   *
   * Later we can replace this with a virtual
   * cooling trajectory.
   */

  const storageLossPerStep =
    calculateStorageHeatLossAtTemperature(
      storage,
      targetTemperature,
      timeStepHours,
    );


  /*
   * ==================================================
   * 3. INITIAL MOVING WINDOW
   * ==================================================
   *
   * For point 0 we integrate:
   *
   * point 1 ... point forecastSteps
   *
   * We deliberately look FORWARD from the
   * current point.
   */

  let rollingHotWaterEnergy =
    0;


  for (
    let offset = 1;
    offset <= forecastSteps;
    offset++
  ) {

    const index =
      wrapIndex(
        offset,
        hotWaterData.length,
      );


    rollingHotWaterEnergy +=
      hotWaterData[index]
        .thermalEnergy;
  }


  /*
   * With the current conservative loss model,
   * the loss per time step is constant over
   * the moving horizon.
   */

  const rollingStorageLossEnergy =
    storageLossPerStep *
    forecastSteps;


  /*
   * ==================================================
   * 4. MOVING INTEGRAL
   * ==================================================
   *
   * Efficient O(n) rolling calculation.
   *
   * We do NOT recalculate the complete horizon
   * for every 15-minute point.
   */

  const points:
    RollingHotWaterDemandForecastPoint[] =
      [];


  for (
    let currentIndex = 0;
    currentIndex < hotWaterData.length;
    currentIndex++
  ) {

    const baseForecastEnergy =
      rollingHotWaterEnergy +
      rollingStorageLossEnergy;


    /*
     * User-defined safety margin:
     *
     * +25 % applied to:
     *
     * hot-water demand
     * + storage losses
     */

    const forecastDemandEnergy =
      baseForecastEnergy *
      safetyFactor;


    points.push({

      hotWaterDemandEnergy:
        rollingHotWaterEnergy,

      storageLossEnergy:
        rollingStorageLossEnergy,

      baseForecastEnergy,

      forecastDemandEnergy,
    });


    /*
     * Prepare next rolling window.
     *
     * Current window:
     *
     * i+1 ... i+n
     *
     * Next window:
     *
     * i+2 ... i+n+1
     */

    const outgoingIndex =
      wrapIndex(
        currentIndex + 1,
        hotWaterData.length,
      );


    const incomingIndex =
      wrapIndex(
        currentIndex +
          forecastSteps +
          1,
        hotWaterData.length,
      );


    rollingHotWaterEnergy -=
      hotWaterData[outgoingIndex]
        .thermalEnergy;


    rollingHotWaterEnergy +=
      hotWaterData[incomingIndex]
        .thermalEnergy;
  }


  return {

    passiveCoolingTimeHours,

    planningHorizonHours,

    forecastSteps,

    points,
  };
}


function validateInput(
  hotWaterData: HotWaterData,
  storage: ThermalStorage,
  timeStepHours: number,
  targetTemperature: number,
  minimumTemperature: number,
  safetyFactor: number,
  maximumForecastHours: number,
): void {

  if (
    hotWaterData.length === 0
  ) {
    throw new Error(
      "hotWaterData must not be empty.",
    );
  }


  if (
    !Number.isFinite(
      timeStepHours,
    ) ||
    timeStepHours <= 0
  ) {
    throw new Error(
      "timeStepHours must be a finite number > 0.",
    );
  }


  if (
    targetTemperature <=
    minimumTemperature
  ) {
    throw new Error(
      "targetTemperature must be above minimumTemperature.",
    );
  }


  if (
    targetTemperature >
    storage.maxTemperature
  ) {
    throw new Error(
      "targetTemperature cannot exceed storage.maxTemperature.",
    );
  }


  if (
    minimumTemperature <
    storage.minTemperature
  ) {
    throw new Error(
      "minimumTemperature cannot be below storage.minTemperature.",
    );
  }


  if (
    !Number.isFinite(
      safetyFactor,
    ) ||
    safetyFactor < 1
  ) {
    throw new Error(
      "safetyFactor must be a finite number >= 1.",
    );
  }


  if (
    !Number.isFinite(
      maximumForecastHours,
    ) ||
    maximumForecastHours <= 0
  ) {
    throw new Error(
      "maximumForecastHours must be a finite number > 0.",
    );
  }
}


function wrapIndex(
  index: number,
  length: number,
): number {

  return (
    (
      index %
      length
    ) +
    length
  ) %
  length;
}