// src/domain/hot-water-forecast-control.ts

import type {
  HotWaterControl,
} from "./hot-water-control";

import type {
  HotWaterData,
} from "./hot-water";

import type {
  HeatPump,
} from "./heat-pump";

import type {
  ThermalStorage,
} from "./thermal-storage";

import type {
  WeatherData,
} from "./weather";


import {
  calculateHeatPumpStep,
} from "./heat-pump-calculation";


import {
  calculatePassiveCoolingTimeHours,
  calculateStorageEnergyAtTemperature,
  calculateStorageHeatLossAtTemperature,
} from "./thermal-storage-calculation";


/*
 * Maximum useful weather forecast.
 *
 * 7 days = 168 h.
 *
 * The actual planning horizon can be shorter
 * because the tank itself may lose its useful
 * 75 -> 60 °C energy sooner.
 */
const MAX_WEATHER_FORECAST_HOURS =
  7 * 24;


export interface HotWaterForecastTarget {

  /**
   * Storage target after charging in the
   * current interval.
   *
   * kWh_th above minTemperature.
   */
  targetEnergy: number;


  /**
   * Short-term reserve.
   *
   * Currently returned mainly for debugging.
   */
  minimumReserveEnergy: number;


  /**
   * Energy which should be stored according
   * to the rolling forecast.
   */
  forecastRequiredEnergy: number;


  /**
   * Maximum usable storage energy between
   * minTemperature and pvTargetTemperature.
   */
  targetCapacity: number;


  /**
   * Passive cooling time from target
   * temperature to minimum temperature.
   */
  passiveCoolingTimeHours: number;


  /**
   * Actual forecast horizon used by this tank.
   */
  planningHorizonHours: number;


  /**
   * Integral of forecast DHW demand including
   * the safety factor.
   */
  forecastDemandEnergy: number;


  /**
   * Integral of forecast storage losses.
   */
  forecastStorageLossEnergy: number;


  /**
   * Integral of PV electrical energy available
   * above the household reserve.
   */
  forecastPvElectricalEnergy: number;


  /**
   * Maximum thermal energy which the heat pump
   * could produce from that PV within the
   * planning horizon.
   */
  forecastPvThermalEnergy: number;
}


export interface HotWaterForecastInput {

  currentIndex: number;

  weatherData: WeatherData;

  hotWaterData: HotWaterData;

  heatPump: HeatPump;

  storage: ThermalStorage;

  control: HotWaterControl;

  timeStepHours?: number;
}


interface ForecastStep {

  thermalDemand: number;

  storageLoss: number;

  availablePvElectricalEnergy: number;

  availablePvThermalEnergy: number;
}


export function calculateHotWaterForecastTarget(
  input: HotWaterForecastInput,
): HotWaterForecastTarget {

  const timeStepHours =
    input.timeStepHours ?? 1;


  validateInput(
    input,
    timeStepHours,
  );


  const {
    currentIndex,
    weatherData,
    hotWaterData,
    heatPump,
    storage,
    control,
  } = input;


  /*
   * ==================================================
   * 1. STORAGE CAPACITY
   * ==================================================
   *
   * Energy which can be stored between:
   *
   * minTemperature
   * and
   * pvTargetTemperature.
   */

  const targetCapacity =
    calculateStorageEnergyAtTemperature(
      storage,
      control.pvTargetTemperature,
    );


  /*
   * ==================================================
   * 2. PASSIVE STORAGE EXPIRY TIME
   * ==================================================
   *
   * How long does the tank need to cool from
   *
   * pvTargetTemperature
   * to
   * minTemperature
   *
   * without DHW consumption?
   */

  const passiveCoolingTimeHours =
    calculatePassiveCoolingTimeHours(
      storage,
      control.pvTargetTemperature,
      storage.minTemperature,
    );


  /*
   * ==================================================
   * 3. EFFECTIVE PLANNING HORIZON
   * ==================================================
   *
   * We never look farther than:
   *
   * - passive storage lifetime
   * - 7-day weather forecast
   * - available annual simulation data
   */

  const availableDataHours =
    weatherData.length *
    timeStepHours;


  const planningHorizonHours =
    Math.min(
      passiveCoolingTimeHours,
      MAX_WEATHER_FORECAST_HOURS,
      availableDataHours,
    );


  const forecastSteps =
    Math.max(
      1,

      Math.floor(
        planningHorizonHours /
        timeStepHours,
      ),
    );


  /*
   * ==================================================
   * 4. HEAT-PUMP COP
   * ==================================================
   *
   * Current model still uses the COP returned by
   * calculateHeatPumpStep().
   *
   * Later this can become temperature dependent.
   */

  const referenceHeatPumpStep =
    calculateHeatPumpStep(
      heatPump,
      heatPump.thermalPower *
        timeStepHours,
      timeStepHours,
    );


  const cop =
    referenceHeatPumpStep.cop;


  /*
   * ==================================================
   * 5. SAFETY FACTOR
   * ==================================================
   *
   * The control object currently contains the
   * configurable value.
   *
   * Set this to 1.25 in the control configuration
   * for the desired +25 % DHW margin.
   *
   * Storage losses themselves are NOT multiplied
   * by the safety factor.
   */

  const demandSafetyFactor =
    control.forecastSafetyFactor;


  /*
   * ==================================================
   * 6. SHORT-TERM RESERVE
   * ==================================================
   *
   * Kept for diagnostics.
   *
   * IMPORTANT:
   * this value must NOT by itself authorize
   * grid charging.
   */

  const reserveSteps =
    Math.min(
      forecastSteps,

      Math.ceil(
        control.minimumReserveHours /
        timeStepHours,
      ),
    );


  let minimumReserveEnergy =
    0;


  for (
    let offset = 1;
    offset <= reserveSteps;
    offset++
  ) {

    const index =
      wrapIndex(
        currentIndex + offset,
        hotWaterData.length,
      );


    const thermalDemand =
      hotWaterData[index]
        .thermalEnergy *
      demandSafetyFactor;


    const storageLoss =
      calculateStorageHeatLossAtTemperature(
        storage,
        control.pvTargetTemperature,
        timeStepHours,
      );


    minimumReserveEnergy +=
      thermalDemand +
      storageLoss;
  }


  minimumReserveEnergy =
    Math.min(
      minimumReserveEnergy,
      targetCapacity,
    );


  /*
   * ==================================================
   * 7. BUILD ROLLING FORECAST
   * ==================================================
   *
   * For every future interval:
   *
   *   demand + loss
   *
   * versus
   *
   *   PV above household reserve
   *   converted through the heat pump.
   *
   * This is the discrete equivalent of the
   * rolling energy integrals.
   */

  const forecast:
    ForecastStep[] = [];


  let forecastDemandEnergy =
    0;


  let forecastStorageLossEnergy =
    0;


  let forecastPvElectricalEnergy =
    0;


  let forecastPvThermalEnergy =
    0;


  for (
    let offset = 1;
    offset <= forecastSteps;
    offset++
  ) {

    const index =
      wrapIndex(
        currentIndex + offset,
        weatherData.length,
      );


    const weatherPoint =
      weatherData[index];


    const hotWaterPoint =
      hotWaterData[index];


    /*
     * ------------------------------------------
     * DHW demand + safety margin
     * ------------------------------------------
     */

    const thermalDemand =
      hotWaterPoint.thermalEnergy *
      demandSafetyFactor;


    /*
     * ------------------------------------------
     * Storage loss
     * ------------------------------------------
     *
     * For now conservatively evaluated at the
     * target temperature.
     *
     * Later we can use a virtual forecast tank
     * temperature here.
     */

    const storageLoss =
      calculateStorageHeatLossAtTemperature(
        storage,
        control.pvTargetTemperature,
        timeStepHours,
      );


    /*
     * ------------------------------------------
     * PV above household reserve
     * ------------------------------------------
     */

    const availablePvPower =
      Math.max(
        0,
        weatherPoint.pvPower -
          control.householdReservePower,
      );


    const availablePvElectricalEnergy =
      availablePvPower *
      timeStepHours;


    /*
     * ------------------------------------------
     * Maximum heat-pump energy from PV
     * ------------------------------------------
     *
     * Limited by:
     *
     * 1. available PV
     * 2. COP
     * 3. heat-pump thermal power
     */

    const availablePvThermalEnergy =
      Math.min(
        heatPump.thermalPower *
          timeStepHours,

        availablePvElectricalEnergy *
          cop,
      );


    forecast.push({
      thermalDemand,

      storageLoss,

      availablePvElectricalEnergy,

      availablePvThermalEnergy,
    });


    /*
     * Rolling-integral diagnostics.
     */

    forecastDemandEnergy +=
      thermalDemand;


    forecastStorageLossEnergy +=
      storageLoss;


    forecastPvElectricalEnergy +=
      availablePvElectricalEnergy;


    forecastPvThermalEnergy +=
      availablePvThermalEnergy;
  }


  /*
   * ==================================================
   * 8. BUILD THE STORAGE GUIDE CURVE
   * ==================================================
   *
   * This is the crucial part.
   *
   * We calculate BACKWARDS, but the value we want
   * is:
   *
   * "How much energy must be in the tank AFTER
   * charging in the CURRENT interval?"
   *
   *
   * For future interval i:
   *
   * requiredBefore =
   *
   * demand_i
   * + loss_i
   * + max(
   *     0,
   *     futureRequired
   *     - futurePVCharge_i
   *   )
   *
   *
   * Important causal property:
   *
   * PV in interval i can help with requirements
   * AFTER interval i,
   *
   * but it cannot supply demand which occurred
   * BEFORE interval i.
   *
   *
   * Also important:
   *
   * PV in the CURRENT interval is NOT subtracted
   * here.
   *
   * The current PV is precisely what the controller
   * should use to charge towards this target.
   */

  let requiredEnergyAfterCurrentInterval =
    0;


  for (
    let offset = forecast.length - 1;
    offset >= 0;
    offset--
  ) {

    const step =
      forecast[offset];


    const intervalDemand =
      step.thermalDemand +
      step.storageLoss;


    /*
     * Future PV may satisfy future storage demand.
     *
     * But demand in this interval must already be
     * available before this interval's PV charge
     * can help later intervals.
     */

    requiredEnergyAfterCurrentInterval =
      intervalDemand +
      Math.max(
        0,

        requiredEnergyAfterCurrentInterval -
          step.availablePvThermalEnergy,
      );


    /*
     * Physical storage limit.
     */

    requiredEnergyAfterCurrentInterval =
      Math.min(
        targetCapacity,
        requiredEnergyAfterCurrentInterval,
      );
  }


  /*
   * ==================================================
   * 9. FORECAST STORAGE TARGET
   * ==================================================
   *
   * This is the actual precharge guide.
   *
   * During today's PV window this value should rise
   * as the remaining future PV opportunity becomes
   * smaller.
   *
   * That should cause the heat pump to fill the
   * tank BEFORE the evening/night/morning demand.
   */

  const forecastRequiredEnergy =
    Math.min(
      targetCapacity,
      Math.max(
        0,
        requiredEnergyAfterCurrentInterval,
      ),
    );


  /*
   * We deliberately use the forecast target here.
   *
   * minimumReserveEnergy remains diagnostic.
   *
   * Emergency grid heating below 60 °C belongs to
   * hot-water-simulation.ts and must not be caused
   * merely by this target.
   */

  const targetEnergy =
    forecastRequiredEnergy;


  return {
    targetEnergy,

    minimumReserveEnergy,

    forecastRequiredEnergy,

    targetCapacity,

    passiveCoolingTimeHours,

    planningHorizonHours,

    forecastDemandEnergy,

    forecastStorageLossEnergy,

    forecastPvElectricalEnergy,

    forecastPvThermalEnergy,
  };
}


function validateInput(
  input: HotWaterForecastInput,
  timeStepHours: number,
): void {

  if (
    input.weatherData.length !==
    input.hotWaterData.length
  ) {
    throw new Error(
      "Weather data and hot-water data must have the same length.",
    );
  }


  if (
    input.weatherData.length === 0
  ) {
    throw new Error(
      "Forecast requires weather data.",
    );
  }


  if (
    !Number.isFinite(
      timeStepHours,
    ) ||
    timeStepHours <= 0
  ) {
    throw new Error(
      "timeStepHours must be a finite number greater than zero.",
    );
  }


  if (
    !Number.isFinite(
      input.control.minimumReserveHours,
    ) ||
    input.control.minimumReserveHours < 0
  ) {
    throw new Error(
      "minimumReserveHours must be a finite number >= 0.",
    );
  }


  if (
    !Number.isFinite(
      input.control.forecastSafetyFactor,
    ) ||
    input.control.forecastSafetyFactor < 1
  ) {
    throw new Error(
      "forecastSafetyFactor must be a finite number >= 1.",
    );
  }


  if (
    !Number.isFinite(
      input.control.householdReservePower,
    ) ||
    input.control.householdReservePower < 0
  ) {
    throw new Error(
      "householdReservePower must be a finite number >= 0.",
    );
  }


  if (
    input.control.pvTargetTemperature <=
    input.storage.minTemperature
  ) {
    throw new Error(
      "PV target temperature must be above storage minimum temperature.",
    );
  }


  if (
    input.control.pvTargetTemperature >
    input.storage.maxTemperature
  ) {
    throw new Error(
      "PV target temperature cannot exceed storage maximum temperature.",
    );
  }


  if (
    input.control.pvTargetTemperature >
    input.heatPump.maxFlowTemperature
  ) {
    throw new Error(
      "PV target temperature cannot exceed heat-pump maximum flow temperature.",
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