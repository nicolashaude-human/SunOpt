// src/simulation/control-model-plasma-wave.ts

import type {
  HeatPump,
  HotWaterControl,
  HotWaterData,
  ThermalStorage,
  WeatherData,
} from "@/domain";

import {
  calculateHotWaterForecastTarget,
} from "@/domain";

import {
  CONFIG,
} from "@/config/config";

import type {
  HotWaterControlModel,
} from "./hot-water-control-model";

import {
  calculatePlasmaWaveAllPackagesForecast,
} from "./plasma-wave-user-forecast-all-packages";

import {
  createPlasmaWaveAccelerationScope,
} from "./plasma-wave-acceleration-scope";

import {
  calculateCombinedWaterline,
} from "./combined-waterline-scheduler";


export interface PlasmaWaveControlDecision {
  currentIndex: number;

  required48hEnergy: number;
  kWhNoSunshine: number;

  userForecastWaterlinePower: number;
  noSunshineWaterlinePower: number;
  effectiveWaterlinePower: number;

  cutLevelPower: number;

  limitingStrategy:
    | "userForecast"
    | "noSunshine"
    | "equal";

  exportBeforeHpPower: number;
  exportAfterHpPower: number;

  heatPumpElectricalPower: number;
}


export interface PlasmaWaveControlModelInput {
  weatherData: WeatherData;
  hotWaterData: HotWaterData;

  heatPump: HeatPump;
  storage: ThermalStorage;
  control: HotWaterControl;

  timeStepHours: number;

  planningHorizonHours?: number;
  targetTemperature?: number;

  /*
   * Optional debug hook.
   *
   * It does not influence the controller.
   */
  onDecision?: (
    decision: PlasmaWaveControlDecision,
  ) => void;
}


/**
 * Creates the combined Plasma-Wave control model.
 *
 * Two independent forecast strategies calculate a Waterline:
 *
 * 1. kWh_user
 *    Rolling 48-hour chronological demand forecast.
 *
 * 2. kWh_noSunshine
 *    Seven-day no-sunshine forecast.
 *
 * Expensive numerical work is coordinated by one global
 * Plasma-Wave acceleration scope.
 *
 * The effective Waterline is always the lower one:
 *
 * L_effective = min(L_user, L_noSunshine)
 *
 * Mandatory reserve behavior remains part of the common
 * hot-water simulation and is not changed here.
 */
export function createPlasmaWaveControlModel(
  input: PlasmaWaveControlModelInput,
): HotWaterControlModel {

  const {
    weatherData,
    hotWaterData,
    heatPump,
    storage,
    control,
    timeStepHours,
  } = input;


  const planningHorizonHours =
    input.planningHorizonHours ??
    CONFIG.control.plasmaWavePlanningHorizonHours;


  const targetTemperature =
    input.targetTemperature ??
    CONFIG.control.pvTargetTemperature;


  /*
   * ==============================================================
   * PRECALCULATE 48-HOUR USER FORECAST
   * ==============================================================
   *
   * Calculated once for the complete annual simulation.
   */
  const allPackagesForecast =
    calculatePlasmaWaveAllPackagesForecast({

      weatherData,

      hotWaterData,

      heatPump,

      storage,

      timeStepHours,

      planningHorizonHours,

      householdReservePower:
        control.householdReservePower,

      lossReferenceTemperature:
        storage.minTemperature,

      safetyFactor:
        control.forecastSafetyFactor,
    });


  /*
   * ==============================================================
   * GLOBAL ACCELERATION SCOPE
   * ==============================================================
   *
   * Created ONCE for the annual controller.
   *
   * It centrally owns the expensive:
   *
   * - 48 h kWh_user scheduler
   * - 7 day kWh_noSunshine scheduler
   *
   * Later the 48 h causal hot loop can move to Rust/Cranelift without
   * changing this controller again.
   */
  const accelerationScope =
    createPlasmaWaveAccelerationScope({

      weatherData,

      hotWaterData,

      heatPump,

      storage,

      timeStepHours,

      planningHorizonHours,

      householdReservePower:
        control.householdReservePower,

      safetyFactor:
        control.forecastSafetyFactor,

      targetTemperature,

      noSunshineHorizonHours:
        CONFIG.control
          .noSunshineHorizonHours,

      noSunshineProtectedHorizonHours:
        CONFIG.control
          .noSunshineProtectedHorizonHours,

      noSunshineInitialWaterlinePower:
        CONFIG.control
          .noSunshineInitialWaterlinePower,
    });


  /*
   * ==============================================================
   * CONTROL MODEL
   * ==============================================================
   */
  const plasmaWaveControlModel:
    HotWaterControlModel = (
      context,
    ) => {

    const {
      currentIndex,
      storageState,
    } = context;


    /*
     * --------------------------------------------------------------
     * CURRENT 48-HOUR FORECAST
     * --------------------------------------------------------------
     */
    const forecastPoint =
      allPackagesForecast.points[
        currentIndex
      ];


    if (!forecastPoint) {
      throw new Error(
        `Plasma Wave forecast missing at index ${currentIndex}.`,
      );
    }


    /*
     * --------------------------------------------------------------
     * BASELINE / MANDATORY RESERVE
     * --------------------------------------------------------------
     */
    const baselineForecast =
      calculateHotWaterForecastTarget({
        currentIndex,
        weatherData,
        hotWaterData,
        heatPump,
        storage,
        control,
        timeStepHours,
      });


    const missingToMinimumReserve =
      Math.max(
        0,
        baselineForecast
          .minimumReserveEnergy -
          storageState.storedEnergy,
      );


    const storageEnergyAfterMandatoryReserve =
      Math.min(
        baselineForecast.targetCapacity,
        storageState.storedEnergy +
          missingToMinimumReserve,
      );


    /*
     * ==============================================================
     * GLOBAL EXPENSIVE FORECAST SCOPE
     * ==============================================================
     *
     * Both expensive forecasting paths are called through one
     * coordinator.
     */
    const accelerated =
      accelerationScope.run({

        currentIndex,

        currentStorageEnergy:
          storageEnergyAfterMandatoryReserve,

        requiredThermalInventory:
          forecastPoint
            .lossAdjustedHotWaterForecastEnergy,

        minimumReserveEnergy:
          baselineForecast
            .minimumReserveEnergy,
      });


    const peakShaving =
      accelerated.peakShaving;


    const noSunshine =
      accelerated.noSunshine;


    /*
     * ==============================================================
     * COMBINE BOTH WATERLINES
     * ==============================================================
     */
    const combinedWaterline =
      calculateCombinedWaterline({

        userForecastWaterlinePower:
          peakShaving.cutLevelPower,

        noSunshineWaterlinePower:
          noSunshine
            .currentWaterlinePower,
      });


    /*
     * ==============================================================
     * CURRENT PV EXPORT
     * ==============================================================
     */
    const currentWeather =
      weatherData[currentIndex];


    if (!currentWeather) {
      throw new Error(
        `Weather point missing at index ${currentIndex}.`,
      );
    }


    const currentPvExportPower =
      Math.max(
        0,
        currentWeather.pvPower -
          control.householdReservePower,
      );


    /*
     * ==============================================================
     * EFFECTIVE HEAT-PUMP POWER
     * ==============================================================
     *
     * Combine Waterlines first, calculate HP power once.
     */
    const scheduledForecastHpElectricalPower =
      Math.min(

        peakShaving
          .maximumHeatPumpElectricalPower,

        Math.max(
          0,
          currentPvExportPower -
            combinedWaterline
              .effectiveWaterlinePower,
        ),
      );


    const scheduledForecastThermalEnergy =
      scheduledForecastHpElectricalPower *
      peakShaving.cop *
      timeStepHours;


    const exportAfterHpPower =
      Math.max(
        0,
        currentPvExportPower -
          scheduledForecastHpElectricalPower,
      );


    /*
     * ==============================================================
     * DEBUG INFORMATION
     * ==============================================================
     */
    input.onDecision?.({
      currentIndex,

      required48hEnergy:
        forecastPoint
          .lossAdjustedHotWaterForecastEnergy,

      kWhNoSunshine:
        noSunshine.kWhNoSunshine,

      userForecastWaterlinePower:
        peakShaving.cutLevelPower,

      noSunshineWaterlinePower:
        noSunshine
          .currentWaterlinePower,

      effectiveWaterlinePower:
        combinedWaterline
          .effectiveWaterlinePower,

      cutLevelPower:
        combinedWaterline
          .effectiveWaterlinePower,

      limitingStrategy:
        combinedWaterline
          .limitingStrategy,

      exportBeforeHpPower:
        currentPvExportPower,

      exportAfterHpPower,

      heatPumpElectricalPower:
        scheduledForecastHpElectricalPower,
    });


    /*
     * ==============================================================
     * TARGET PASSED TO COMMON SIMULATOR
     * ==============================================================
     */
    const plasmaWaveTargetEnergy =
      Math.min(

        baselineForecast.targetCapacity,

        storageEnergyAfterMandatoryReserve +
          scheduledForecastThermalEnergy,
      );


    return {
      ...baselineForecast,

      targetEnergy:
        plasmaWaveTargetEnergy,

      forecastRequiredEnergy:
        peakShaving
          .requestedChargeThermalEnergy,

      minimumReserveEnergy:
        baselineForecast
          .minimumReserveEnergy,

      targetCapacity:
        baselineForecast
          .targetCapacity,
    };
  };


  return plasmaWaveControlModel;
}
