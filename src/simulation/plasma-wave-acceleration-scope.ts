// src/simulation/plasma-wave-acceleration-scope.ts

import type {
  HeatPump,
  HotWaterData,
  ThermalStorage,
  WeatherData,
} from "@/domain";

import {
  calculatePlasmaWavePeakShavingSchedule,
  type PlasmaWavePeakShavingScheduleResult,
} from "./plasma-wave-peak-shaving-scheduler-profiled";

import {
  createNoSunshineAccelerator,
  createPeakShavingCausalAccelerator,
} from "./rust-acceleration-adapters";

import {
  SUNOPT_ACCELERATION_SCOPE,
  type GlobalAccelerationScopeStatus,
} from "./global-acceleration-scope";

import {
  getPlasmaWaveHistoricalSeries,
} from "./plasma-wave-historical-series-cache";


/*
 * ================================================================
 * PLASMA-WAVE GLOBAL ACCELERATION SCOPE
 * ================================================================
 *
 * This is the high-level coordinator for expensive Plasma-Wave work.
 *
 * Static annual/model data is captured ONCE.
 *
 * Per 15-minute controller call only dynamic state is supplied:
 *
 * - currentIndex
 * - currentStorageEnergy
 * - requiredThermalInventory
 * - minimumReserveEnergy
 *
 * The controller does not need to know whether a hot loop currently
 * runs in TypeScript, Rust/Cranelift, SIMD or another backend.
 */


export interface PlasmaWaveAccelerationScopeInput {
  weatherData: WeatherData;
  hotWaterData: HotWaterData;

  heatPump: HeatPump;
  storage: ThermalStorage;

  timeStepHours: number;
  planningHorizonHours: number;

  householdReservePower: number;
  safetyFactor: number;

  targetTemperature: number;

  noSunshineHorizonHours: number;
  noSunshineProtectedHorizonHours: number;
  noSunshineInitialWaterlinePower: number;
}


export interface PlasmaWaveAccelerationRunInput {
  currentIndex: number;

  currentStorageEnergy: number;

  requiredThermalInventory: number;

  minimumReserveEnergy: number;
}


export interface PlasmaWaveAccelerationRunResult {
  peakShaving:
    PlasmaWavePeakShavingScheduleResult;

  noSunshine:
    ReturnType<
      ReturnType<
        typeof createNoSunshineAccelerator
      >["run"]
    >;
}


export interface PlasmaWaveAccelerationProfile {
  calls: number;

  totalMs: number;

  peakShavingMs: number;

  noSunshineMs: number;

  orchestrationMs: number;

  native:
    GlobalAccelerationScopeStatus;
}


export interface PlasmaWaveAccelerationScope {
  run(
    input: PlasmaWaveAccelerationRunInput,
  ): PlasmaWaveAccelerationRunResult;

  getProfile():
    PlasmaWaveAccelerationProfile;

  printProfile():
    void;
}


export function createPlasmaWaveAccelerationScope(
  input: PlasmaWaveAccelerationScopeInput,
): PlasmaWaveAccelerationScope {

  const {
    weatherData,
    hotWaterData,
    heatPump,
    storage,
    timeStepHours,
    planningHorizonHours,
    householdReservePower,
    safetyFactor,
    targetTemperature,
    noSunshineHorizonHours,
    noSunshineProtectedHorizonHours,
    noSunshineInitialWaterlinePower,
  } = input;


  /*
   * ================================================================
   * INPUT VALIDATION
   * ================================================================
   */

  if (
    weatherData.length === 0
  ) {
    throw new Error(
      "weatherData must not be empty.",
    );
  }


  if (
    weatherData.length !==
    hotWaterData.length
  ) {
    throw new Error(
      "Weather data and hot-water data must have the same length.",
    );
  }


  /*
   * ================================================================
   * SHARED HISTORICAL SERIES
   * ================================================================
   *
   * The weather and hot-water history is immutable during one
   * historical simulation.
   *
   * allPackagesForecast normally requests the same dataset before
   * this acceleration scope is created. Therefore this call should
   * normally be a cache HIT.
   *
   * The returned typed arrays are shared by:
   *
   * - the 48-hour causal scheduler
   * - the 7-day no-sunshine scheduler
   *
   * Tank-dependent storage physics is NOT cached here.
   */

  const historicalSeries =
    getPlasmaWaveHistoricalSeries({
      weatherData,
      hotWaterData,
      heatPump,
      storage,
      timeStepHours,
      householdReservePower,
      safetyFactor,

      lossReferenceTemperature:
        storage.minTemperature,
    });


  if (
    historicalSeries.length !==
    weatherData.length
  ) {
    throw new Error(
      "Historical series and weather data must have same length.",
    );
  }


  /*
   * Resolve the global native runtime once.
   *
   * Individual kernels still decide independently whether native
   * execution is available. A failure in one kernel does not disable
   * the others.
   */

  SUNOPT_ACCELERATION_SCOPE.initialise();


  /*
   * ================================================================
   * PREPARE 48-HOUR CAUSAL ACCELERATOR ONCE
   * ================================================================
   *
   * The accelerator now receives the already prepared historical
   * Float64Arrays directly.
   *
   * No annual export/demand arrays are rebuilt here.
   */

  const peakShavingCausalAccelerator =
    createPeakShavingCausalAccelerator({
      historicalSeries,
      storage,
    });


  /*
   * ================================================================
   * PREPARE 7-DAY NO-SUNSHINE ACCELERATOR ONCE
   * ================================================================
   *
   * This accelerator uses the SAME historical arrays as the
   * 48-hour accelerator.
   */

  const noSunshineAccelerator =
    createNoSunshineAccelerator({
      historicalSeries,
      heatPump,
      storage,

      horizonHours:
        noSunshineHorizonHours,

      protectedHorizonHours:
        noSunshineProtectedHorizonHours,

      initialWaterlinePower:
        noSunshineInitialWaterlinePower,

      targetTemperature,
    });


  /*
   * ================================================================
   * PROFILE ACCUMULATORS
   * ================================================================
   */

  let calls = 0;

  let totalMs = 0;

  let peakShavingMs = 0;

  let noSunshineMs = 0;


  /*
   * ================================================================
   * RUN
   * ================================================================
   */

  function run(
    dynamic:
      PlasmaWaveAccelerationRunInput,
  ): PlasmaWaveAccelerationRunResult {

    const totalStart =
      performance.now();


    /*
     * ==============================================================
     * 48-HOUR kWh_user
     * ==============================================================
     */

    const peakStart =
      performance.now();


      const peakShaving =
      calculatePlasmaWavePeakShavingSchedule({
        weatherData,
        heatPump,
        storage,
    
        currentIndex:
          dynamic.currentIndex,
    
        currentStorageEnergy:
          dynamic.currentStorageEnergy,
    
        requiredThermalInventory:
          dynamic.requiredThermalInventory,
    
        historicalSeries,
    
        minimumReserveEnergy:
          dynamic.minimumReserveEnergy,
    
        planningHorizonHours,
    
        timeStepHours,
    
        householdReservePower,
    
        targetTemperature,
    
        causalAccelerator:
          peakShavingCausalAccelerator,
      });


    peakShavingMs +=
      performance.now() -
      peakStart;


    /*
     * ==============================================================
     * 7-DAY kWh_noSunshine
     * ==============================================================
     */

    const noSunshineStart =
      performance.now();


    const noSunshine =
      noSunshineAccelerator.run({
        currentIndex:
          dynamic.currentIndex,

        currentStorageEnergy:
          dynamic.currentStorageEnergy,
      });


    noSunshineMs +=
      performance.now() -
      noSunshineStart;


    calls += 1;


    totalMs +=
      performance.now() -
      totalStart;


    return {
      peakShaving,
      noSunshine,
    };
  }


  /*
   * ================================================================
   * PROFILE
   * ================================================================
   */

  function getProfile():
    PlasmaWaveAccelerationProfile {

    const measuredKernelMs =
      peakShavingMs +
      noSunshineMs;


    return {
      calls,

      totalMs,

      peakShavingMs,

      noSunshineMs,

      orchestrationMs:
        Math.max(
          0,
          totalMs -
          measuredKernelMs,
        ),

      native:
        SUNOPT_ACCELERATION_SCOPE
          .status(),
    };
  }


  function printProfile():
    void {

    const profile =
      getProfile();


    console.log(
      "[SunOpt] Plasma-Wave global acceleration profile:",
      {
        calls:
          profile.calls,

        totalMs:
          profile.totalMs
            .toFixed(1),

        peakShavingMs:
          profile.peakShavingMs
            .toFixed(1),

        noSunshineMs:
          profile.noSunshineMs
            .toFixed(1),

        orchestrationMs:
          profile.orchestrationMs
            .toFixed(1),

        native:
          profile.native,
      },
    );
  }


  return {
    run,
    getProfile,
    printProfile,
  };
}