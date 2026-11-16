// src/simulation/rust-acceleration-adapters.ts

import type {
  HeatPump,
  ThermalStorage,
} from "@/domain";

import {
  calculateStorageEnergyAtTemperature,
} from "@/domain";

import {
  calculateStorageThermalConductance,
} from "@/domain/thermal-storage-calculation";

import {
  CONFIG,
} from "@/config/config";

import type {
  PlasmaWaveHistoricalSeries,
} from "./plasma-wave-historical-series-cache";

import {
  runRustNoSunshineScope,
  runRustPeakShavingCausalScope,

  type RustNoSunshineScopeInput,
  type RustNoSunshineScopeResult,

  type RustPeakShavingCausalScopeInput,
  type RustPeakShavingCausalScopeResult,
} from "./rust-acceleration-scopes";

import {
  SUNOPT_ACCELERATION_SCOPE,
} from "./global-acceleration-scope";


/*
 * ================================================================
 * ADAPTERS ONLY
 * ================================================================
 *
 * The expensive algorithms do NOT live here.
 *
 * Their canonical TypeScript source is:
 *
 *   src/simulation/rust-acceleration-scopes.ts
 *
 * This file only converts SunOpt domain objects into compact numeric
 * arrays/scalars and chooses native execution or the SAME TypeScript
 * scope as fallback.
 */


function getStorageKernelConstants(
  storage:
    ThermalStorage,
) {

  const temperatureSpan =
    storage.maxTemperature -
    storage.minTemperature;


  if (
    temperatureSpan <=
    0
  ) {

    throw new Error(
      "Storage temperature span must be greater than zero.",
    );
  }


  const maximumStorageEnergy =
    calculateStorageEnergyAtTemperature(
      storage,
      storage.maxTemperature,
    );


  const storageEnergyPerKelvin =
    maximumStorageEnergy /
    temperatureSpan;


  const storageConductanceWPerKelvin =
    calculateStorageThermalConductance(
      storage,
    );


  const insulation =
    storage.insulation ??
    CONFIG.storage.insulation;


  const ambientTemperature =
    insulation
      .ambientTemperature;


  /*
   * Preserve the canonical legacy branch:
   *
   * no explicit insulation + dailyHeatLoss defined
   * -> fixed kWh/day loss.
   *
   * Otherwise use conductance / ambient temperature.
   */
  const dailyHeatLoss =
    storage.insulation ===
      undefined &&
    storage.dailyHeatLoss !==
      undefined
      ? storage.dailyHeatLoss
      : -1;


  return {
    storageEnergyPerKelvin,

    storageConductanceWPerKelvin,

    storageMinTemperature:
      storage.minTemperature,

    ambientTemperature,

    dailyHeatLoss,
  };
}


/*
 * ================================================================
 * RESULT VALIDATION
 * ================================================================
 */


function validateNoSunshineResult(
  value: unknown,
): RustNoSunshineScopeResult {

  if (
    value === null ||
    typeof value !==
      "object"
  ) {

    throw new Error(
      "rust_noSunshine returned no object.",
    );
  }


  const candidate =
    value as Record<
      string,
      unknown
    >;


  const keys = [
    "kWhNoSunshine",
    "bucketDemandEnergy",
    "bucketCoolingLossEnergy",
    "preBucketHoldingLossEnergy",
    "allocatedPvThermalEnergy",
    "unavoidableThermalDeficit",
    "requiredStorageEnergyAtStart",
    "additionalStorageEnergyNeededAtStart",
    "currentWaterlinePower",
    "currentScheduledHpElectricalPower",
    "currentScheduledHpThermalPower",
  ] as const;


  for (
    const key of keys
  ) {

    if (
      typeof candidate[
        key
      ] !==
      "number"
    ) {

      throw new Error(
        `rust_noSunshine result "${key}" is not numeric.`,
      );
    }
  }


  return candidate as unknown as
    RustNoSunshineScopeResult;
}


function validatePeakShavingCausalResult(
  value: unknown,
): RustPeakShavingCausalScopeResult {

  if (
    value === null ||
    typeof value !==
      "object"
  ) {

    throw new Error(
      "rust_peakShavingCausal returned no object.",
    );
  }


  const candidate =
    value as Record<
      string,
      unknown
    >;


  for (
    const key of [
      "cutLevelPower",
      "survivalChecks",
      "binarySearchIterations",
    ] as const
  ) {

    if (
      typeof candidate[
        key
      ] !==
      "number"
    ) {

      throw new Error(
        `rust_peakShavingCausal result "${key}" is not numeric.`,
      );
    }
  }


  for (
    const key of [
      "normalPlanSurvives",
      "maximumPvPlanSurvives",
    ] as const
  ) {

    if (
      typeof candidate[
        key
      ] !==
      "boolean"
    ) {

      throw new Error(
        `rust_peakShavingCausal result "${key}" is not boolean.`,
      );
    }
  }


  return candidate as unknown as
    RustPeakShavingCausalScopeResult;
}


/*
 * ================================================================
 * 7-DAY ADAPTER
 * ================================================================
 */


export interface NoSunshineAcceleratorInput {
  historicalSeries:
    PlasmaWaveHistoricalSeries;

  heatPump:
    HeatPump;

  storage:
    ThermalStorage;

  horizonHours:
    number;

  protectedHorizonHours:
    number;

  initialWaterlinePower:
    number;

  targetTemperature:
    number;
}


export interface NoSunshineAcceleratorRunInput {
  currentIndex:
    number;

  currentStorageEnergy:
    number;
}


export interface NoSunshineAccelerator {
  run(
    input:
      NoSunshineAcceleratorRunInput,
  ): RustNoSunshineScopeResult;
}


export function createNoSunshineAccelerator(
  input:
    NoSunshineAcceleratorInput,
): NoSunshineAccelerator {

  const {
    historicalSeries,
    heatPump,
    storage,
    horizonHours,
    protectedHorizonHours,
    initialWaterlinePower,
    targetTemperature:
      requestedTargetTemperature,
  } = input;
  
  
  const {
    length,
    timeStepHours,
    householdReservePower,
    cop,
    pvPowerByIndex:
      pvPower,
    loadEnergyByIndex:
      loadEnergy,
  } = historicalSeries;


  if (length === 0) {
    throw new Error(
      "Historical Plasma-Wave series must not be empty.",
    );
  }
  
  
  if (cop <= 0) {
    throw new Error(
      "Heat-pump COP must be greater than zero.",
    );
  }


  const maximumHeatPumpElectricalPower =
    heatPump.thermalPower /
    cop;


  const physicalTargetTemperature =
    Math.min(
      requestedTargetTemperature,

      storage.maxTemperature,

      heatPump.maxFlowTemperature,
    );


  const targetStorageEnergy =
    calculateStorageEnergyAtTemperature(
      storage,
      physicalTargetTemperature,
    );


    const horizonSteps =
    Math.min(
      Math.ceil(
        horizonHours /
        timeStepHours,
      ),
      length,
    );


  const protectedSteps =
    Math.min(
      Math.ceil(
        protectedHorizonHours /
        timeStepHours,
      ),

      horizonSteps,
    );


  const storageConstants =
    getStorageKernelConstants(
      storage,
    );


  function run(
    dynamic:
      NoSunshineAcceleratorRunInput,
  ): RustNoSunshineScopeResult {

    const scopeInput:
      RustNoSunshineScopeInput = {

      pvPower,

      loadEnergy,

      currentIndex:
        dynamic.currentIndex,

      horizonSteps,

      protectedSteps,

      timeStepHours,

      householdReservePower,

      maximumHeatPumpElectricalPower,

      cop,

      initialWaterlinePower,

      targetStorageEnergy,

      currentStorageEnergy:
        dynamic.currentStorageEnergy,

      ...storageConstants,
    };


    const parameters = {
      currentIndex:
        scopeInput.currentIndex,

      horizonSteps:
        scopeInput.horizonSteps,

      protectedSteps:
        scopeInput.protectedSteps,

      timeStepHours:
        scopeInput.timeStepHours,

      householdReservePower:
        scopeInput.householdReservePower,

      maximumHeatPumpElectricalPower:
        scopeInput
          .maximumHeatPumpElectricalPower,

      cop:
        scopeInput.cop,

      initialWaterlinePower:
        scopeInput.initialWaterlinePower,

      targetStorageEnergy:
        scopeInput.targetStorageEnergy,

      storageEnergyPerKelvin:
        scopeInput.storageEnergyPerKelvin,

      storageConductanceWPerKelvin:
        scopeInput
          .storageConductanceWPerKelvin,

      storageMinTemperature:
        scopeInput.storageMinTemperature,

      ambientTemperature:
        scopeInput.ambientTemperature,

      dailyHeatLoss:
        scopeInput.dailyHeatLoss,

      currentStorageEnergy:
        scopeInput.currentStorageEnergy,
    };


    const nativeResult =
      SUNOPT_ACCELERATION_SCOPE
        .run<
          RustNoSunshineScopeResult
        >(
          "rust_noSunshine",

          [
            pvPower,
            loadEnergy,
            parameters,
          ],

          validateNoSunshineResult,
        );


    if (
      nativeResult !==
      null
    ) {

      return nativeResult;
    }


    return runRustNoSunshineScope(
      scopeInput,
    );
  }


  return {
    run,
  };
}


/*
 * ================================================================
 * 48-HOUR CAUSAL ADAPTER
 * ================================================================
 */


export interface PeakShavingCausalAcceleratorInput {
  historicalSeries:
    PlasmaWaveHistoricalSeries;

  storage:
    ThermalStorage;
}


export interface PeakShavingCausalRunInput {
  currentIndex:
    number;

  horizonSteps:
    number;

  currentStorageEnergy:
    number;

  minimumReserveEnergy:
    number;

  targetStorageEnergy:
    number;

  maximumHeatPumpElectricalPower:
    number;

  cop:
    number;

  energyCutLevel:
    number;
}


export interface PeakShavingCausalResult
  extends RustPeakShavingCausalScopeResult {

  usedNative:
    boolean;
}


export interface PeakShavingCausalAccelerator {
  run(
    input:
      PeakShavingCausalRunInput,
  ): PeakShavingCausalResult;
}


export function createPeakShavingCausalAccelerator(
  input:
    PeakShavingCausalAcceleratorInput,
): PeakShavingCausalAccelerator {

  const {
    historicalSeries,
    storage,
  } = input;
  
  
  const {
    length,
    timeStepHours,
  
    exportPowerByIndex:
      exportPower,
  
    loadEnergyByIndex:
      demandEnergy,
  } = historicalSeries;
  
  
  if (length === 0) {
    throw new Error(
      "Historical Plasma-Wave series must not be empty.",
    );
  }

  if (
    length === 0
  ) {
    throw new Error(
      "Historical Plasma-Wave series must not be empty.",
    );
  }

  const storageConstants =
    getStorageKernelConstants(
      storage,
    );


  function run(
    dynamic:
      PeakShavingCausalRunInput,
  ): PeakShavingCausalResult {

    const scopeInput:
      RustPeakShavingCausalScopeInput = {

      exportPower,

      demandEnergy,

      currentIndex:
        dynamic.currentIndex,

      horizonSteps:
        dynamic.horizonSteps,

      currentStorageEnergy:
        dynamic.currentStorageEnergy,

      minimumReserveEnergy:
        dynamic.minimumReserveEnergy,

      targetStorageEnergy:
        dynamic.targetStorageEnergy,

      maximumHeatPumpElectricalPower:
        dynamic
          .maximumHeatPumpElectricalPower,

      cop:
        dynamic.cop,

      energyCutLevel:
        dynamic.energyCutLevel,

      timeStepHours,

      ...storageConstants,
    };


    const parameters = {
      currentIndex:
        scopeInput.currentIndex,

      horizonSteps:
        scopeInput.horizonSteps,

      currentStorageEnergy:
        scopeInput.currentStorageEnergy,

      minimumReserveEnergy:
        scopeInput.minimumReserveEnergy,

      targetStorageEnergy:
        scopeInput.targetStorageEnergy,

      maximumHeatPumpElectricalPower:
        scopeInput
          .maximumHeatPumpElectricalPower,

      cop:
        scopeInput.cop,

      energyCutLevel:
        scopeInput.energyCutLevel,

      timeStepHours:
        scopeInput.timeStepHours,

      storageEnergyPerKelvin:
        scopeInput.storageEnergyPerKelvin,

      storageConductanceWPerKelvin:
        scopeInput
          .storageConductanceWPerKelvin,

      storageMinTemperature:
        scopeInput.storageMinTemperature,

      ambientTemperature:
        scopeInput.ambientTemperature,

      dailyHeatLoss:
        scopeInput.dailyHeatLoss,
    };


    const nativeResult =
      SUNOPT_ACCELERATION_SCOPE
        .run<
          RustPeakShavingCausalScopeResult
        >(
          "rust_peakShavingCausal",

          [
            exportPower,
            demandEnergy,
            parameters,
          ],

          validatePeakShavingCausalResult,
        );


    if (
      nativeResult !==
      null
    ) {

      return {
        ...nativeResult,

        usedNative:
          true,
      };
    }


    return {
      ...runRustPeakShavingCausalScope(
        scopeInput,
      ),

      usedNative:
        false,
    };
  }


  return {
    run,
  };
}
