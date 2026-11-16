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
 * APPROACH A6 — PACKED-INTO RESULT DECODERS
 * ================================================================
 *
 * Enabled only when:
 *
 *   SUNOPT_A6_PACKED_INTO=1
 *
 * Otherwise the established object path below remains unchanged.
 */


function decodeNoSunshinePackedOutput(
  output:
    Float64Array,
): RustNoSunshineScopeResult {

  return {
    kWhNoSunshine:
      output[0],

    bucketDemandEnergy:
      output[1],

    bucketCoolingLossEnergy:
      output[2],

    preBucketHoldingLossEnergy:
      output[3],

    allocatedPvThermalEnergy:
      output[4],

    unavoidableThermalDeficit:
      output[5],

    requiredStorageEnergyAtStart:
      output[6],

    additionalStorageEnergyNeededAtStart:
      output[7],

    currentWaterlinePower:
      output[8],

    currentScheduledHpElectricalPower:
      output[9],

    currentScheduledHpThermalPower:
      output[10],
  };
}


function decodeCausalPackedOutput(
  output:
    Float64Array,
): RustPeakShavingCausalScopeResult {

  return {
    cutLevelPower:
      output[0],

    normalPlanSurvives:
      output[1] !==
      0,

    maximumPvPlanSurvives:
      output[2] !==
      0,

    survivalChecks:
      output[3],

    binarySearchIterations:
      output[4],
  };
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


  const a6PackedInput =
    new Float64Array(
      15,
    );

  const a6Output =
    new Float64Array(
      11,
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


if (
      process.env
        .SUNOPT_A6_PACKED_INTO ===
      "1"
    ) {

      a6PackedInput[0] =
        scopeInput.currentIndex;

      a6PackedInput[1] =
        scopeInput.horizonSteps;

      a6PackedInput[2] =
        scopeInput.protectedSteps;

      a6PackedInput[3] =
        scopeInput.timeStepHours;

      a6PackedInput[4] =
        scopeInput.householdReservePower;

      a6PackedInput[5] =
        scopeInput.maximumHeatPumpElectricalPower;

      a6PackedInput[6] =
        scopeInput.cop;

      a6PackedInput[7] =
        scopeInput.initialWaterlinePower;

      a6PackedInput[8] =
        scopeInput.targetStorageEnergy;

      a6PackedInput[9] =
        scopeInput.currentStorageEnergy;

      a6PackedInput[10] =
        scopeInput.storageEnergyPerKelvin;

      a6PackedInput[11] =
        scopeInput.storageConductanceWPerKelvin;

      a6PackedInput[12] =
        scopeInput.storageMinTemperature;

      a6PackedInput[13] =
        scopeInput.ambientTemperature;

      a6PackedInput[14] =
        scopeInput.dailyHeatLoss;


      const a6Result =
        SUNOPT_ACCELERATION_SCOPE
          .runPackedInto<
            RustNoSunshineScopeResult
          >(
            "rust_noSunshine",
            pvPower,
            loadEnergy,
            a6PackedInput,
            a6Output,
            decodeNoSunshinePackedOutput,
          );


      if (
        a6Result !==
        null
      ) {

        return a6Result;
      }
    }


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


  const a6PackedInput =
    new Float64Array(
      14,
    );

  const a6Output =
    new Float64Array(
      5,
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


if (
      process.env
        .SUNOPT_A6_PACKED_INTO ===
      "1"
    ) {

      a6PackedInput[0] =
        scopeInput.currentIndex;

      a6PackedInput[1] =
        scopeInput.horizonSteps;

      a6PackedInput[2] =
        scopeInput.currentStorageEnergy;

      a6PackedInput[3] =
        scopeInput.minimumReserveEnergy;

      a6PackedInput[4] =
        scopeInput.targetStorageEnergy;

      a6PackedInput[5] =
        scopeInput.maximumHeatPumpElectricalPower;

      a6PackedInput[6] =
        scopeInput.cop;

      a6PackedInput[7] =
        scopeInput.energyCutLevel;

      a6PackedInput[8] =
        scopeInput.timeStepHours;

      a6PackedInput[9] =
        scopeInput.storageEnergyPerKelvin;

      a6PackedInput[10] =
        scopeInput.storageConductanceWPerKelvin;

      a6PackedInput[11] =
        scopeInput.storageMinTemperature;

      a6PackedInput[12] =
        scopeInput.ambientTemperature;

      a6PackedInput[13] =
        scopeInput.dailyHeatLoss;


      const a6Result =
        SUNOPT_ACCELERATION_SCOPE
          .runPackedInto<
            RustPeakShavingCausalScopeResult
          >(
            "rust_peakShavingCausal",
            exportPower,
            demandEnergy,
            a6PackedInput,
            a6Output,
            decodeCausalPackedOutput,
          );


      if (
        a6Result !==
        null
      ) {

        return {
          ...a6Result,
          usedNative:
            true,
        };
      }
    }


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
