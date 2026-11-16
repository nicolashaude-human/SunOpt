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

import {
  getSunOptNativeBinding,
} from "./rust-acceleration-runtime";


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


interface CudaCausalJob {
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

  timeStepHours:
    number;

  storageEnergyPerKelvin:
    number;

  storageConductanceWPerKelvin:
    number;

  storageMinTemperature:
    number;

  ambientTemperature:
    number;

  dailyHeatLoss:
    number;
}


interface CudaCausalNativeProfile {
  causalCalls:
    number;

  causalJobs:
    number;

  parameterH2dMs:
    number;

  kernelMs:
    number;

  resultD2hMs:
    number;

  gpuTotalMs:
    number;

  hostWaitMs:
    number;

  wallMs:
    number;

  averageKernelUsPerJob:
    number;

  averageGpuUsPerJob:
    number;
}


interface CudaCausalWorkerLike {
  run(
    job:
      CudaCausalJob,
  ): unknown;

  profile():
    CudaCausalNativeProfile;
}


type CudaCausalWorkerConstructor =
  new (
    exportPower:
      Float64Array,

    demandEnergy:
      Float64Array,

    deviceId?:
      number,
  ) => CudaCausalWorkerLike;


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


  /*
   * ================================================================
   * PRODUCTIVE CUDA CAUSAL WORKER
   * ================================================================
   *
   * This is the adapter actually used by
   * plasma-wave-acceleration-scope.ts.
   *
   * The annual export and demand arrays are uploaded once when the
   * accelerator is created and remain resident in VRAM.
   *
   * Each run() transfers only the compact dynamic causal parameters
   * and receives the compact result.
   *
   * A CUDA failure disables CUDA for this accelerator instance and
   * falls back to the existing CPU-native / TypeScript chain.
   */
  let cudaWorker:
    CudaCausalWorkerLike | null =
    null;

  let cudaDisabled =
    false;

  let cudaFailureReported =
    false;

  let cudaCalls =
    0;

  let cudaFailures =
    0;

  let cpuNativeFallbackCalls =
    0;

  let typescriptFallbackCalls =
    0;

  let performanceProfileReported =
    false;


  function reportPerformanceProfile(
    currentIndex:
      number,
  ): void {

    if (
      performanceProfileReported ||
      currentIndex !==
        length - 1
    ) {
      return;
    }


    performanceProfileReported =
      true;


    let gpu:
      CudaCausalNativeProfile | null =
      null;


    if (
      cudaWorker !== null
    ) {

      try {

        gpu =
          cudaWorker.profile();

      } catch (error) {

        console.warn(
          "[SunOpt] Could not read CUDA causal performance profile.",
          error,
        );
      }
    }


    console.log(
      "[SunOpt] CUDA causal performance profile:",
      {
        annualSteps:
          length,

        cudaCalls,

        cudaFailures,

        cpuNativeFallbackCalls,

        typescriptFallbackCalls,

        gpu,
      },
    );
  }


  try {

    const binding =
      getSunOptNativeBinding();

    const CudaCausalWorker =
      binding?.CudaCausalWorker as
        CudaCausalWorkerConstructor |
        undefined;


    if (
      typeof CudaCausalWorker ===
      "function"
    ) {

      cudaWorker =
        new CudaCausalWorker(
          exportPower,
          demandEnergy,
          0,
        );


      console.log(
        "[SunOpt] CUDA causal worker active (productive adapter).",
        {
          deviceId:
            0,

          residentAnnualValues:
            length,
        },
      );
    }

  } catch (error) {

    cudaWorker =
      null;

    cudaDisabled =
      true;

    cudaFailureReported =
      true;


    console.warn(
      "[SunOpt] Productive CUDA causal worker unavailable; using CPU acceleration.",
      error,
    );
  }


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


    const parameters:
      CudaCausalJob = {

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


    /*
     * --------------------------------------------------------------
     * 1. CUDA
     * --------------------------------------------------------------
     */
    if (
      cudaWorker !== null &&
      !cudaDisabled
    ) {

      try {

        const cudaResult =
          validatePeakShavingCausalResult(
            cudaWorker.run(
              parameters,
            ),
          );


        cudaCalls +=
          1;


        reportPerformanceProfile(
          dynamic.currentIndex,
        );


        return {
          ...cudaResult,

          usedNative:
            true,
        };

      } catch (error) {

        cudaFailures +=
          1;

        cudaDisabled =
          true;


        if (
          !cudaFailureReported
        ) {

          cudaFailureReported =
            true;


          console.warn(
            "[SunOpt] Productive CUDA causal worker failed; disabling CUDA and falling back to CPU acceleration.",
            error,
          );
        }
      }
    }


    /*
     * --------------------------------------------------------------
     * 2. EXISTING CPU NATIVE / CRANELIFT
     * --------------------------------------------------------------
     */
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

      cpuNativeFallbackCalls +=
        1;


      reportPerformanceProfile(
        dynamic.currentIndex,
      );


      return {
        ...nativeResult,

        usedNative:
          true,
      };
    }


    /*
     * --------------------------------------------------------------
     * 3. CANONICAL TYPESCRIPT FALLBACK
     * --------------------------------------------------------------
     */
    typescriptFallbackCalls +=
      1;


    reportPerformanceProfile(
      dynamic.currentIndex,
    );


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
