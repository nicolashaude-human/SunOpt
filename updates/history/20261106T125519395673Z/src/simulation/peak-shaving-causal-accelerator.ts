// src/simulation/peak-shaving-causal-accelerator.ts

import type {
  ThermalStorage,
  WeatherData,
} from "@/domain";

import {
  calculateStorageEnergyAtTemperature,
} from "@/domain";

import {
  calculateStorageThermalConductance,
  calculateThermalStorageStep,
} from "@/domain/thermal-storage-calculation";

import {
  CONFIG,
} from "@/config/config";

import {
  SUNOPT_ACCELERATION_SCOPE,
} from "./global-acceleration-scope";

import {
  getSunOptNativeBinding,
} from "./rust-acceleration-runtime";


const ENERGY_EPSILON =
  1e-6;


export interface PeakShavingCausalAcceleratorInput {
  weatherData:
    WeatherData;

  futureDemandEnergyByIndex:
    number[];

  storage:
    ThermalStorage;

  timeStepHours:
    number;

  householdReservePower:
    number;
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


export interface PeakShavingCausalResult {
  cutLevelPower:
    number;

  normalPlanSurvives:
    boolean;

  maximumPvPlanSurvives:
    boolean;

  survivalChecks:
    number;

  binarySearchIterations:
    number;

  usedNative:
    boolean;
}


export interface PeakShavingCausalAccelerator {
  run(
    input:
      PeakShavingCausalRunInput,
  ): PeakShavingCausalResult;
}


interface NativePeakShavingCausalResult {
  cutLevelPower:
    number;

  normalPlanSurvives:
    boolean;

  maximumPvPlanSurvives:
    boolean;

  survivalChecks:
    number;

  binarySearchIterations:
    number;
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


function validateNativeResult(
  value: unknown,
): NativePeakShavingCausalResult {

  if (
    value === null ||
    typeof value !== "object"
  ) {
    throw new Error(
      "Native peak-shaving causal kernel returned no object.",
    );
  }


  const candidate =
    value as Record<
      string,
      unknown
    >;


  const numericKeys = [
    "cutLevelPower",
    "survivalChecks",
    "binarySearchIterations",
  ] as const;


  for (
    const key of numericKeys
  ) {

    if (
      typeof candidate[key] !==
      "number"
    ) {
      throw new Error(
        `Native causal result "${key}" is not numeric.`,
      );
    }
  }


  const booleanKeys = [
    "normalPlanSurvives",
    "maximumPvPlanSurvives",
  ] as const;


  for (
    const key of booleanKeys
  ) {

    if (
      typeof candidate[key] !==
      "boolean"
    ) {
      throw new Error(
        `Native causal result "${key}" is not boolean.`,
      );
    }
  }


  return candidate as unknown as
    NativePeakShavingCausalResult;
}


function wrapIndex(
  index: number,
  length: number,
): number {

  return (
    (index % length) +
    length
  ) % length;
}


/**
 * Creates the causal 48 h survival accelerator.
 *
 * The expensive annual arrays are converted once to compact
 * Float64Array storage.
 *
 * One run() call performs the complete causal Waterline correction.
 * The native backend therefore crosses the JS/native boundary only
 * once per 15-minute controller step.
 */
export function createPeakShavingCausalAccelerator(
  input:
    PeakShavingCausalAcceleratorInput,
): PeakShavingCausalAccelerator {

  const {
    weatherData,
    futureDemandEnergyByIndex,
    storage,
    timeStepHours,
    householdReservePower,
  } = input;


  if (
    weatherData.length === 0
  ) {
    throw new Error(
      "weatherData must not be empty.",
    );
  }


  if (
    futureDemandEnergyByIndex.length !==
    weatherData.length
  ) {
    throw new Error(
      "Demand and weather arrays must have equal length.",
    );
  }


  const exportPowerByIndex =
    Float64Array.from(
      weatherData,
      (point) =>
        Math.max(
          0,
          point.pvPower -
            householdReservePower,
        ),
    );


  const demandEnergyByIndex =
    Float64Array.from(
      futureDemandEnergyByIndex,
      (value) =>
        Math.max(
          0,
          value,
        ),
    );


  /*
   * Static storage constants passed to the native kernel.
   *
   * They are derived through the canonical domain helpers once.
   */
  const storageTemperatureSpan =
    storage.maxTemperature -
    storage.minTemperature;


  const storageMaximumEnergy =
    calculateStorageEnergyAtTemperature(
      storage,
      storage.maxTemperature,
    );


  const storageEnergyPerKelvin =
    storageMaximumEnergy /
    storageTemperatureSpan;


  const storageConductanceWPerKelvin =
    calculateStorageThermalConductance(
      storage,
    );


  const storageInsulation =
    storage.insulation ??
    CONFIG.storage.insulation;


  const ambientTemperature =
    storageInsulation
      .ambientTemperature;


  const dailyHeatLoss =
    storage.insulation === undefined &&
    storage.dailyHeatLoss !== undefined
      ? storage.dailyHeatLoss
      : -1;


  /*
   * ==============================================================
   * CUDA – PERSISTENT GPU-RESIDENT CAUSAL WORKER
   * ==============================================================
   *
   * The annual export/demand arrays are uploaded exactly once here.
   *
   * Every later run() sends only the small dynamic causal state.
   *
   * This does NOT batch dependent 15-minute steps. The annual
   * simulation remains causal and numerically ordered.
   *
   * If CUDA is unavailable or fails, the worker is disabled and the
   * existing Rust/Cranelift -> TypeScript fallback path remains active.
   */
  let cudaWorker:
    CudaCausalWorkerLike | null =
    null;

  let cudaWorkerDisabled =
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


  function maybeReportPerformanceProfile(
    currentIndex:
      number,
  ): void {

    if (
      performanceProfileReported ||
      currentIndex !==
        exportPowerByIndex.length - 1
    ) {
      return;
    }


    performanceProfileReported =
      true;


    let gpuProfile:
      CudaCausalNativeProfile | null =
      null;


    if (
      cudaWorker !== null
    ) {

      try {

        gpuProfile =
          cudaWorker.profile();

      } catch (error) {

        console.warn(
          "[SunOpt] Could not read CUDA causal performance profile.",
          error,
        );
      }
    }


    console.info(
      "[SunOpt] CUDA causal performance profile:",
      {
        annualSteps:
          exportPowerByIndex.length,

        cudaCalls,
        cudaFailures,

        cpuNativeFallbackCalls,

        typescriptFallbackCalls,

        gpu:
          gpuProfile,
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
          exportPowerByIndex,
          demandEnergyByIndex,
          0,
        );


      console.log(
        "[SunOpt] CUDA causal worker active.",
        {
          deviceId:
            0,

          residentAnnualValues:
            exportPowerByIndex.length,
        },
      );
    }

  } catch (error) {

    cudaWorker =
      null;

    cudaWorkerDisabled =
      true;

    cudaFailureReported =
      true;


    console.warn(
      "[SunOpt] CUDA causal worker unavailable; using CPU acceleration.",
      error,
    );
  }


  /*
   * ==============================================================
   * TYPESCRIPT REFERENCE / FALLBACK
   * ==============================================================
   *
   * This deliberately uses the canonical storage function.
   *
   * It therefore remains the numerical reference against which the
   * native implementation is verified.
   */
  function runReference(
    dynamic:
      PeakShavingCausalRunInput,
  ): NativePeakShavingCausalResult {

    let survivalChecks =
      0;

    let binarySearchIterations =
      0;


    const reserveFloorEnergy =
      Math.min(
        dynamic.targetStorageEnergy,
        dynamic.minimumReserveEnergy,
      );


    const clampedCurrentStorageEnergy =
      Math.min(
        dynamic.targetStorageEnergy,
        Math.max(
          0,
          dynamic.currentStorageEnergy,
        ),
      );


    function hpPowerAtCutLevel(
      exportPower: number,
      cutLevelPower: number,
    ): number {

      return Math.min(
        dynamic
          .maximumHeatPumpElectricalPower,

        Math.max(
          0,
          exportPower -
            cutLevelPower,
        ),
      );
    }


    function survivesAtCutLevel(
      cutLevelPower: number,
    ): boolean {

      survivalChecks += 1;


      let simulatedEnergy =
        clampedCurrentStorageEnergy;


      for (
        let offset = 0;
        offset < dynamic.horizonSteps;
        offset++
      ) {

        const index =
          wrapIndex(
            dynamic.currentIndex +
              offset,
            exportPowerByIndex.length,
          );


        const exportPower =
          exportPowerByIndex[index] ??
          0;


        const hpPower =
          hpPowerAtCutLevel(
            exportPower,
            cutLevelPower,
          );


        const hpThermalEnergy =
          hpPower *
          dynamic.cop *
          timeStepHours;


        /*
         * Current demand/loss has already happened before the
         * controller receives currentStorageEnergy.
         */
        if (
          offset === 0
        ) {

          simulatedEnergy =
            Math.min(
              dynamic
                .targetStorageEnergy,

              simulatedEnergy +
                hpThermalEnergy,
            );

          continue;
        }


        const demandEnergy =
          demandEnergyByIndex[index] ??
          0;


        const storageStep =
          calculateThermalStorageStep({
            storage,

            currentEnergy:
              simulatedEnergy,

            thermalInputEnergy:
              0,

            thermalOutputEnergy:
              demandEnergy,

            timeStepHours,
          });


        const energyBeforeForecastCharge =
          storageStep
            .state
            .storedEnergy;


        if (
          storageStep.unmetDemand >
          ENERGY_EPSILON
        ) {
          return false;
        }


        if (
          reserveFloorEnergy -
            energyBeforeForecastCharge >
          ENERGY_EPSILON
        ) {
          return false;
        }


        simulatedEnergy =
          Math.min(
            dynamic
              .targetStorageEnergy,

            energyBeforeForecastCharge +
              hpThermalEnergy,
          );
      }


      return true;
    }


    let cutLevelPower =
      dynamic.energyCutLevel;


    const normalPlanSurvives =
      survivesAtCutLevel(
        dynamic.energyCutLevel,
      );


    /*
     * Only evaluate Waterline zero when the normal energy Waterline
     * actually fails.
     */
    let maximumPvPlanSurvives =
      normalPlanSurvives;


    if (
      !normalPlanSurvives
    ) {

      maximumPvPlanSurvives =
        survivesAtCutLevel(
          0,
        );


      if (
        !maximumPvPlanSurvives
      ) {

        cutLevelPower =
          0;

      } else {

        let lowerCut =
          0;

        let upperCut =
          dynamic.energyCutLevel;


        for (
          let iteration = 0;
          iteration < 20;
          iteration++
        ) {

          binarySearchIterations +=
            1;


          const middleCut =
            (
              lowerCut +
              upperCut
            ) /
            2;


          if (
            survivesAtCutLevel(
              middleCut,
            )
          ) {

            lowerCut =
              middleCut;

          } else {

            upperCut =
              middleCut;
          }
        }


        cutLevelPower =
          lowerCut;
      }
    }


    return {
      cutLevelPower,

      normalPlanSurvives,

      maximumPvPlanSurvives,

      survivalChecks,

      binarySearchIterations,
    };
  }


  function run(
    dynamic:
      PeakShavingCausalRunInput,
  ): PeakShavingCausalResult {

    /*
     * ============================================================
     * CUDA FIRST
     * ============================================================
     *
     * The annual arrays already live in VRAM. Only the dynamic state
     * below crosses the JS/native boundary for this 15-minute step.
     */
    if (
      cudaWorker !== null &&
      !cudaWorkerDisabled
    ) {

      try {

        const cudaResult =
          validateNativeResult(
            cudaWorker.run({
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

              storageEnergyPerKelvin,

              storageConductanceWPerKelvin,

              storageMinTemperature:
                storage.minTemperature,

              ambientTemperature,

              dailyHeatLoss,
            }),
          );


        cudaCalls +=
          1;


        maybeReportPerformanceProfile(
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

        cudaWorkerDisabled =
          true;


        if (
          !cudaFailureReported
        ) {

          cudaFailureReported =
            true;


          console.warn(
            "[SunOpt] CUDA causal worker failed; disabling it and falling back to CPU acceleration.",
            error,
          );
        }
      }
    }


    /*
     * ============================================================
     * EXISTING CPU NATIVE FALLBACK
     * ============================================================
     *
     * Rust/Cranelift remains untouched. If it is unavailable too,
     * runReference() below remains the canonical TypeScript fallback.
     */
    const nativeResult =
      SUNOPT_ACCELERATION_SCOPE.run<
        NativePeakShavingCausalResult
      >(
        "peakShavingCausal",

        [
          exportPowerByIndex,
          demandEnergyByIndex,

          dynamic.currentIndex,
          dynamic.horizonSteps,

          dynamic.currentStorageEnergy,
          dynamic.minimumReserveEnergy,
          dynamic.targetStorageEnergy,

          dynamic
            .maximumHeatPumpElectricalPower,

          dynamic.cop,
          dynamic.energyCutLevel,

          timeStepHours,

          storageEnergyPerKelvin,

          storageConductanceWPerKelvin,

          storage.minTemperature,

          ambientTemperature,

          dailyHeatLoss,
        ],

        validateNativeResult,
      );


    if (
      nativeResult !== null
    ) {

      cpuNativeFallbackCalls +=
        1;


      maybeReportPerformanceProfile(
        dynamic.currentIndex,
      );


      return {
        ...nativeResult,

        usedNative:
          true,
      };
    }


    const reference =
      runReference(
        dynamic,
      );


    typescriptFallbackCalls +=
      1;


    maybeReportPerformanceProfile(
      dynamic.currentIndex,
    );


    return {
      ...reference,

      usedNative:
        false,
    };
  }


  return {
    run,
  };
}
