// src/simulation/tank-comparison-gpu-lockstep.ts

import type {
  HeatPump,
  HotWaterControl,
  HotWaterData,
  ThermalStorage,
  ThermalStorageState,
  WeatherData,
} from "@/domain";

import {
  calculateHeatPumpStep,
  calculateStorageCapacity,
  calculateStorageEnergyAtTemperature,
  createInitialStorageState,
} from "@/domain";

import {
  calculateStorageThermalConductance,
  calculateThermalStorageStep,
} from "@/domain/thermal-storage-calculation";

import {
  CONFIG,
  HOT_WATER_TEST_CONFIG,
} from "@/config/config";

import {
  calculatePlasmaWave15MinuteModel,
} from "./plasma-wave-15min-model";

import {
  getPlasmaWaveHistoricalSeries,
  type PlasmaWaveHistoricalSeries,
} from "./plasma-wave-historical-series-cache";

import {
  calculateCombinedWaterline,
} from "./combined-waterline-scheduler";

import {
  getSunOptNativeBinding,
} from "./rust-acceleration-runtime";

import type {
  TankComparisonPoint,
} from "@/types/tank-comparison";


interface CudaCausalJob {
  currentIndex: number;
  horizonSteps: number;

  currentStorageEnergy: number;
  minimumReserveEnergy: number;
  targetStorageEnergy: number;

  maximumHeatPumpElectricalPower: number;
  cop: number;

  requestedElectricalEnergy?:
    number;

  energyCutLevel: number;

  timeStepHours: number;

  storageEnergyPerKelvin: number;
  storageConductanceWPerKelvin: number;
  storageMinTemperature: number;
  ambientTemperature: number;
  dailyHeatLoss: number;
}


interface CudaCausalResult {
  cutLevelPower: number;
  normalPlanSurvives: boolean;
  maximumPvPlanSurvives: boolean;
  survivalChecks: number;
  binarySearchIterations: number;
}


interface CudaCausalProfile {
  causalCalls: number;
  causalJobs: number;

  parameterH2dMs: number;
  kernelMs: number;
  resultD2hMs: number;
  gpuTotalMs: number;
  hostWaitMs: number;
  wallMs: number;

  averageKernelUsPerJob: number;
  averageGpuUsPerJob: number;

}


interface CudaNoSunshineTankJob {
  horizonSteps:
    number;

  protectedSteps:
    number;

  timeStepHours:
    number;

  maximumHeatPumpElectricalPower:
    number;

  cop:
    number;

  initialWaterlinePower:
    number;

  targetStorageEnergy:
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


interface CudaCausalWorkerLike {
  runBatch(
    jobs: CudaCausalJob[],
  ): unknown[];

  runLockstepPacked(
    currentIndex:
      number,

    horizonSteps:
      number,

    packed:
      Float64Array,
  ): number[];

  precomputeNoSunshineWaterlines(
    tanks:
      CudaNoSunshineTankJob[],
  ): number[];

  profile():
    CudaCausalProfile;

  resetProfile?():
    void;
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


export interface TankComparisonGpuLockstepInput {
  weatherData:
    WeatherData;

  hotWaterData:
    HotWaterData;

  heatPump:
    HeatPump;

  control:
    HotWaterControl;

  timeStepHours:
    number;

  pvProduction:
    number;

  gridPriceChfPerKWh:
    number;

  feedInPriceChfPerKWh:
    number;

  storageVolumes:
    number[];
}


export interface TankComparisonGpuLockstepProfile {
  tanks:
    number;

  annualSteps:
    number;

  batchCalls:
    number;

  causalJobs:
    number;

  preparationMs:
    number;

  requiredInventoryPreparationMs:
    number;

  baselineReservePrecomputeMs:
    number;

  schedulerPreparationMs:
    number;

  noSunshineMs:
    number;

  cudaBatchWallMs:
    number;

  physicalSimulationMs:
    number;

  totalMs:
    number;

  gpu:
    CudaCausalProfile;
}


export interface TankComparisonGpuLockstepResult {
  points:
    TankComparisonPoint[];

  profile:
    TankComparisonGpuLockstepProfile;
}


interface PreparedPeakShaving {
  job:
    CudaCausalJob;

  cop:
    number;

  maximumHeatPumpElectricalPower:
    number;

  requestedChargeThermalEnergy:
    number;
}


/*
 * ================================================================
 * ANALYTIC NO-SUNSHINE REVERSE STORAGE
 * ================================================================
 *
 * The previous no-sunshine path called the Oxc/Cranelift kernel once
 * per tank and quarter hour:
 *
 *   35,040 * 28 = 981,120 native calls/year
 *
 * Inside every 7-day reverse forecast the canonical algorithm then
 * solved EVERY storage reverse step with a 32-iteration binary
 * search.
 *
 * For SunOpt's current fully mixed storage model that inner binary
 * search is unnecessary.
 *
 * Forward storage loss is either:
 *
 *   legacy:
 *     fixed kWh/day
 *
 * or:
 *
 *   Newton cooling:
 *     loss =
 *       G * (T - Tambient) * dt / 1000
 *
 * with:
 *
 *     T = Tmin + E / C
 *
 * Therefore, while the storage is above the physical zero boundary,
 * the post-loss energy is an affine function of E and can be inverted
 * analytically.
 *
 * To preserve the exact numerical semantics of the old 32-step binary
 * search, the analytic threshold is quantised to the same 2^32 grid.
 */

const NO_SUNSHINE_ENERGY_EPSILON =
  1e-6;

const NO_SUNSHINE_BINARY_GRID =
  4294967296;


interface NoSunshineAnalyticPlan {
  historicalSeries:
    PlasmaWaveHistoricalSeries;

  horizonSteps:
    number;

  protectedSteps:
    number;

  maximumHeatPumpElectricalPower:
    number;

  cop:
    number;

  initialWaterlinePower:
    number;

  targetStorageEnergy:
    number;

  storageCapacity:
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


interface NoSunshineCurrentResult {
  currentWaterlinePower:
    number;
}


function createNoSunshineAnalyticPlan(
  historicalSeries:
    PlasmaWaveHistoricalSeries,

  heatPump:
    HeatPump,

  storage:
    ThermalStorage,

  targetTemperature:
    number,
): NoSunshineAnalyticPlan {

  const {
    length,
    timeStepHours,
    cop,
  } =
    historicalSeries;


  if (
    length === 0 ||
    cop <= 0
  ) {
    throw new Error(
      "Invalid historical series for analytic no-sunshine.",
    );
  }


  const physicalTargetTemperature =
    Math.min(
      targetTemperature,
      storage.maxTemperature,
      heatPump.maxFlowTemperature,
    );


  const constants =
    storageKernelConstants(
      storage,
    );


  return {
    historicalSeries,

    horizonSteps:
      Math.min(
        Math.ceil(
          CONFIG.control
            .noSunshineHorizonHours /
          timeStepHours,
        ),
        length,
      ),

    protectedSteps:
      Math.min(
        Math.ceil(
          CONFIG.control
            .noSunshineProtectedHorizonHours /
          timeStepHours,
        ),
        length,
      ),

    maximumHeatPumpElectricalPower:
      heatPump.thermalPower /
      cop,

    cop,

    initialWaterlinePower:
      CONFIG.control
        .noSunshineInitialWaterlinePower,

    targetStorageEnergy:
      calculateStorageEnergyAtTemperature(
        storage,
        physicalTargetTemperature,
      ),

    storageCapacity:
      calculateStorageCapacity(
        storage,
      ),

    ...constants,
  };
}


function simulateNoChargeStorageStep(
  plan:
    NoSunshineAnalyticPlan,

  currentEnergy:
    number,

  demandEnergy:
    number,
) {

  const timeStepHours =
    plan.historicalSeries
      .timeStepHours;


  let storedEnergy =
    Math.min(
      plan.storageCapacity,
      Math.max(
        0,
        currentEnergy,
      ),
    );


  let requestedHeatLoss =
    0;


  if (
    plan.dailyHeatLoss >=
    0
  ) {

    requestedHeatLoss =
      plan.dailyHeatLoss *
      timeStepHours /
      24;

  } else {

    const currentTemperature =
      plan.storageMinTemperature +
      storedEnergy /
        plan.storageEnergyPerKelvin;


    const deltaTemperature =
      Math.max(
        0,
        currentTemperature -
          plan.ambientTemperature,
      );


    requestedHeatLoss =
      plan
        .storageConductanceWPerKelvin *
      deltaTemperature *
      timeStepHours /
      1000;
  }


  const heatLoss =
    Math.min(
      storedEnergy,
      requestedHeatLoss,
    );


  storedEnergy -=
    heatLoss;


  const dischargedEnergy =
    Math.min(
      demandEnergy,
      storedEnergy,
    );


  storedEnergy -=
    dischargedEnergy;


  const unmetDemand =
    Math.max(
      0,
      demandEnergy -
        dischargedEnergy,
    );


  return {
    storedEnergy,
    unmetDemand,
    heatLoss,
  };
}


function calculateRequiredEnergyBeforeStepAnalytic(
  plan:
    NoSunshineAnalyticPlan,

  requiredEnergyAfterStep:
    number,

  demandEnergy:
    number,
) {

  const maximumStep =
    simulateNoChargeStorageStep(
      plan,
      plan.targetStorageEnergy,
      demandEnergy,
    );


  const maximumIsSufficient =
    maximumStep.unmetDemand <=
      NO_SUNSHINE_ENERGY_EPSILON &&
    maximumStep.storedEnergy +
      NO_SUNSHINE_ENERGY_EPSILON >=
      requiredEnergyAfterStep;


  if (
    !maximumIsSufficient
  ) {

    return {
      requiredEnergyBeforeStep:
        plan.targetStorageEnergy,

      coolingLossEnergy:
        maximumStep.heatLoss,

      unavoidableThermalDeficit:
        maximumStep.unmetDemand +
        Math.max(
          0,
          requiredEnergyAfterStep -
            maximumStep.storedEnergy,
        ),
    };
  }


  /*
   * Minimal energy that must remain AFTER passive loss.
   *
   * This reproduces the two epsilon checks of the canonical
   * isSufficient() predicate:
   *
   *   unmetDemand <= 1e-6
   *   storedEnergy + 1e-6 >= requiredAfter
   */
  const requiredAfterLoss =
    requiredEnergyAfterStep <=
      NO_SUNSHINE_ENERGY_EPSILON

      ? Math.max(
          0,
          demandEnergy -
            NO_SUNSHINE_ENERGY_EPSILON,
        )

      : Math.max(
          0,
          demandEnergy +
            requiredEnergyAfterStep -
            NO_SUNSHINE_ENERGY_EPSILON,
        );


  let exactRequiredEnergy =
    0;


  if (
    requiredAfterLoss >
    0
  ) {

    const timeStepHours =
      plan.historicalSeries
        .timeStepHours;


    if (
      plan.dailyHeatLoss >=
      0
    ) {

      const fixedHeatLoss =
        plan.dailyHeatLoss *
        timeStepHours /
        24;


      exactRequiredEnergy =
        requiredAfterLoss +
        fixedHeatLoss;

    } else {

      /*
       * Newton cooling:
       *
       *   loss(E) =
       *     alpha *
       *     (Tmin - Tambient + E/C)
       *
       *   afterLoss(E) =
       *     (1 - alpha/C) E
       *     - alpha (Tmin - Tambient)
       */
      const alpha =
        plan
          .storageConductanceWPerKelvin *
        timeStepHours /
        1000;


      const linearFactor =
        alpha /
        plan.storageEnergyPerKelvin;


      const zeroLossThresholdEnergy =
        Math.max(
          0,
          (
            plan.ambientTemperature -
            plan.storageMinTemperature
          ) *
          plan.storageEnergyPerKelvin,
        );


      if (
        requiredAfterLoss <=
        zeroLossThresholdEnergy
      ) {

        exactRequiredEnergy =
          requiredAfterLoss;

      } else {

        const denominator =
          1 -
          linearFactor;


        if (
          denominator <=
          0
        ) {
          throw new Error(
            "Analytic no-sunshine storage inverse is not stable for this timestep.",
          );
        }


        exactRequiredEnergy =
          (
            requiredAfterLoss +
            alpha *
              (
                plan.storageMinTemperature -
                plan.ambientTemperature
              )
          ) /
          denominator;
      }
    }
  }


  exactRequiredEnergy =
    Math.min(
      plan.targetStorageEnergy,
      Math.max(
        0,
        exactRequiredEnergy,
      ),
    );


  /*
   * Preserve the exact upper-bound resolution produced by the old:
   *
   *   for (iteration = 0; iteration < 32; iteration++)
   *
   * binary search.
   *
   * Starting from [0, target], after 32 iterations all possible upper
   * bounds lie on target / 2^32.
   *
   * Even if zero is sufficient, the canonical binary search returns
   * one quantum rather than exactly zero because the lower bound is
   * never evaluated directly.
   */
  const quantum =
    plan.targetStorageEnergy /
    NO_SUNSHINE_BINARY_GRID;


  const requiredEnergyBeforeStep =
    Math.min(
      plan.targetStorageEnergy,

      Math.max(
        quantum,

        Math.ceil(
          exactRequiredEnergy /
          quantum,
        ) *
        quantum,
      ),
    );


  const finalStep =
    simulateNoChargeStorageStep(
      plan,
      requiredEnergyBeforeStep,
      demandEnergy,
    );


  return {
    requiredEnergyBeforeStep,

    coolingLossEnergy:
      finalStep.heatLoss,

    unavoidableThermalDeficit:
      0,
  };
}


function runNoSunshineAnalytic(
  plan:
    NoSunshineAnalyticPlan,

  currentIndex:
    number,

  _currentStorageEnergy:
    number,
): NoSunshineCurrentResult {

  const series =
    plan.historicalSeries;


  const exportPower =
    series.exportPowerByIndex;


  const loadEnergy =
    series.loadEnergyByIndex;


  const timeStepHours =
    series.timeStepHours;


  let requiredEnergyAfterStep =
    0;


  let currentWaterlinePower =
    plan.initialWaterlinePower;


  for (
    let offset =
      plan.horizonSteps -
      1;

    offset >= 0;

    offset -= 1
  ) {

    const dataIndex =
      (
        currentIndex +
        offset
      ) %
      series.length;


    const protectedBy48hForecast =
      offset <
      plan.protectedSteps;


    const bucketDemandEnergy =
      protectedBy48hForecast

        ? 0

        : Math.max(
            0,
            loadEnergy[
              dataIndex
            ] ??
              0,
          );


    const currentExportPower =
      Math.max(
        0,
        exportPower[
          dataIndex
        ] ??
          0,
      );


    const availableHpElectricalPower =
      Math.min(
        currentExportPower,
        plan
          .maximumHeatPumpElectricalPower,
      );


    const availablePvHpThermalEnergy =
      availableHpElectricalPower *
      plan.cop *
      timeStepHours;


    const allocatedHpThermalEnergy =
      Math.min(
        requiredEnergyAfterStep,
        availablePvHpThermalEnergy,
      );


    const allocatedHpElectricalPower =
      allocatedHpThermalEnergy >
        0

        ? allocatedHpThermalEnergy /
          (
            plan.cop *
            timeStepHours
          )

        : 0;


    const inactiveWaterlinePower =
      Math.max(
        plan.initialWaterlinePower,
        currentExportPower,
      );


    const waterlinePower =
      allocatedHpElectricalPower >
        NO_SUNSHINE_ENERGY_EPSILON

        ? Math.max(
            0,
            currentExportPower -
              allocatedHpElectricalPower,
          )

        : inactiveWaterlinePower;


    const remainingRequiredAfterCharge =
      Math.max(
        0,
        requiredEnergyAfterStep -
          allocatedHpThermalEnergy,
      );


    const reverseStorageStep =
      calculateRequiredEnergyBeforeStepAnalytic(
        plan,
        remainingRequiredAfterCharge,
        bucketDemandEnergy,
      );


    requiredEnergyAfterStep =
      reverseStorageStep
        .requiredEnergyBeforeStep;


    if (
      offset ===
      0
    ) {
      currentWaterlinePower =
        waterlinePower;
    }
  }


  return {
    currentWaterlinePower,
  };
}


/*
 * ================================================================
 * FAST LOCKSTEP 48-HOUR REQUIRED-INVENTORY FORECAST
 * ================================================================
 *
 * The generic all-packages forecast computes many diagnostic values
 * that the GPU lockstep path never consumes. The lockstep controller
 * only needs the loss-adjusted 48-hour DHW inventory for each annual
 * currentIndex.
 *
 * Canonical definition:
 *
 *   sum(
 *     protectedHotWaterEnergy[i + offset] /
 *     exp(-offset * dt / tau)
 *   )
 *
 * over the planning horizon.
 *
 * The weight is geometric:
 *
 *   w(offset) = r^offset
 *   r = exp(dt / tau)
 *
 * so consecutive windows satisfy:
 *
 *   S(i+1) =
 *     (S(i) - demand(i)) / r
 *     + demand(i+H) * r^(H-1)
 *
 * This changes the annual forecast from O(N * H) to O(N).
 *
 * To bound floating-point drift, the exact H-term sum is recomputed
 * every FAST_FORECAST_RESEED_STEPS positions.
 */

const FAST_FORECAST_RESEED_STEPS =
  256;


function calculateFastRequiredThermalInventorySeries(
  input: {
    weatherData:
      WeatherData;

    hotWaterData:
      HotWaterData;

    heatPump:
      HeatPump;

    storage:
      ThermalStorage;

    timeStepHours:
      number;

    planningHorizonHours:
      number;

    householdReservePower:
      number;

    safetyFactor:
      number;
  },
): Float64Array {

  const {
    weatherData,
    hotWaterData,
    heatPump,
    storage,
    timeStepHours,
    planningHorizonHours,
    householdReservePower,
    safetyFactor,
  } = input;


  const length =
    weatherData.length;


  if (
    length ===
    0
  ) {
    return new Float64Array(
      0,
    );
  }


  const forecastSteps =
    Math.min(
      Math.ceil(
        planningHorizonHours /
        timeStepHours,
      ),
      length,
    );


  const segmentModel =
    calculatePlasmaWave15MinuteModel({
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


  const protectedDemand =
    new Float64Array(
      length,
    );


  for (
    let index = 0;
    index < length;
    index += 1
  ) {

    protectedDemand[
      index
    ] =
      Math.max(
        0,
        segmentModel
          .points[
            index
          ]
          ?.protectedHotWaterEnergy ??
          0,
      );
  }


  const thermalConductance =
    calculateStorageThermalConductance(
      storage,
    );


  const weights =
    new Float64Array(
      forecastSteps,
    );


  if (
    thermalConductance <=
    0
  ) {

    weights.fill(
      1,
    );

  } else {

    const thermalCapacityJPerK =
      storage.volume *
      4186;


    const timeConstantSeconds =
      thermalCapacityJPerK /
      thermalConductance;


    if (
      timeConstantSeconds <=
      0
    ) {

      weights.fill(
        1,
      );

    } else {

      for (
        let offset = 0;
        offset < forecastSteps;
        offset += 1
      ) {

        const storageSeconds =
          offset *
          timeStepHours *
          3600;


        const retentionFactor =
          Math.exp(
            -storageSeconds /
            timeConstantSeconds,
          );


        weights[
          offset
        ] =
          retentionFactor >
          0

            ? 1 /
              retentionFactor

            : 1;
      }
    }
  }


  const result =
    new Float64Array(
      length,
    );


  function exactWindow(
    currentIndex:
      number,
  ): number {

    let total =
      0;


    for (
      let offset = 0;
      offset < forecastSteps;
      offset += 1
    ) {

      const index =
        (
          currentIndex +
          offset
        ) %
        length;


      total +=
        protectedDemand[
          index
        ] *
        weights[
          offset
        ];
    }


    return total;
  }


  let current =
    exactWindow(
      0,
    );


  const oneStepWeight =
    forecastSteps >
      1
      ? weights[
          1
        ]
      : 1;


  const finalWeight =
    weights[
      forecastSteps -
      1
    ] ??
    1;


  for (
    let currentIndex = 0;
    currentIndex < length;
    currentIndex += 1
  ) {

    if (
      currentIndex >
        0 &&
      currentIndex %
        FAST_FORECAST_RESEED_STEPS ===
        0
    ) {

      current =
        exactWindow(
          currentIndex,
        );
    }


    result[
      currentIndex
    ] =
      current;


    if (
      currentIndex +
        1 >=
      length
    ) {
      continue;
    }


    const leaving =
      protectedDemand[
        currentIndex
      ];


    const enteringIndex =
      (
        currentIndex +
        forecastSteps
      ) %
      length;


    const entering =
      protectedDemand[
        enteringIndex
      ];


    if (
      oneStepWeight ===
      1
    ) {

      current =
        current -
        leaving +
        entering;

    } else {

      current =
        (
          current -
          leaving
        ) /
        oneStepWeight +
        entering *
        finalWeight;
    }
  }


  return result;
}


interface FastBaselineForecast {
  targetCapacity:
    number;

  minimumReserveEnergy:
    number;
}


interface TankState {
  storageVolume:
    number;

  storage:
    ThermalStorage;

  storageCapacity:
    number;

  baselineTargetCapacity:
    number;

  storageState:
    ThermalStorageState;

  historicalSeries:
    PlasmaWaveHistoricalSeries;

  requiredThermalInventoryByIndex:
    Float64Array;

  noSunshine:
    NoSunshineAnalyticPlan;

  gridImport:
    number;

  feedIn:
    number;

  storageLosses:
    number;

  unmetThermalDemand:
    number;
}


function validateCudaResult(
  value:
    unknown,
): CudaCausalResult {

  if (
    value === null ||
    typeof value !==
      "object"
  ) {
    throw new Error(
      "CUDA causal batch returned no result object.",
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
      typeof candidate[key] !==
        "number"
    ) {
      throw new Error(
        `CUDA causal result "${key}" is not numeric.`,
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
      typeof candidate[key] !==
        "boolean"
    ) {
      throw new Error(
        `CUDA causal result "${key}" is not boolean.`,
      );
    }
  }


  return candidate as unknown as
    CudaCausalResult;
}


function createStorage(
  storageVolume:
    number,
): ThermalStorage {

  return {
    volume:
      storageVolume,

    minTemperature:
      60,

    maxTemperature:
      75,

    initialTemperature:
      60,

    insulation: {
      thermalConductivity:
        0.030,

      thickness:
        0.10,

      cylinderHeight:
        1.70,

      ambientTemperature:
        20,
    },
  };
}


function storageKernelConstants(
  storage:
    ThermalStorage,
) {

  const span =
    storage.maxTemperature -
    storage.minTemperature;


  if (
    span <= 0
  ) {
    throw new Error(
      "Storage temperature span must be greater than zero.",
    );
  }


  const maximumEnergy =
    calculateStorageEnergyAtTemperature(
      storage,
      storage.maxTemperature,
    );


  const storageEnergyPerKelvin =
    maximumEnergy /
    span;


  const storageConductanceWPerKelvin =
    calculateStorageThermalConductance(
      storage,
    );


  const insulation =
    storage.insulation ??
    CONFIG.storage.insulation;


  return {
    storageEnergyPerKelvin,

    storageConductanceWPerKelvin,

    storageMinTemperature:
      storage.minTemperature,

    ambientTemperature:
      insulation.ambientTemperature,

    dailyHeatLoss:
      storage.insulation ===
        undefined &&
      storage.dailyHeatLoss !==
        undefined
        ? storage.dailyHeatLoss
        : -1,
  };
}


function assertSharedHistoricalSeries(
  reference:
    PlasmaWaveHistoricalSeries,

  candidate:
    PlasmaWaveHistoricalSeries,

  storageVolume:
    number,
): void {

  if (
    reference.length !==
      candidate.length ||
    reference.timeStepHours !==
      candidate.timeStepHours ||
    reference.householdReservePower !==
      candidate.householdReservePower ||
    reference.cop !==
      candidate.cop
  ) {
    throw new Error(
      `Tank ${storageVolume} L does not share the common historical-series constants.`,
    );
  }


  const referenceExport =
    reference.exportPowerByIndex;

  const candidateExport =
    candidate.exportPowerByIndex;

  const referenceDemand =
    reference.loadEnergyByIndex;

  const candidateDemand =
    candidate.loadEnergyByIndex;


  if (
    referenceExport.length !==
      candidateExport.length ||
    referenceDemand.length !==
      candidateDemand.length
  ) {
    throw new Error(
      `Tank ${storageVolume} L historical arrays have different lengths.`,
    );
  }


  /*
   * Validate all 35,040 positions once during preparation.
   *
   * A single shared CudaCausalWorker can only be correct when the
   * annual export and chronological demand arrays are tank-independent.
   */
  for (
    let index = 0;
    index < reference.length;
    index += 1
  ) {

    if (
      referenceExport[index] !==
        candidateExport[index] ||
      referenceDemand[index] !==
        candidateDemand[index]
    ) {
      throw new Error(
        `Tank-dependent historical causal arrays detected at tank ${storageVolume} L, index ${index}. Central CUDA lockstep would be invalid.`,
      );
    }
  }
}


function preparePeakShaving(
  tank:
    TankState,

  currentIndex:
    number,

  currentStorageEnergy:
    number,

  requiredThermalInventory:
    number,

  minimumReserveEnergy:
    number,

  heatPump:
    HeatPump,

  planningHorizonHours:
    number,

  targetTemperature:
    number,
): PreparedPeakShaving {

  const series =
    tank.historicalSeries;


  const {
    length,
    timeStepHours,
    cop,
  } =
    series;


  if (
    cop <= 0
  ) {
    throw new Error(
      "Heat-pump COP must be greater than zero.",
    );
  }


  const maximumHeatPumpElectricalPower =
    heatPump.thermalPower /
    cop;


  const physicallyReachableTargetTemperature =
    Math.min(
      targetTemperature,
      tank.storage.maxTemperature,
      heatPump.maxFlowTemperature,
    );


  const targetStorageEnergy =
    calculateStorageEnergyAtTemperature(
      tank.storage,
      physicallyReachableTargetTemperature,
    );


  const clampedCurrentStorageEnergy =
    Math.min(
      targetStorageEnergy,
      Math.max(
        0,
        currentStorageEnergy,
      ),
    );


  const availableStorageCapacity =
    Math.max(
      0,
      targetStorageEnergy -
        clampedCurrentStorageEnergy,
    );


  const reserveFloorEnergy =
    Math.min(
      targetStorageEnergy,
      minimumReserveEnergy,
    );


  const missingToRequiredInventory =
    Math.max(
      0,
      requiredThermalInventory -
        clampedCurrentStorageEnergy,
    );


  const requestedChargeThermalEnergy =
    Math.min(
      missingToRequiredInventory,
      availableStorageCapacity,
    );


  const requestedElectricalEnergy =
    requestedChargeThermalEnergy /
    cop;


  const horizonSteps =
    Math.min(
      Math.ceil(
        planningHorizonHours /
        timeStepHours,
      ),
      length,
    );


  return {
    cop,

    maximumHeatPumpElectricalPower,

    requestedChargeThermalEnergy,

    job: {
      currentIndex,

      horizonSteps,

      currentStorageEnergy:
        clampedCurrentStorageEnergy,

      minimumReserveEnergy:
        reserveFloorEnergy,

      targetStorageEnergy,

      maximumHeatPumpElectricalPower,

      cop,

      /*
       * CUDA now performs the canonical 32-iteration energy Waterline
       * search inside the same warp that subsequently executes the
       * causal survival refinement.
       */
      requestedElectricalEnergy,

      /*
       * Legacy/debug fallback value. The fused lockstep kernel
       * overwrites it before causal testing.
       */
      energyCutLevel:
        0,

      timeStepHours,

      ...storageKernelConstants(
        tank.storage,
      ),
    },
  };
}


function executePhysicalStep(
  tank:
    TankState,

  index:
    number,

  forecast: {
    targetEnergy:
      number;

    forecastRequiredEnergy:
      number;

    minimumReserveEnergy:
      number;
  },

  exportAfterHpPower:
    number,

  weatherData:
    WeatherData,

  hotWaterData:
    HotWaterData,

  heatPump:
    HeatPump,

  control:
    HotWaterControl,

  timeStepHours:
    number,
): void {

  const weather =
    weatherData[index];

  const hotWater =
    hotWaterData[index];


  if (
    !weather ||
    !hotWater
  ) {
    throw new Error(
      `Missing simulation data at index ${index}.`,
    );
  }


  const demand =
    hotWater.thermalEnergy;


  const storageEnergyAtStepStart =
    tank.storageState.storedEnergy;


  const storagePreview =
    calculateThermalStorageStep({
      storage:
        tank.storage,

      currentEnergy:
        storageEnergyAtStepStart,

      thermalInputEnergy:
        0,

      thermalOutputEnergy:
        demand,

      timeStepHours,
    });


  const storageStateAfterDemand =
    storagePreview.state;


  const directThermalDemand =
    storagePreview.unmetDemand;


  const missingToReserve =
    Math.max(
      0,
      forecast.minimumReserveEnergy -
        storageStateAfterDemand
          .storedEnergy,
    );


  const availableStorageCapacity =
    Math.max(
      0,
      tank.storageCapacity -
        storageStateAfterDemand
          .storedEnergy,
    );


  const mandatoryStorageCharge =
    Math.min(
      missingToReserve,
      availableStorageCapacity,
    );


  const availablePvPower =
    Math.max(
      0,
      weather.pvPower -
        control.householdReservePower,
    );


  const availablePvEnergy =
    availablePvPower *
    timeStepHours;


  const referenceHeatPumpStep =
    calculateHeatPumpStep(
      heatPump,
      heatPump.thermalPower *
        timeStepHours,
      timeStepHours,
    );


  const cop =
    referenceHeatPumpStep.cop;


  const maxThermalEnergyFromPv =
    Math.min(
      heatPump.thermalPower *
        timeStepHours,

      availablePvEnergy *
        cop,
    );


  const mandatoryThermalEnergy =
    directThermalDemand +
    mandatoryStorageCharge;


  const remainingPvThermalEnergy =
    Math.max(
      0,
      maxThermalEnergyFromPv -
        mandatoryThermalEnergy,
    );


  const storageEnergyAfterMandatoryCharge =
    storageStateAfterDemand
      .storedEnergy +
    mandatoryStorageCharge;


  const missingToForecastTarget =
    Math.max(
      0,
      forecast.targetEnergy -
        storageEnergyAfterMandatoryCharge,
    );


  const forecastPvCharge =
    Math.min(
      missingToForecastTarget,
      remainingPvThermalEnergy,

      Math.max(
        0,
        availableStorageCapacity -
          mandatoryStorageCharge,
      ),
    );


  const requestedThermalEnergy =
    mandatoryThermalEnergy +
    forecastPvCharge;


  const heatPumpStep =
    calculateHeatPumpStep(
      heatPump,
      requestedThermalEnergy,
      timeStepHours,
    );


  const directThermalSupply =
    Math.min(
      directThermalDemand,
      heatPumpStep
        .deliveredThermalEnergy,
    );


  const finalUnmetThermalDemand =
    Math.max(
      0,
      directThermalDemand -
        directThermalSupply,
    );


  const thermalEnergyForStorage =
    Math.max(
      0,
      heatPumpStep
        .deliveredThermalEnergy -
        directThermalSupply,
    );


  const storageStep =
    calculateThermalStorageStep({
      storage:
        tank.storage,

      currentEnergy:
        storageEnergyAtStepStart,

      thermalInputEnergy:
        thermalEnergyForStorage,

      thermalOutputEnergy:
        demand,

      timeStepHours,
    });


  tank.storageState =
    storageStep.state;


  tank.gridImport +=
    Math.max(
      0,
      heatPumpStep.electricalEnergy -
        availablePvEnergy,
    );


  tank.feedIn +=
    Math.max(
      0,
      exportAfterHpPower,
    ) *
    timeStepHours;


  tank.storageLosses +=
    storageStep.heatLoss;


  tank.unmetThermalDemand +=
    finalUnmetThermalDemand;
}


export function runTankComparisonGpuLockstep(
  input:
    TankComparisonGpuLockstepInput,
): TankComparisonGpuLockstepResult {

  const totalStart =
    performance.now();


  if (
    input.weatherData.length ===
      0 ||
    input.weatherData.length !==
      input.hotWaterData.length
  ) {
    throw new Error(
      "GPU lockstep requires equal non-empty weather and hot-water arrays.",
    );
  }


  if (
    input.storageVolumes.length ===
    0
  ) {
    throw new Error(
      "GPU lockstep requires at least one storage volume.",
    );
  }


  const planningHorizonHours =
    HOT_WATER_TEST_CONFIG
      .userForecastHours;


  const targetTemperature =
    75;


  const preparationStart =
    performance.now();


  let requiredInventoryPreparationMs =
    0;


  const tanks:
    TankState[] =
    input.storageVolumes.map(
      (
        storageVolume,
      ) => {

        const storage =
          createStorage(
            storageVolume,
          );


        const requiredInventoryStart =
          performance.now();


        const requiredThermalInventoryByIndex =
          calculateFastRequiredThermalInventorySeries({
            weatherData:
              input.weatherData,

            hotWaterData:
              input.hotWaterData,

            heatPump:
              input.heatPump,

            storage,

            timeStepHours:
              input.timeStepHours,

            planningHorizonHours,

            householdReservePower:
              input.control
                .householdReservePower,

            safetyFactor:
              input.control
                .forecastSafetyFactor,
          });


        requiredInventoryPreparationMs +=
          performance.now() -
          requiredInventoryStart;


        const historicalSeries =
          getPlasmaWaveHistoricalSeries({
            weatherData:
              input.weatherData,

            hotWaterData:
              input.hotWaterData,

            heatPump:
              input.heatPump,

            storage,

            timeStepHours:
              input.timeStepHours,

            householdReservePower:
              input.control
                .householdReservePower,

            safetyFactor:
              input.control
                .forecastSafetyFactor,

            lossReferenceTemperature:
              storage.minTemperature,
          });


        const noSunshine =
          createNoSunshineAnalyticPlan(
            historicalSeries,

            input.heatPump,

            storage,

            targetTemperature,
          );


        return {
          storageVolume,

          storage,

          storageCapacity:
            calculateStorageCapacity(
              storage,
            ),

          /*
           * calculateHotWaterForecastTarget() recomputed this same
           * constant for every quarter hour. Cache it once per tank.
           */
          baselineTargetCapacity:
            calculateStorageEnergyAtTemperature(
              storage,
              input.control
                .pvTargetTemperature,
            ),

          storageState:
            createInitialStorageState(
              storage,
            ),

          historicalSeries,

          requiredThermalInventoryByIndex,

          noSunshine,

          gridImport:
            0,

          feedIn:
            0,

          storageLosses:
            0,

          unmetThermalDemand:
            0,
        };
      },
    );


  /*
   * Reusable low-overhead N-API buffer.
   *
   * Constants are written once; only:
   *   0 currentStorageEnergy
   *   1 minimumReserveEnergy
   *   5 requestedElectricalEnergy
   * change at each annual timestep.
   */
  const LOCKSTEP_PACKED_STRIDE =
    12;


  const packedLockstepParameters =
    new Float64Array(
      tanks.length *
      LOCKSTEP_PACKED_STRIDE,
    );


  for (
    let tankIndex = 0;
    tankIndex <
      tanks.length;
    tankIndex += 1
  ) {

    const tank =
      tanks[
        tankIndex
      ];


    if (
      !tank
    ) {
      throw new Error(
        `Missing tank state at index ${tankIndex}.`,
      );
    }


    const base =
      tankIndex *
      LOCKSTEP_PACKED_STRIDE;


    const constants =
      storageKernelConstants(
        tank.storage,
      );


    packedLockstepParameters[
      base +
      2
    ] =
      calculateStorageEnergyAtTemperature(
        tank.storage,
        Math.min(
          targetTemperature,
          tank.storage
            .maxTemperature,
          input.heatPump
            .maxFlowTemperature,
        ),
      );


    packedLockstepParameters[
      base +
      3
    ] =
      input.heatPump
        .thermalPower /
      tank.historicalSeries
        .cop;


    packedLockstepParameters[
      base +
      4
    ] =
      tank.historicalSeries
        .cop;


    packedLockstepParameters[
      base +
      6
    ] =
      input.timeStepHours;


    packedLockstepParameters[
      base +
      7
    ] =
      constants
        .storageEnergyPerKelvin;


    packedLockstepParameters[
      base +
      8
    ] =
      constants
        .storageConductanceWPerKelvin;


    packedLockstepParameters[
      base +
      9
    ] =
      constants
        .storageMinTemperature;


    packedLockstepParameters[
      base +
      10
    ] =
      constants
        .ambientTemperature;


    packedLockstepParameters[
      base +
      11
    ] =
      constants
        .dailyHeatLoss;
  }


  const referenceSeries =
    tanks[0]
      ?.historicalSeries;


  if (
    !referenceSeries
  ) {
    throw new Error(
      "GPU lockstep could not create historical series.",
    );
  }


  for (
    let index = 1;
    index < tanks.length;
    index += 1
  ) {

    const tank =
      tanks[index];


    if (
      tank
    ) {
      assertSharedHistoricalSeries(
        referenceSeries,
        tank.historicalSeries,
        tank.storageVolume,
      );
    }
  }


  const binding =
    getSunOptNativeBinding();


  const CudaCausalWorker =
    binding?.CudaCausalWorker as
      CudaCausalWorkerConstructor |
      undefined;


  if (
    typeof CudaCausalWorker !==
    "function"
  ) {
    throw new Error(
      "CudaCausalWorker is not available in sunopt-node.",
    );
  }


  const cudaWorker =
    new CudaCausalWorker(
      referenceSeries
        .exportPowerByIndex,

      referenceSeries
        .loadEnergyByIndex,

      0,
    );


  cudaWorker.resetProfile?.();


  const preparationMs =
    performance.now() -
    preparationStart;


  console.log(
    "[SunOpt] GPU lockstep active.",
    {
      deviceId:
        0,

      tanks:
        tanks.length,

      annualSteps:
        referenceSeries.length,

      residentAnnualValues:
        referenceSeries.length,

      cudaContexts:
        1,

      causalKernel:
        "block256-eight-warps-four-candidate-waves",

      energySearchMode:
        "cuda-warp32-fused",

      baselineReserveMode:
        "precomputed-short-reserve-only",

      requiredInventoryMode:
        "rolling-geometric-o-n",

      cudaBatchInterface:
        "packed-f64-cut-only",

      cudaBlocksPerBatch:
        tanks.length,

      cudaThreadsPerBlock:
        256,

      causalIndexMode:
        "incremental-ring-no-modulo",

      causalIndexWidth:
        "shared-window-offset-u32",

      causalWindowMode:
        "shared-256-staged-with-global-fallback",

      causalStepMode:
        "precomputed-heatloss-model",

      causalDataMode:
        "resident-nonnegative-fastpath",

      causalArithmeticMode:
        "precombined-affine-loss-hp",

      causalConstraintMode:
        "single-net-threshold",

      causalHeatLossMode:
        "unified-affine-clamp",

      causalRecurrenceMode:
        "warp-parallel-affine-segment-compose-with-serial-fallback",

      causalSegmentMode:
        "32-contiguous-chunks-five-stage-warp-scan",

      causalCandidateSchedule:
        "8-warps-x4-waves-base32",

      causalSharedLoadMode:
        "precombined-retention-thermal-load",

      causalProfileMode:
        "off-production",

      causalHotParamMode:
        "register-local-scalars",

      causalResultMode:
        "compact-24-byte",

      noSunshineMode:
        "cuda-full-year-matrix",
    },
  );


  /*
   * ================================================================
   * BASELINE MINIMUM-RESERVE TABLE
   * ================================================================
   *
   * The normal baseline helper also evaluates the complete 72-hour
   * forecast. In the Plasma-Wave lockstep path we never consume that
   * long-horizon baseline result; we only use:
   *
   *   - targetCapacity        (constant per tank)
   *   - minimumReserveEnergy (next minimumReserveHours DHW demand)
   *
   * Recomputing the complete baseline 981,120 times therefore spent
   * tens of seconds on values that were immediately discarded.
   *
   * Preserve the canonical minimum-reserve semantics exactly:
   *
   *   offsets 1 .. reserveSteps
   *   sum thermalEnergy
   *   multiply forecastSafetyFactor
   *   clamp to targetCapacity per tank
   *
   * We deliberately use the same simple summation order instead of a
   * rolling-prefix optimisation, so floating-point ordering remains
   * identical to calculateHotWaterForecastTarget().
   */
  const baselineReservePrecomputeStart =
    performance.now();


  const reserveSteps =
    Math.ceil(
      input.control
        .minimumReserveHours /
      input.timeStepHours,
    );


  const rawMinimumReserveByIndex =
    new Float64Array(
      referenceSeries.length,
    );


  for (
    let currentIndex = 0;
    currentIndex <
      referenceSeries.length;
    currentIndex += 1
  ) {

    let minimumReserveEnergy =
      0;


    for (
      let offset = 1;
      offset <= reserveSteps;
      offset += 1
    ) {

      const index =
        (
          currentIndex +
          offset
        ) %
        input.hotWaterData.length;


      const point =
        input.hotWaterData[
          index
        ];


      if (
        !point
      ) {
        throw new Error(
          `Missing hot-water reserve point at index ${index}.`,
        );
      }


      minimumReserveEnergy +=
        point.thermalEnergy;
    }


    rawMinimumReserveByIndex[
      currentIndex
    ] =
      minimumReserveEnergy *
      input.control
        .forecastSafetyFactor;
  }


  const baselineReservePrecomputeMs =
    performance.now() -
    baselineReservePrecomputeStart;


  const noSunshinePrecomputeStart =
    performance.now();


  const noSunshineJobs:
    CudaNoSunshineTankJob[] =
    tanks.map(
      (
        tank,
      ) => {

        const plan =
          tank.noSunshine;


        return {
          horizonSteps:
            plan.horizonSteps,

          protectedSteps:
            plan.protectedSteps,

          timeStepHours:
            plan
              .historicalSeries
              .timeStepHours,

          maximumHeatPumpElectricalPower:
            plan
              .maximumHeatPumpElectricalPower,

          cop:
            plan.cop,

          initialWaterlinePower:
            plan
              .initialWaterlinePower,

          targetStorageEnergy:
            plan
              .targetStorageEnergy,

          storageEnergyPerKelvin:
            plan
              .storageEnergyPerKelvin,

          storageConductanceWPerKelvin:
            plan
              .storageConductanceWPerKelvin,

          storageMinTemperature:
            plan
              .storageMinTemperature,

          ambientTemperature:
            plan
              .ambientTemperature,

          dailyHeatLoss:
            plan.dailyHeatLoss,
        };
      },
    );


  const noSunshineWaterlines =
    cudaWorker
      .precomputeNoSunshineWaterlines(
        noSunshineJobs,
      );


  const expectedNoSunshineValues =
    tanks.length *
    referenceSeries.length;


  if (
    noSunshineWaterlines.length !==
    expectedNoSunshineValues
  ) {
    throw new Error(
      `CUDA no-sunshine matrix returned ${noSunshineWaterlines.length} values; expected ${expectedNoSunshineValues}.`,
    );
  }


  const noSunshineMs =
    performance.now() -
    noSunshinePrecomputeStart;


  console.log(
    "[SunOpt] GPU no-sunshine annual matrix ready.",
    {
      tanks:
        tanks.length,

      annualSteps:
        referenceSeries.length,

      jobs:
        expectedNoSunshineValues,

      durationMs:
        noSunshineMs
          .toFixed(1),
    },
  );


  let schedulerPreparationMs =
    0;

  let cudaBatchWallMs =
    0;

  let physicalSimulationMs =
    0;

  let batchCalls =
    0;

  let causalJobs =
    0;


  for (
    let currentIndex = 0;
    currentIndex <
      referenceSeries.length;
    currentIndex += 1
  ) {

    const prepared:
      Array<{
        baseline:
          FastBaselineForecast;

        peak:
          PreparedPeakShaving;

        noSunshine:
          NoSunshineCurrentResult;

        storageEnergyAfterMandatoryReserve:
          number;
      }> =
      [];


    for (
      let tankIndex = 0;
      tankIndex <
        tanks.length;
      tankIndex += 1
    ) {

      const tank =
        tanks[
          tankIndex
        ];


      if (
        !tank
      ) {
        throw new Error(
          `Missing tank state at index ${tankIndex}.`,
        );
      }


      const requiredThermalInventory =
        tank
          .requiredThermalInventoryByIndex[
            currentIndex
          ];


      if (
        typeof requiredThermalInventory !==
          "number"
      ) {
        throw new Error(
          `Plasma-Wave required inventory missing at index ${currentIndex} for ${tank.storageVolume} L.`,
        );
      }


      const baseline:
        FastBaselineForecast = {

        targetCapacity:
          tank
            .baselineTargetCapacity,

        minimumReserveEnergy:
          Math.min(
            tank
              .baselineTargetCapacity,

            rawMinimumReserveByIndex[
              currentIndex
            ] ??
              0,
          ),
      };


      /*
       * Preserve the current controller semantics:
       * storageState here is the state presented to the controller.
       */
      const missingToMinimumReserve =
        Math.max(
          0,
          baseline
            .minimumReserveEnergy -
            tank.storageState
              .storedEnergy,
        );


      const storageEnergyAfterMandatoryReserve =
        Math.min(
          baseline.targetCapacity,

          tank.storageState
            .storedEnergy +
            missingToMinimumReserve,
        );


      const schedulerStart =
        performance.now();


      const peak =
        preparePeakShaving(
          tank,

          currentIndex,

          storageEnergyAfterMandatoryReserve,

          requiredThermalInventory,

          baseline.minimumReserveEnergy,

          input.heatPump,

          planningHorizonHours,

          targetTemperature,
        );


      schedulerPreparationMs +=
        performance.now() -
        schedulerStart;


      const noSunshineWaterline =
        noSunshineWaterlines[
          tankIndex *
            referenceSeries.length +
          currentIndex
        ];


      if (
        typeof noSunshineWaterline !==
          "number"
      ) {
        throw new Error(
          `Missing no-sunshine waterline for tank ${tankIndex}, index ${currentIndex}.`,
        );
      }


      const noSunshine:
        NoSunshineCurrentResult = {

        currentWaterlinePower:
          noSunshineWaterline,
      };


      const packedBase =
        tankIndex *
        LOCKSTEP_PACKED_STRIDE;


      packedLockstepParameters[
        packedBase +
        0
      ] =
        peak.job
          .currentStorageEnergy;


      packedLockstepParameters[
        packedBase +
        1
      ] =
        peak.job
          .minimumReserveEnergy;


      packedLockstepParameters[
        packedBase +
        5
      ] =
        peak.job
          .requestedElectricalEnergy ??
        0;


      prepared.push({
        baseline,

        peak,

        noSunshine,

        storageEnergyAfterMandatoryReserve,
      });
    }


    const batchStart =
      performance.now();


    const cutLevelPowers =
      cudaWorker
        .runLockstepPacked(
          currentIndex,

          Math.min(
            Math.ceil(
              planningHorizonHours /
              input.timeStepHours,
            ),
            referenceSeries.length,
          ),

          packedLockstepParameters,
        );


    cudaBatchWallMs +=
      performance.now() -
      batchStart;


    batchCalls +=
      1;

    causalJobs +=
      tanks.length;


    if (
      cutLevelPowers.length !==
      tanks.length
    ) {
      throw new Error(
        `CUDA packed lockstep returned ${cutLevelPowers.length} cut levels for ${tanks.length} tanks.`,
      );
    }


    for (
      let tankIndex = 0;
      tankIndex < tanks.length;
      tankIndex += 1
    ) {

      const tank =
        tanks[tankIndex];

      const state =
        prepared[tankIndex];

      const cutLevelPower =
        cutLevelPowers[
          tankIndex
        ];


      if (
        !tank ||
        !state ||
        typeof cutLevelPower !==
          "number"
      ) {
        throw new Error(
          `Lockstep state/result missing for tank index ${tankIndex}.`,
        );
      }


      const combinedWaterline =
        calculateCombinedWaterline({
          userForecastWaterlinePower:
            cutLevelPower,

          noSunshineWaterlinePower:
            state.noSunshine
              .currentWaterlinePower,
        });


      const currentPvExportPower =
        Math.max(
          0,

          referenceSeries
            .exportPowerByIndex[
              currentIndex
            ] ??
            0,
        );


      const scheduledForecastHpElectricalPower =
        Math.min(
          state
            .peak
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
        state.peak.cop *
        input.timeStepHours;


      const exportAfterHpPower =
        Math.max(
          0,

          currentPvExportPower -
            scheduledForecastHpElectricalPower,
        );


      const targetEnergy =
        Math.min(
          state.baseline
            .targetCapacity,

          state
            .storageEnergyAfterMandatoryReserve +
            scheduledForecastThermalEnergy,
        );


      const physicalStart =
        performance.now();


      executePhysicalStep(
        tank,

        currentIndex,

        {
          targetEnergy,

          forecastRequiredEnergy:
            state
              .peak
              .requestedChargeThermalEnergy,

          minimumReserveEnergy:
            state
              .baseline
              .minimumReserveEnergy,
        },

        exportAfterHpPower,

        input.weatherData,

        input.hotWaterData,

        input.heatPump,

        input.control,

        input.timeStepHours,
      );


      physicalSimulationMs +=
        performance.now() -
        physicalStart;
    }
  }


  const totalMs =
    performance.now() -
    totalStart;


  const gpu =
    cudaWorker.profile();


  const points =
    tanks.map(
      (
        tank,
      ): TankComparisonPoint => {

        const selfConsumption =
          Math.max(
            0,

            input.pvProduction -
              tank.feedIn,
          );


        const gridCost =
          tank.gridImport *
          input.gridPriceChfPerKWh;


        const feedInRevenue =
          tank.feedIn *
          input.feedInPriceChfPerKWh;


        return {
          storageVolume:
            tank.storageVolume,

          gridImport:
            tank.gridImport,

          feedIn:
            tank.feedIn,

          selfConsumption,

          storageLosses:
            tank.storageLosses,

          unmetThermalDemand:
            tank.unmetThermalDemand,

          gridCost,

          feedInRevenue,

          netEnergyCost:
            gridCost -
            feedInRevenue,

          /*
           * Every tank advances concurrently through the complete
           * lockstep wall-clock interval.
           */
          simulationDurationMs:
            totalMs,
        };
      },
    );


  const profile:
    TankComparisonGpuLockstepProfile = {

    tanks:
      tanks.length,

    annualSteps:
      referenceSeries.length,

    batchCalls,

    causalJobs,

    preparationMs,

    requiredInventoryPreparationMs,

    baselineReservePrecomputeMs,

    schedulerPreparationMs,

    noSunshineMs,

    cudaBatchWallMs,

    physicalSimulationMs,

    totalMs,

    gpu,
  };


  console.log(
    "[SunOpt] GPU lockstep comparison complete:",
    {
      tanks:
        profile.tanks,

      annualSteps:
        profile.annualSteps,

      batchCalls:
        profile.batchCalls,

      causalJobs:
        profile.causalJobs,

      preparationMs:
        profile.preparationMs
          .toFixed(1),

      requiredInventoryPreparationMs:
        profile
          .requiredInventoryPreparationMs
          .toFixed(1),

      baselineReservePrecomputeMs:
        profile
          .baselineReservePrecomputeMs
          .toFixed(1),

      schedulerPreparationMs:
        profile.schedulerPreparationMs
          .toFixed(1),

      noSunshineMs:
        profile.noSunshineMs
          .toFixed(1),

      cudaBatchWallMs:
        profile.cudaBatchWallMs
          .toFixed(1),

      physicalSimulationMs:
        profile.physicalSimulationMs
          .toFixed(1),

      totalMs:
        profile.totalMs
          .toFixed(1),

      gpu:
        profile.gpu,


    },
  );


  return {
    points,
    profile,
  };
}
