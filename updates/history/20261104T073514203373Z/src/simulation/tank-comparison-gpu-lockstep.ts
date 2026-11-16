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
  calculateHotWaterForecastTarget,
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
  calculatePlasmaWaveAllPackagesForecast,
} from "./plasma-wave-user-forecast-all-packages";

import {
  getPlasmaWaveHistoricalSeries,
  type PlasmaWaveHistoricalSeries,
} from "./plasma-wave-historical-series-cache";

import {
  createNoSunshineAccelerator,
} from "./rust-acceleration-adapters";

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


interface CudaCausalWorkerLike {
  runBatch(
    jobs: CudaCausalJob[],
  ): unknown[];

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


interface TankState {
  storageVolume:
    number;

  storage:
    ThermalStorage;

  storageCapacity:
    number;

  storageState:
    ThermalStorageState;

  historicalSeries:
    PlasmaWaveHistoricalSeries;

  allPackagesForecast:
    ReturnType<
      typeof calculatePlasmaWaveAllPackagesForecast
    >;

  noSunshine:
    ReturnType<
      typeof createNoSunshineAccelerator
    >;

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

    exportPowerByIndex:
      exportPower,
  } = series;


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


  const horizonSteps =
    Math.min(
      Math.ceil(
        planningHorizonHours /
        timeStepHours,
      ),
      length,
    );


  let maximumExportPower =
    0;


  for (
    let offset = 0;
    offset < horizonSteps;
    offset += 1
  ) {

    const index =
      (
        currentIndex +
        offset
      ) %
      length;


    maximumExportPower =
      Math.max(
        maximumExportPower,
        exportPower[index] ??
          0,
      );
  }


  function electricalEnergyAtCutLevel(
    cutLevelPower:
      number,
  ): number {

    let energy =
      0;


    for (
      let offset = 0;
      offset < horizonSteps;
      offset += 1
    ) {

      const index =
        (
          currentIndex +
          offset
        ) %
        length;


      const availablePower =
        Math.max(
          0,
          (exportPower[index] ?? 0) -
            cutLevelPower,
        );


      energy +=
        Math.min(
          maximumHeatPumpElectricalPower,
          availablePower,
        ) *
        timeStepHours;
    }


    return energy;
  }


  const requestedElectricalEnergy =
    requestedChargeThermalEnergy /
    cop;


  const maximumSchedulableElectricalEnergy =
    electricalEnergyAtCutLevel(
      0,
    );


  const targetElectricalEnergy =
    Math.min(
      requestedElectricalEnergy,
      maximumSchedulableElectricalEnergy,
    );


  let energyCutLevel =
    maximumExportPower;


  if (
    targetElectricalEnergy > 0 &&
    maximumExportPower > 0
  ) {

    if (
      targetElectricalEnergy >=
      maximumSchedulableElectricalEnergy
    ) {

      energyCutLevel =
        0;

    } else {

      let lowerCut =
        0;

      let upperCut =
        maximumExportPower;


      for (
        let iteration = 0;
        iteration < 32;
        iteration += 1
      ) {

        const middleCut =
          (
            lowerCut +
            upperCut
          ) /
          2;


        const energyAtMiddle =
          electricalEnergyAtCutLevel(
            middleCut,
          );


        if (
          energyAtMiddle >
          targetElectricalEnergy
        ) {

          lowerCut =
            middleCut;

        } else {

          upperCut =
            middleCut;
        }
      }


      energyCutLevel =
        (
          lowerCut +
          upperCut
        ) /
        2;
    }
  }


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

      energyCutLevel,

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


        const allPackagesForecast =
          calculatePlasmaWaveAllPackagesForecast({
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

            lossReferenceTemperature:
              storage.minTemperature,

            safetyFactor:
              input.control
                .forecastSafetyFactor,
          });


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
          createNoSunshineAccelerator({
            historicalSeries,

            heatPump:
              input.heatPump,

            storage,

            horizonHours:
              CONFIG.control
                .noSunshineHorizonHours,

            protectedHorizonHours:
              CONFIG.control
                .noSunshineProtectedHorizonHours,

            initialWaterlinePower:
              CONFIG.control
                .noSunshineInitialWaterlinePower,

            targetTemperature,
          });


        return {
          storageVolume,

          storage,

          storageCapacity:
            calculateStorageCapacity(
              storage,
            ),

          storageState:
            createInitialStorageState(
              storage,
            ),

          historicalSeries,

          allPackagesForecast,

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
    },
  );


  let schedulerPreparationMs =
    0;

  let noSunshineMs =
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

    const jobs:
      CudaCausalJob[] =
      [];

    const prepared:
      Array<{
        baseline:
          ReturnType<
            typeof calculateHotWaterForecastTarget
          >;

        peak:
          PreparedPeakShaving;

        noSunshine:
          ReturnType<
            TankState["noSunshine"]["run"]
          >;

        storageEnergyAfterMandatoryReserve:
          number;
      }> =
      [];


    for (
      const tank of tanks
    ) {

      const forecastPoint =
        tank
          .allPackagesForecast
          .points[
            currentIndex
          ];


      if (
        !forecastPoint
      ) {
        throw new Error(
          `Plasma-Wave forecast missing at index ${currentIndex} for ${tank.storageVolume} L.`,
        );
      }


      const baseline =
        calculateHotWaterForecastTarget({
          currentIndex,

          weatherData:
            input.weatherData,

          hotWaterData:
            input.hotWaterData,

          heatPump:
            input.heatPump,

          storage:
            tank.storage,

          control:
            input.control,

          timeStepHours:
            input.timeStepHours,
        });


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

          forecastPoint
            .lossAdjustedHotWaterForecastEnergy,

          baseline.minimumReserveEnergy,

          input.heatPump,

          planningHorizonHours,

          targetTemperature,
        );


      schedulerPreparationMs +=
        performance.now() -
        schedulerStart;


      const noSunshineStart =
        performance.now();


      const noSunshine =
        tank.noSunshine.run({
          currentIndex,

          currentStorageEnergy:
            storageEnergyAfterMandatoryReserve,
        });


      noSunshineMs +=
        performance.now() -
        noSunshineStart;


      jobs.push(
        peak.job,
      );


      prepared.push({
        baseline,

        peak,

        noSunshine,

        storageEnergyAfterMandatoryReserve,
      });
    }


    const batchStart =
      performance.now();


    const rawResults =
      cudaWorker.runBatch(
        jobs,
      );


    cudaBatchWallMs +=
      performance.now() -
      batchStart;


    batchCalls +=
      1;

    causalJobs +=
      jobs.length;


    if (
      rawResults.length !==
      tanks.length
    ) {
      throw new Error(
        `CUDA lockstep returned ${rawResults.length} results for ${tanks.length} tanks.`,
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

      const rawResult =
        rawResults[tankIndex];


      if (
        !tank ||
        !state
      ) {
        throw new Error(
          `Lockstep state missing for tank index ${tankIndex}.`,
        );
      }


      const causal =
        validateCudaResult(
          rawResult,
        );


      const combinedWaterline =
        calculateCombinedWaterline({
          userForecastWaterlinePower:
            causal.cutLevelPower,

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
