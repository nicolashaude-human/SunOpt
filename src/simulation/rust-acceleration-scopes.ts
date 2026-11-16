// src/simulation/rust-acceleration-scopes.ts

/*
 * ==================================================================
 * SUNOPT GLOBAL RUST ACCELERATION SCOPES
 * ==================================================================
 *
 * THIS is the single TypeScript source file consumed by the
 * Oxc -> Cranelift parser pipeline.
 *
 * Expensive numerical blocks are identified by labels:
 *
 *   rust_noSunshine:
 *   rust_peakShavingCausal:
 *
 * The label is the scope ID.
 *
 * New CPU hot loops belong HERE. They do not get their own .rs file.
 *
 * The exported TypeScript functions also remain executable reference
 * implementations. Native execution must match these results.
 */


const ENERGY_EPSILON =
  1e-6;


function wrapIndex(
  index: number,
  length: number,
): number {

  return (
    (index % length) +
    length
  ) % length;
}


function clamp(
  value: number,
  minimum: number,
  maximum: number,
): number {

  return Math.min(
    maximum,
    Math.max(
      minimum,
      value,
    ),
  );
}


/*
 * ==================================================================
 * SHARED STORAGE PHYSICS
 * ==================================================================
 *
 * These scalar helpers reproduce the storage physics used by the
 * accelerated loops. They deliberately use only numeric inputs.
 */


interface StorageKernelParameters {
  timeStepHours: number;

  targetStorageEnergy: number;

  storageEnergyPerKelvin: number;
  storageConductanceWPerKelvin: number;

  storageMinTemperature: number;
  ambientTemperature: number;

  /*
   * >= 0:
   * legacy constant kWh/day heat loss.
   *
   * < 0:
   * conductance / ambient-temperature model.
   */
  dailyHeatLoss: number;
}


function calculateRequestedHeatLoss(
  storedEnergy: number,
  parameters: StorageKernelParameters,
): number {

  if (
    parameters.dailyHeatLoss >=
    0
  ) {

    return (
      parameters.dailyHeatLoss *
      parameters.timeStepHours /
      24
    );
  }


  const temperatureDelta =
    parameters.storageMinTemperature -
    parameters.ambientTemperature +
    storedEnergy /
    parameters.storageEnergyPerKelvin;


  return (
    parameters.storageConductanceWPerKelvin *
    parameters.timeStepHours /
    1000 *
    Math.max(
      0,
      temperatureDelta,
    )
  );
}


interface ReverseStorageResult {
  requiredBefore: number;
  coolingLoss: number;
  unavoidableDeficit: number;
}


function calculateRequiredBefore(
  requiredAfter: number,
  demandEnergy: number,
  parameters: StorageKernelParameters,
): ReverseStorageResult {

  const requiredSum =
    requiredAfter +
    demandEnergy;


  let candidate: number;


  if (
    parameters.dailyHeatLoss >=
    0
  ) {

    candidate =
      requiredSum <= 0
        ? 0
        : requiredSum +
          parameters.dailyHeatLoss *
          parameters.timeStepHours /
          24;

  } else {

    const q =
      parameters.storageConductanceWPerKelvin *
      parameters.timeStepHours /
      1000;


    const ambientDelta =
      parameters.storageMinTemperature -
      parameters.ambientTemperature;


    const zeroLossTemperatureDelta =
      ambientDelta +
      requiredSum /
      parameters.storageEnergyPerKelvin;


    if (
      zeroLossTemperatureDelta <=
      0
    ) {

      candidate =
        requiredSum;

    } else {

      const denominator =
        1 -
        q /
        parameters.storageEnergyPerKelvin;


      candidate =
        (
          requiredSum +
          q *
          ambientDelta
        ) /
        denominator;
    }
  }


  const capped =
    candidate >
    parameters.targetStorageEnergy;


  const requiredBefore =
    Math.min(
      candidate,
      parameters.targetStorageEnergy,
    );


  const requestedLoss =
    calculateRequestedHeatLoss(
      requiredBefore,
      parameters,
    );


  const coolingLoss =
    Math.min(
      requiredBefore,
      Math.max(
        0,
        requestedLoss,
      ),
    );


  const afterLoss =
    requiredBefore -
    coolingLoss;


  const discharged =
    Math.min(
      demandEnergy,
      afterLoss,
    );


  const unmetDemand =
    Math.max(
      0,
      demandEnergy -
      discharged,
    );


  const storedAfter =
    afterLoss -
    discharged;


  const missingFuture =
    Math.max(
      0,
      requiredAfter -
      storedAfter,
    );


  return {
    requiredBefore,

    coolingLoss,

    unavoidableDeficit:
      capped
        ? unmetDemand +
          missingFuture
        : 0,
  };
}


/*
 * ==================================================================
 * rust_noSunshine
 * ==================================================================
 */


export interface RustNoSunshineScopeInput
  extends StorageKernelParameters {

  pvPower:
    Float64Array;

  loadEnergy:
    Float64Array;

  currentIndex:
    number;

  horizonSteps:
    number;

  protectedSteps:
    number;

  householdReservePower:
    number;

  maximumHeatPumpElectricalPower:
    number;

  cop:
    number;

  initialWaterlinePower:
    number;

  currentStorageEnergy:
    number;
}


export interface RustNoSunshineScopeResult {
  kWhNoSunshine: number;

  bucketDemandEnergy: number;
  bucketCoolingLossEnergy: number;

  preBucketHoldingLossEnergy: number;

  allocatedPvThermalEnergy: number;
  unavoidableThermalDeficit: number;

  requiredStorageEnergyAtStart: number;
  additionalStorageEnergyNeededAtStart: number;

  currentWaterlinePower: number;

  currentScheduledHpElectricalPower: number;
  currentScheduledHpThermalPower: number;
}


export function runRustNoSunshineScope(
  input:
    RustNoSunshineScopeInput,
): RustNoSunshineScopeResult {

  const {
    pvPower,
    loadEnergy,

    currentIndex,

    horizonSteps,
    protectedSteps,

    timeStepHours,

    householdReservePower,

    maximumHeatPumpElectricalPower,

    cop,

    initialWaterlinePower,

    targetStorageEnergy,

    currentStorageEnergy,
  } = input;


  if (
    pvPower.length === 0 ||
    pvPower.length !==
      loadEnergy.length
  ) {
    throw new Error(
      "Invalid annual arrays for rust_noSunshine.",
    );
  }


  let requiredEnergyAfterStep =
    0;


  let bucketDemandEnergyTotal =
    0;

  let bucketCoolingLossEnergyTotal =
    0;

  let preBucketHoldingLossEnergy =
    0;

  let allocatedPvThermalEnergy =
    0;

  let unavoidableThermalDeficit =
    0;


  let currentWaterlinePower =
    initialWaterlinePower;

  let currentScheduledHpElectricalPower =
    0;


  /*
   * --------------------------------------------------------------
   * PARSER / JIT SCOPE ID
   * --------------------------------------------------------------
   */
  rust_noSunshine: {

    for (
      let offset =
        horizonSteps - 1;

      offset >= 0;

      offset--
    ) {

      const dataIndex =
        wrapIndex(
          currentIndex +
          offset,

          pvPower.length,
        );


      const protectedBy48hForecast =
        offset <
        protectedSteps;


      const bucketDemandEnergy =
        protectedBy48hForecast
          ? 0
          : Math.max(
              0,
              loadEnergy[
                dataIndex
              ] ?? 0,
            );


      const exportPower =
        Math.max(
          0,

          (
            pvPower[
              dataIndex
            ] ?? 0
          ) -
          householdReservePower,
        );


      const availableHpElectricalPower =
        Math.min(
          exportPower,
          maximumHeatPumpElectricalPower,
        );


      const availablePvHpThermalEnergy =
        availableHpElectricalPower *
        cop *
        timeStepHours;


      const allocatedHpThermalEnergy =
        Math.min(
          requiredEnergyAfterStep,
          availablePvHpThermalEnergy,
        );


      const allocatedHpElectricalPower =
        allocatedHpThermalEnergy >
        ENERGY_EPSILON
          ? allocatedHpThermalEnergy /
            (
              cop *
              timeStepHours
            )
          : 0;


      allocatedPvThermalEnergy +=
        allocatedHpThermalEnergy;


      const inactiveWaterlinePower =
        Math.max(
          initialWaterlinePower,
          exportPower,
        );


      const waterlinePower =
        allocatedHpElectricalPower >
        ENERGY_EPSILON
          ? Math.max(
              0,

              exportPower -
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
        calculateRequiredBefore(
          remainingRequiredAfterCharge,
          bucketDemandEnergy,
          input,
        );


      if (
        protectedBy48hForecast
      ) {

        preBucketHoldingLossEnergy +=
          reverseStorageStep
            .coolingLoss;

      } else {

        bucketDemandEnergyTotal +=
          bucketDemandEnergy;

        bucketCoolingLossEnergyTotal +=
          reverseStorageStep
            .coolingLoss;
      }


      unavoidableThermalDeficit +=
        reverseStorageStep
          .unavoidableDeficit;


      if (
        offset === 0
      ) {

        currentWaterlinePower =
          waterlinePower;

        currentScheduledHpElectricalPower =
          allocatedHpElectricalPower;
      }


      requiredEnergyAfterStep =
        reverseStorageStep
          .requiredBefore;
    }
  }


  const requiredStorageEnergyAtStart =
    requiredEnergyAfterStep;


  return {
    kWhNoSunshine:
      bucketDemandEnergyTotal +
      bucketCoolingLossEnergyTotal,

    bucketDemandEnergy:
      bucketDemandEnergyTotal,

    bucketCoolingLossEnergy:
      bucketCoolingLossEnergyTotal,

    preBucketHoldingLossEnergy,

    allocatedPvThermalEnergy,

    unavoidableThermalDeficit,

    requiredStorageEnergyAtStart,

    additionalStorageEnergyNeededAtStart:
      Math.max(
        0,

        requiredStorageEnergyAtStart -
        currentStorageEnergy,
      ),

    currentWaterlinePower,

    currentScheduledHpElectricalPower,

    currentScheduledHpThermalPower:
      currentScheduledHpElectricalPower *
      cop,
  };
}


/*
 * ==================================================================
 * rust_peakShavingCausal
 * ==================================================================
 */


export interface RustPeakShavingCausalScopeInput
  extends StorageKernelParameters {

  exportPower:
    Float64Array;

  demandEnergy:
    Float64Array;

  currentIndex:
    number;

  horizonSteps:
    number;

  currentStorageEnergy:
    number;

  minimumReserveEnergy:
    number;

  maximumHeatPumpElectricalPower:
    number;

  cop:
    number;

  energyCutLevel:
    number;
}


export interface RustPeakShavingCausalScopeResult {
  cutLevelPower: number;

  normalPlanSurvives: boolean;
  maximumPvPlanSurvives: boolean;

  survivalChecks: number;
  binarySearchIterations: number;
}


export function runRustPeakShavingCausalScope(
  input:
    RustPeakShavingCausalScopeInput,
): RustPeakShavingCausalScopeResult {

  const {
    exportPower,
    demandEnergy,

    currentIndex,
    horizonSteps,

    currentStorageEnergy,

    minimumReserveEnergy,

    targetStorageEnergy,

    maximumHeatPumpElectricalPower,

    cop,

    energyCutLevel,

    timeStepHours,
  } = input;


  if (
    exportPower.length === 0 ||
    exportPower.length !==
      demandEnergy.length
  ) {
    throw new Error(
      "Invalid annual arrays for rust_peakShavingCausal.",
    );
  }


  const reserveFloorEnergy =
    Math.min(
      targetStorageEnergy,
      minimumReserveEnergy,
    );


  const clampedCurrentStorageEnergy =
    clamp(
      currentStorageEnergy,
      0,
      targetStorageEnergy,
    );


  let survivalChecks =
    0;

  let binarySearchIterations =
    0;


  function hpPowerAtCutLevel(
    exportPowerValue:
      number,

    cutLevelPower:
      number,
  ): number {

    return Math.min(
      maximumHeatPumpElectricalPower,

      Math.max(
        0,

        exportPowerValue -
        cutLevelPower,
      ),
    );
  }


  function survivesAtCutLevel(
    cutLevelPower:
      number,
  ): boolean {

    survivalChecks +=
      1;


    let simulatedEnergy =
      clampedCurrentStorageEnergy;


    for (
      let offset = 0;
      offset < horizonSteps;
      offset++
    ) {

      const dataIndex =
        wrapIndex(
          currentIndex +
          offset,

          exportPower.length,
        );


      const hpPower =
        hpPowerAtCutLevel(
          exportPower[
            dataIndex
          ] ?? 0,

          cutLevelPower,
        );


      const hpThermalEnergy =
        hpPower *
        cop *
        timeStepHours;


      /*
       * Current demand/loss has already happened before
       * currentStorageEnergy reaches this controller.
       */
      if (
        offset === 0
      ) {

        simulatedEnergy =
          Math.min(
            targetStorageEnergy,

            simulatedEnergy +
            hpThermalEnergy,
          );

        continue;
      }


      const requestedHeatLoss =
        calculateRequestedHeatLoss(
          simulatedEnergy,
          input,
        );


      const heatLoss =
        Math.min(
          simulatedEnergy,

          Math.max(
            0,
            requestedHeatLoss,
          ),
        );


      const afterLoss =
        Math.max(
          0,

          simulatedEnergy -
          heatLoss,
        );


      const demand =
        Math.max(
          0,

          demandEnergy[
            dataIndex
          ] ?? 0,
        );


      const dischargedEnergy =
        Math.min(
          demand,
          afterLoss,
        );


      const energyBeforeForecastCharge =
        afterLoss -
        dischargedEnergy;


      const unmetDemand =
        Math.max(
          0,

          demand -
          dischargedEnergy,
        );


      if (
        unmetDemand >
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
          targetStorageEnergy,

          energyBeforeForecastCharge +
          hpThermalEnergy,
        );
    }


    return true;
  }


  let cutLevelPower =
    energyCutLevel;


  let normalPlanSurvives =
    false;

  let maximumPvPlanSurvives =
    false;


  /*
   * --------------------------------------------------------------
   * PARSER / JIT SCOPE ID
   * --------------------------------------------------------------
   */
  rust_peakShavingCausal: {

    normalPlanSurvives =
      survivesAtCutLevel(
        energyCutLevel,
      );


    if (
      normalPlanSurvives
    ) {

      maximumPvPlanSurvives =
        true;

      break rust_peakShavingCausal;
    }


    maximumPvPlanSurvives =
      survivesAtCutLevel(
        0,
      );


    if (
      !maximumPvPlanSurvives
    ) {

      cutLevelPower =
        0;

      break rust_peakShavingCausal;
    }


    let lowerCut =
      0;

    let upperCut =
      energyCutLevel;


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


      const middleSurvives =
        survivesAtCutLevel(
          middleCut,
        );


      if (
        middleSurvives
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


  return {
    cutLevelPower,

    normalPlanSurvives,

    maximumPvPlanSurvives,

    survivalChecks,

    binarySearchIterations,
  };
}
