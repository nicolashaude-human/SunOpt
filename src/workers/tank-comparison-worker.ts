// src/workers/tank-comparison-worker.ts

import type {
  ThermalStorage,
} from "@/domain";

import {
  HOT_WATER_TEST_CONFIG,
} from "@/config/config";

import {
  createPlasmaWaveControlModel,
} from "@/simulation/control-model-plasma-wave";

import {
  simulateHotWaterSystem,
} from "@/simulation/hot-water-simulation";

import type {
  TankComparisonPoint,
} from "@/types/tank-comparison";

import {
  getSunOptNativeBinding,
  getAllNativeScopeStatus,
} from "@/simulation/rust-acceleration-runtime";

import type {
  ApproachANativeScopes,
} from "./tank-comparison-process-pool";


import type {
  TankComparisonWorkerInit,
  TankComparisonWorkerRequest,
  TankComparisonWorkerResponse,
} from "./tank-comparison-process-pool";


let init:
  TankComparisonWorkerInit |
  undefined;


function send(
  message:
    TankComparisonWorkerResponse,
): void {

  if (!process.send) {
    throw new Error(
      "Tank comparison worker has no IPC channel.",
    );
  }

  process.send(message);
}


function simulateTank(
  storageVolume: number,
): TankComparisonPoint {

  if (!init) {
    throw new Error(
      "Tank comparison worker is not initialized.",
    );
  }


  /*
   * ==============================================================
   * STORAGE
   * ==============================================================
   */

  const storage:
    ThermalStorage = {

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


  /*
   * ==============================================================
   * EXPORT AFTER HEAT PUMP
   * ==============================================================
   *
   * Store the controller decision for every quarter-hour.
   *
   * NaN lets us distinguish:
   *
   *   actual export = 0
   *
   * from:
   *
   *   no decision recorded
   */

  const exportAfterHpPowerByIndex =
    new Float64Array(
      init.weatherData.length,
    );

  exportAfterHpPowerByIndex.fill(
    Number.NaN,
  );


  /*
   * ==============================================================
   * SIMULATION
   * ==============================================================
   */

  const simulationStart =
    performance.now();


  const controlModel =
    createPlasmaWaveControlModel({
      weatherData:
        init.weatherData,

      hotWaterData:
        init.hotWaterData,

      heatPump:
        init.heatPump,

      storage,

      control:
        init.control,

      timeStepHours:
        init.timeStepHours,

      planningHorizonHours:
        HOT_WATER_TEST_CONFIG
          .userForecastHours,

      targetTemperature:
        75,

      onDecision:
        (decision) => {

          exportAfterHpPowerByIndex[
            decision.currentIndex
          ] =
            decision.exportAfterHpPower;
        },
    });


  const simulation =
    simulateHotWaterSystem({
      weatherData:
        init.weatherData,

      hotWaterData:
        init.hotWaterData,

      heatPump:
        init.heatPump,

      storage,

      control:
        init.control,

      timeStepHours:
        init.timeStepHours,

      controlModel,
    });


  const simulationDurationMs =
    performance.now() -
    simulationStart;


  /*
   * ==============================================================
   * ANNUAL ENERGY BALANCE
   * ==============================================================
   */

  let gridImport =
    0;

  let feedIn =
    0;

  let storageLosses =
    0;

  let unmetThermalDemand =
    0;


  for (
    let index = 0;
    index < simulation.steps.length;
    index += 1
  ) {

    const step =
      simulation.steps[index];

    const weatherPoint =
      init.weatherData[index];


    if (
      !step ||
      !weatherPoint
    ) {
      throw new Error(
        `Missing simulation data at index ${index}.`,
      );
    }


    gridImport +=
      step.gridImport;


    storageLosses +=
      step.storageLoss;


    unmetThermalDemand +=
      step.unmetThermalDemand;


    /*
     * Prefer the actual Plasma-Wave controller decision.
     *
     * The fallback reproduces the physical balance if no
     * decision was recorded for a particular index.
     */

    const recordedExport =
      exportAfterHpPowerByIndex[
        index
      ];


    const exportBeforeHpPower =
      Math.max(
        0,
        weatherPoint.pvPower -
          init.control
            .householdReservePower,
      );


    const heatPumpElectricalPower =
      step.heatPumpElectricalEnergy /
      init.timeStepHours;


    const fallbackExport =
      Math.max(
        0,
        exportBeforeHpPower -
          heatPumpElectricalPower,
      );


    const exportAfterHpPower =
      Number.isFinite(
        recordedExport,
      )
        ? recordedExport
        : fallbackExport;


    feedIn +=
      exportAfterHpPower *
      init.timeStepHours;
  }


  const selfConsumption =
    Math.max(
      0,
      init.pvProduction -
        feedIn,
    );


  /*
   * ==============================================================
   * ENERGY COST
   * ==============================================================
   */

  const gridCost =
    gridImport *
    init.gridPriceChfPerKWh;


  const feedInRevenue =
    feedIn *
    init.feedInPriceChfPerKWh;


  const netEnergyCost =
    gridCost -
    feedInRevenue;


  return {
    storageVolume,

    gridImport,

    feedIn,

    selfConsumption,

    storageLosses,

    unmetThermalDemand,

    gridCost,

    feedInRevenue,

    netEnergyCost,

    simulationDurationMs,
  };
}


process.on(
  "message",
  (
    message:
      TankComparisonWorkerRequest,
  ) => {

    try {

      if (
        message.type ===
        "init"
      ) {

        const nativeBinding = getSunOptNativeBinding();
        const required = [
          "registerAccelerationScope",
          "registerAccelerationDataPair",
          "runNoSunshineKernelContext",
          "runPeakShavingCausalKernelContext",
        ] as const;

        const missing = required.filter(
          (name) => typeof nativeBinding?.[name] !== "function",
        );
        if (missing.length > 0) {
          throw new Error(
            `Approach A requires the consolidated A1+A2 native addon; missing exports: ${missing.join(", ")}`,
          );
        }

        init =
          message.payload;


        send({
          type:
            "ready",
        });

        return;
      }


      if (
        message.type ===
        "run"
      ) {

        const point =
          simulateTank(
            message.storageVolume,
          );


        const nativeScopes: ApproachANativeScopes =
          getAllNativeScopeStatus().map((status) => ({
            scopeId: status.scopeId,
            calls: status.calls,
            nativeCalls: status.nativeCalls,
            fallbackCalls: status.fallbackCalls,
            failures: status.failures,
          }));

        // Do not claim A1+A2 success if the native scopes silently fell back.
        for (const scopeId of [
          "rust_noSunshine",
          "rust_peakShavingCausal",
        ]) {
          const status = nativeScopes.find((s) => s.scopeId === scopeId);
          if (!status || status.nativeCalls === 0 ||
              status.fallbackCalls !== 0 || status.failures !== 0) {
            throw new Error(
              `Approach A native execution incomplete for ${scopeId}: ${JSON.stringify(status ?? null)}`,
            );
          }
        }

        send({
          type:
            "result",

          point,
          nativeScopes,
        });
      }

    } catch (error) {

      send({
        type:
          "error",

        message:
          error instanceof Error
            ? error.message
            : String(error),

        stack:
          error instanceof Error
            ? error.stack
            : undefined,
      });
    }
  },
);