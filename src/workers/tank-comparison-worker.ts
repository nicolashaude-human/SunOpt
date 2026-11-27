// src/workers/tank-comparison-worker.ts

import { createTankComparisonStorage } from "@/simulation/tank-comparison-thermal-reserve";

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

import { getAllNativeScopeStatus } from "@/simulation/rust-acceleration-runtime";

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


  const storage = createTankComparisonStorage(storageVolume);

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


    const exportAfterHpPower =
      Math.max(
        0,
        exportBeforeHpPower -
          heatPumpElectricalPower,
      );


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
