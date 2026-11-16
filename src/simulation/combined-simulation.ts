// src/simulation/combined-simulation.ts

import type {
  CombinedHeatDemand,
  CombinedThermalStorage,
  HeatPump,
  ThermalStorageState,
  WeatherData,
} from "@/domain";

import {
  calculateHeatPumpStep,
  calculateStorageCapacity,
  createInitialStorageState,
  simulateStorageStep,
} from "@/domain";

import type {
  EnergyBalance,
} from "./simulation-types";


export interface CombinedSimulationInput {
  weatherData: WeatherData;
  heatDemand: CombinedHeatDemand;

  heatPump: HeatPump;
  storage: CombinedThermalStorage;

  timeStepHours?: number;
}


export interface CombinedSimulationStep {
  timestamp: string;

  pvProduction: number;

  hotWaterDemand: number;
  spaceHeatingDemand: number;

  heatPumpThermalEnergy: number;
  heatPumpElectricalEnergy: number;

  pvUsedByHeatPump: number;
  gridImport: number;
  pvSurplus: number;

  hotWaterStorageEnergy: number;
  heatingStorageEnergy: number;

  unmetHotWaterDemand: number;
  unmetSpaceHeatingDemand: number;

  storageLoss: number;
}


export interface CombinedSimulationResult {
  balance: EnergyBalance;

  finalHotWaterStorageState:
    ThermalStorageState;

  finalHeatingStorageState:
    ThermalStorageState;

  steps: CombinedSimulationStep[];
}


export function simulateCombinedSystem(
  input: CombinedSimulationInput,
): CombinedSimulationResult {

  const {
    weatherData,
    heatDemand,
    heatPump,
    storage,
    timeStepHours = 1,
  } = input;


  if (
    weatherData.length !==
    heatDemand.length
  ) {
    throw new Error(
      "Weather data and heat demand must contain the same number of time steps.",
    );
  }


  const hotWaterCapacity =
    calculateStorageCapacity(
      storage.hotWaterZone,
    );

  const heatingCapacity =
    calculateStorageCapacity(
      storage.heatingZone,
    );


  let hotWaterState =
    createInitialStorageState(
      storage.hotWaterZone,
    );

  let heatingState =
    createInitialStorageState(
      storage.heatingZone,
    );


  const balance: EnergyBalance = {
    pvProduction: 0,
    pvUsedByHeatPump: 0,
    gridImport: 0,
    pvSurplus: 0,

    heatPumpConsumption: 0,
    thermalProduction: 0,

    thermalDemand: 0,
    unmetThermalDemand: 0,

    storageLosses: 0,
  };


  const steps: CombinedSimulationStep[] = [];


  for (
    let i = 0;
    i < weatherData.length;
    i++
  ) {

    const weather =
      weatherData[i];

    const demand =
      heatDemand[i];

    const pvEnergy =
      weather.pvPower *
      timeStepHours;


    // =================================
    // 1. Wärmebedarf aus Speichern
    // =================================

    const hotWaterDischarge =
      simulateStorageStep(
        storage.hotWaterZone,
        hotWaterState,
        0,
        demand.hotWaterDemand,
        timeStepHours,
      );

    hotWaterState =
      hotWaterDischarge.state;


    const heatingDischarge =
      simulateStorageStep(
        storage.heatingZone,
        heatingState,
        0,
        demand.spaceHeatingDemand,
        timeStepHours,
      );

    heatingState =
      heatingDischarge.state;


    // =================================
    // 2. Fehlende Speicherenergie
    // =================================

    const missingHotWaterEnergy =
      Math.max(
        0,
        hotWaterCapacity -
          hotWaterState.storedEnergy,
      );

    const missingHeatingEnergy =
      Math.max(
        0,
        heatingCapacity -
          heatingState.storedEnergy,
      );


    // =================================
    // 3. Warmwasser zuerst
    // =================================

    const hotWaterHeatPump =
      calculateHeatPumpStep(
        heatPump,
        missingHotWaterEnergy,
        timeStepHours,
      );


    const hotWaterCharge =
      simulateStorageStep(
        {
          ...storage.hotWaterZone,
          dailyHeatLoss: 0,
        },
        hotWaterState,
        hotWaterHeatPump
          .deliveredThermalEnergy,
        0,
        timeStepHours,
      );

    hotWaterState =
      hotWaterCharge.state;


    // =================================
    // 4. Verbleibende WP-Leistung
    // =================================

    const maximumThermalEnergy =
      heatPump.thermalPower *
      timeStepHours;

    const remainingThermalEnergy =
      Math.max(
        0,
        maximumThermalEnergy -
          hotWaterHeatPump
            .deliveredThermalEnergy,
      );


    const heatingRequestedEnergy =
      Math.min(
        missingHeatingEnergy,
        remainingThermalEnergy,
      );


    const heatingHeatPump =
      calculateHeatPumpStep(
        heatPump,
        heatingRequestedEnergy,
        timeStepHours,
      );


    const heatingCharge =
      simulateStorageStep(
        {
          ...storage.heatingZone,
          dailyHeatLoss: 0,
        },
        heatingState,
        heatingHeatPump
          .deliveredThermalEnergy,
        0,
        timeStepHours,
      );

    heatingState =
      heatingCharge.state;


    // =================================
    // 5. Gesamtenergie Wärmepumpe
    // =================================

    const hpThermalEnergy =
      hotWaterHeatPump
        .deliveredThermalEnergy +
      heatingHeatPump
        .deliveredThermalEnergy;

    const hpElectricalEnergy =
      hotWaterHeatPump
        .electricalEnergy +
      heatingHeatPump
        .electricalEnergy;


    // =================================
    // 6. Elektrische Bilanz
    // =================================

    const pvUsedByHeatPump =
      Math.min(
        pvEnergy,
        hpElectricalEnergy,
      );

    const gridImport =
      Math.max(
        0,
        hpElectricalEnergy -
          pvEnergy,
      );

    const pvSurplus =
      Math.max(
        0,
        pvEnergy -
          hpElectricalEnergy,
      );


    // =================================
    // 7. Nicht gedeckter Wärmebedarf
    // =================================

    const unmetHotWaterDemand =
      hotWaterDischarge.unmetDemand;

    const unmetSpaceHeatingDemand =
      heatingDischarge.unmetDemand;


    // =================================
    // 8. Bilanz
    // =================================

    balance.pvProduction +=
      pvEnergy;

    balance.pvUsedByHeatPump +=
      pvUsedByHeatPump;

    balance.gridImport +=
      gridImport;

    balance.pvSurplus +=
      pvSurplus;

    balance.heatPumpConsumption +=
      hpElectricalEnergy;

    balance.thermalProduction +=
      hpThermalEnergy;

    balance.thermalDemand +=
      demand.totalThermalDemand;

    balance.unmetThermalDemand +=
      unmetHotWaterDemand +
      unmetSpaceHeatingDemand;

    balance.storageLosses +=
      hotWaterDischarge.heatLoss +
      heatingDischarge.heatLoss;


    // =================================
    // 9. Stundenwert
    // =================================

    steps.push({
      timestamp:
        weather.timestamp,

      pvProduction:
        pvEnergy,

      hotWaterDemand:
        demand.hotWaterDemand,

      spaceHeatingDemand:
        demand.spaceHeatingDemand,

      heatPumpThermalEnergy:
        hpThermalEnergy,

      heatPumpElectricalEnergy:
        hpElectricalEnergy,

      pvUsedByHeatPump,

      gridImport,

      pvSurplus,

      hotWaterStorageEnergy:
        hotWaterState.storedEnergy,

      heatingStorageEnergy:
        heatingState.storedEnergy,

      unmetHotWaterDemand,

      unmetSpaceHeatingDemand,

      storageLoss:
        hotWaterDischarge.heatLoss +
        heatingDischarge.heatLoss,
    });
  }


  return {
    balance,

    finalHotWaterStorageState:
      hotWaterState,

    finalHeatingStorageState:
      heatingState,

    steps,
  };
}