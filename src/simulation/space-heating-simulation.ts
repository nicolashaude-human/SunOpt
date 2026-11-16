// src/simulation/space-heating-simulation.ts

import type {
  HeatPump,
  SpaceHeatingData,
  ThermalStorage,
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


export interface SpaceHeatingSimulationInput {
  weatherData: WeatherData;
  heatingData: SpaceHeatingData;

  heatPump: HeatPump;
  storage: ThermalStorage;

  timeStepHours?: number;
}


export interface SpaceHeatingSimulationStep {
  timestamp: string;

  outdoorTemperature: number;

  pvProduction: number; // kWh_el

  thermalDemand: number; // kWh_th

  heatPumpThermalEnergy: number;    // kWh_th
  heatPumpElectricalEnergy: number; // kWh_el

  pvUsedByHeatPump: number; // kWh_el
  gridImport: number;       // kWh_el
  pvSurplus: number;        // kWh_el

  storageEnergy: number;   // kWh_th
  storageCapacity: number; // kWh_th

  storageLoss: number; // kWh_th
  unmetDemand: number; // kWh_th
}


export interface SpaceHeatingSimulationResult {
  balance: EnergyBalance;

  finalStorageState:
    ThermalStorageState;

  steps:
    SpaceHeatingSimulationStep[];
}


export function simulateSpaceHeatingSystem(
  input: SpaceHeatingSimulationInput,
): SpaceHeatingSimulationResult {

  const {
    weatherData,
    heatingData,
    heatPump,
    storage,
    timeStepHours = 1,
  } = input;


  if (
    weatherData.length !==
    heatingData.length
  ) {
    throw new Error(
      "Weather data and heating data must contain the same number of time steps.",
    );
  }


  let storageState =
    createInitialStorageState(storage);

  const storageCapacity =
    calculateStorageCapacity(storage);


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


  const steps:
    SpaceHeatingSimulationStep[] = [];


  for (
    let i = 0;
    i < weatherData.length;
    i++
  ) {

    const weather =
      weatherData[i];

    const heating =
      heatingData[i];


    // PV-Leistung → Energie des Zeitschritts
    const pvEnergy =
      weather.pvPower *
      timeStepHours;

    const demand =
      heating.thermalEnergy;


    // ---------------------------------
    // 1. Gebäude aus Speicher versorgen
    // ---------------------------------

    const dischargeResult =
      simulateStorageStep(
        storage,
        storageState,
        0,
        demand,
        timeStepHours,
      );

    storageState =
      dischargeResult.state;


    // ---------------------------------
    // 2. Fehlende Speicherenergie
    // ---------------------------------

    const missingStorageEnergy =
      Math.max(
        0,
        storageCapacity -
          storageState.storedEnergy,
      );


    // ---------------------------------
    // 3. Wärmepumpe
    // ---------------------------------

    const heatPumpResult =
      calculateHeatPumpStep(
        heatPump,
        missingStorageEnergy,
        timeStepHours,
      );


    // ---------------------------------
    // 4. Speicher laden
    // ---------------------------------

    const chargeResult =
      simulateStorageStep(
        {
          ...storage,

          // Verlust wurde beim
          // Entladen bereits berücksichtigt.
          dailyHeatLoss: 0,
        },
        storageState,
        heatPumpResult
          .deliveredThermalEnergy,
        0,
        timeStepHours,
      );

    storageState =
      chargeResult.state;


    // ---------------------------------
    // 5. Elektrische Bilanz
    // ---------------------------------

    const hpElectricalEnergy =
      heatPumpResult
        .electricalEnergy;

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


    // ---------------------------------
    // 6. Jahresbilanz
    // ---------------------------------

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
      heatPumpResult
        .deliveredThermalEnergy;

    balance.thermalDemand +=
      demand;

    balance.unmetThermalDemand +=
      dischargeResult.unmetDemand;

    balance.storageLosses +=
      dischargeResult.heatLoss;


    // ---------------------------------
    // 7. Stundenwert speichern
    // ---------------------------------

    steps.push({
      timestamp:
        weather.timestamp,

      outdoorTemperature:
        weather.outdoorTemperature,

      pvProduction:
        pvEnergy,

      thermalDemand:
        demand,

      heatPumpThermalEnergy:
        heatPumpResult
          .deliveredThermalEnergy,

      heatPumpElectricalEnergy:
        hpElectricalEnergy,

      pvUsedByHeatPump,

      gridImport,

      pvSurplus,

      storageEnergy:
        storageState.storedEnergy,

      storageCapacity,

      storageLoss:
        dischargeResult.heatLoss,

      unmetDemand:
        dischargeResult.unmetDemand,
    });
  }


  return {
    balance,

    finalStorageState:
      storageState,

    steps,
  };
}