// src/optimization/hot-water-storage-optimizer.ts

import type {
    HeatPump,
    HotWaterControl,
    HotWaterData,
    ThermalStorage,
    WeatherData,
  } from "@/domain";

import {
  simulateHotWaterSystem,
} from "@/simulation/hot-water-simulation";

import type {
  OptimizationTarget,
  StorageOptimizationRange,
  StorageOptimizationResult,
  StorageOptimizationVariant,
} from "./optimization-types";


export interface HotWaterStorageOptimizationInput {
    weatherData: WeatherData;
  
    hotWaterData: HotWaterData;
  
    heatPump: HeatPump;
  
    storageTemplate:
      Omit<ThermalStorage, "volume">;
  
    control:
      HotWaterControl;
  
    range:
      StorageOptimizationRange;
  
    target:
      OptimizationTarget;
  
    timeStepHours?: number;
  }


export function optimizeHotWaterStorage(
  input: HotWaterStorageOptimizationInput,
): StorageOptimizationResult {

  validateRange(input.range);

  const variants:
    StorageOptimizationVariant[] = [];


  for (
    let volume = input.range.minVolume;
    volume <= input.range.maxVolume;
    volume += input.range.step
  ) {

    const storage: ThermalStorage = {
      ...input.storageTemplate,
      volume,
    };


    const simulation =
    simulateHotWaterSystem({
      weatherData:
        input.weatherData,
  
      hotWaterData:
        input.hotWaterData,
  
      heatPump:
        input.heatPump,
  
      storage,
  
      control:
        input.control,
  
      timeStepHours:
        input.timeStepHours ?? 1,
    });


    const balance =
      simulation.balance;


    const pvSelfConsumptionRate =
      balance.pvProduction > 0
        ? balance.pvUsedByHeatPump /
          balance.pvProduction
        : 0;


    variants.push({
      storageVolume:
        volume,

      gridImport:
        balance.gridImport,

      pvUsedByHeatPump:
        balance.pvUsedByHeatPump,

      pvSurplus:
        balance.pvSurplus,

      heatPumpConsumption:
        balance.heatPumpConsumption,

      storageLosses:
        balance.storageLosses,

      unmetThermalDemand:
        balance.unmetThermalDemand,

      pvSelfConsumptionRate,
    });
  }


  if (variants.length === 0) {
    throw new Error(
      "Storage optimization produced no variants.",
    );
  }


  const optimalVariant =
    selectOptimalVariant(
      variants,
      input.target,
    );


  return {
    target:
      input.target,

    optimalStorageVolume:
      optimalVariant.storageVolume,

    optimalVariant,

    variants,
  };
}


function selectOptimalVariant(
  variants: StorageOptimizationVariant[],
  target: OptimizationTarget,
): StorageOptimizationVariant {

  return variants.reduce(
    (best, current) => {

      if (target === "grid-import") {
        return current.gridImport <
          best.gridImport
          ? current
          : best;
      }


      if (
        target ===
        "pv-self-consumption"
      ) {
        return current.pvSelfConsumptionRate >
          best.pvSelfConsumptionRate
          ? current
          : best;
      }


      return best;
    },
  );
}


function validateRange(
  range: StorageOptimizationRange,
): void {

  if (
    range.minVolume <= 0 ||
    range.maxVolume <= 0 ||
    range.step <= 0
  ) {
    throw new Error(
      "Storage optimization volumes and step must be greater than zero.",
    );
  }

  if (
    range.minVolume >
    range.maxVolume
  ) {
    throw new Error(
      "Minimum storage volume cannot exceed maximum storage volume.",
    );
  }
}