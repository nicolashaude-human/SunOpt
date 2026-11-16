// src/simulation/experimental_QuantumGravety.ts

import type { HotWaterData } from "@/domain/hot-water";
import type { ThermalStorage } from "@/domain/thermal-storage";

import {
  calculateStorageHeatLossAtTemperature,
} from "@/domain/thermal-storage-calculation";


export interface ExperimentalQuantumGravityPoint {
  /*
   * Energy which must be available in the
   * storage NOW to bridge the time until
   * sufficient PV energy becomes available.
   *
   * Unit: kWh_th
   */
  requiredStorageEnergy: number;

  /*
   * How far into the future we had to search
   * before sufficient PV was found.
   *
   * Unit: hours
   */
  bridgeHours: number;

  /*
   * True if sufficient PV was found inside
   * the planning horizon.
   */
  pvRecoveryFound: boolean;
}


export interface ExperimentalQuantumGravityResult {
  points: ExperimentalQuantumGravityPoint[];
}


export interface ExperimentalQuantumGravityInput {
  hotWaterData: HotWaterData;

  /*
   * Already calculated thermal PV forecast
   * per simulation step.
   *
   * IMPORTANT:
   * These values must be individual step
   * energies, NOT rolling integrals.
   *
   * Unit: kWh_th per step
   */
  pvThermalEnergyPerStep: number[];

  storage: ThermalStorage;

  timeStepHours: number;

  planningHorizonHours: number;

  safetyFactor: number;

  /*
   * Conservative temperature used for
   * calculating storage losses.
   */
  lossReferenceTemperature: number;
}


export function calculateExperimentalQuantumGravity(
  input: ExperimentalQuantumGravityInput,
): ExperimentalQuantumGravityResult {

  const {
    hotWaterData,
    pvThermalEnergyPerStep,
    storage,
    timeStepHours,
    planningHorizonHours,
    safetyFactor,
    lossReferenceTemperature,
  } = input;


  if (
    hotWaterData.length !==
    pvThermalEnergyPerStep.length
  ) {
    throw new Error(
      "Hot-water and PV arrays must have the same length.",
    );
  }


  if (timeStepHours <= 0) {
    throw new Error(
      "timeStepHours must be greater than zero.",
    );
  }


  const numberOfPoints =
    hotWaterData.length;


  const maximumSearchSteps =
    Math.min(
      Math.floor(
        planningHorizonHours /
          timeStepHours,
      ),
      numberOfPoints,
    );


  /*
   * For this first experimental version we
   * calculate losses at a constant reference
   * temperature.
   */
  const storageLossPerStep =
    calculateStorageHeatLossAtTemperature(
      storage,
      lossReferenceTemperature,
      timeStepHours,
    );


  const points:
    ExperimentalQuantumGravityPoint[] =
    new Array(numberOfPoints);


  /*
   * Start an independent forward search
   * at every point in the year.
   */
  for (
    let currentIndex = 0;
    currentIndex < numberOfPoints;
    currentIndex++
  ) {

    let accumulatedDemandEnergy = 0;

    let accumulatedPvThermalEnergy = 0;

    let bridgeSteps = maximumSearchSteps;

    let pvRecoveryFound = false;


    /*
     * Search forward chronologically.
     */
    for (
      let offset = 1;
      offset <= maximumSearchSteps;
      offset++
    ) {

      const futureIndex =
        wrapIndex(
          currentIndex + offset,
          numberOfPoints,
        );


      /*
       * Thermal demand which occurs before
       * or while we wait for useful PV.
       */
      accumulatedDemandEnergy +=
        hotWaterData[
          futureIndex
        ].thermalEnergy;


      accumulatedDemandEnergy +=
        storageLossPerStep;


      /*
       * Thermal energy which could be generated
       * from PV during this step.
       */
      accumulatedPvThermalEnergy +=
        pvThermalEnergyPerStep[
          futureIndex
        ];


      /*
       * Once future PV can cover everything
       * accumulated up to this point, we have
       * found the next recovery opportunity.
       */
      if (
        accumulatedPvThermalEnergy >=
        accumulatedDemandEnergy
      ) {
        bridgeSteps = offset;

        pvRecoveryFound = true;

        break;
      }
    }


    /*
     * The storage must bridge the demand until
     * the future PV recovery point.
     *
     * Apply the 25 % safety margin only after
     * finding that bridge.
     */
    const requiredStorageEnergy =
      accumulatedDemandEnergy *
      safetyFactor;


    points[currentIndex] = {
      requiredStorageEnergy,

      bridgeHours:
        bridgeSteps *
        timeStepHours,

      pvRecoveryFound,
    };
  }


  return {
    points,
  };
}


function wrapIndex(
  index: number,
  length: number,
): number {

  return (
    (
      index %
      length
    ) +
    length
  ) % length;
}