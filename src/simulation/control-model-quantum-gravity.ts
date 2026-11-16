// src/simulation/control-model-quantum-gravity.ts

import type {
  HeatPump,
  HotWaterControl,
  HotWaterData,
  ThermalStorage,
  WeatherData,
} from "@/domain";

import {
  calculateHotWaterForecastTarget,
  calculateStorageEnergyAtTemperature,
} from "@/domain";

import {
  calculateExperimentalQuantumGravity,
} from "./experimental_QuantumGravety";

import type {
  HotWaterControlModel,
} from "./hot-water-control-model";


export interface QuantumGravityControlModelInput {

  weatherData:
    WeatherData;

  hotWaterData:
    HotWaterData;

  heatPump:
    HeatPump;

  storage:
    ThermalStorage;

  control:
    HotWaterControl;

  timeStepHours:
    number;

  planningHorizonHours:
    number;
}


export function createQuantumGravityControlModel(
  input:
    QuantumGravityControlModelInput,
): HotWaterControlModel {

  const {
    weatherData,
    hotWaterData,
    heatPump,
    storage,
    control,
    timeStepHours,
    planningHorizonHours,
  } = input;


  /*
   * ==============================================================
   * COP
   * ==============================================================
   */

  const cop =
    heatPump.copModel.type ===
      "constant"

      ? heatPump
          .copModel
          .cop

      : 3;


  /*
   * ==============================================================
   * USABLE PV THERMAL ENERGY PER STEP
   * ==============================================================
   */

  const pvThermalEnergyPerStep =
    weatherData.map(
      (
        weatherPoint,
      ) => {

        const availablePvPower =
          Math.max(
            0,

            weatherPoint
              .pvPower -

              control
                .householdReservePower,
          );


        const availablePvElectricalEnergy =
          availablePvPower *
          timeStepHours;


        const thermalEnergyFromPv =
          availablePvElectricalEnergy *
          cop;


        const maximumHeatPumpThermalEnergy =
          heatPump
            .thermalPower *
          timeStepHours;


        return Math.min(
          thermalEnergyFromPv,
          maximumHeatPumpThermalEnergy,
        );
      },
    );


  /*
   * ==============================================================
   * QUANTUM GRAVITY PRECALCULATION
   * ==============================================================
   *
   * Expensive part:
   * calculate once for the complete annual dataset.
   */

  const quantumGravity =
    calculateExperimentalQuantumGravity({

      hotWaterData,

      pvThermalEnergyPerStep,

      storage,

      timeStepHours,

      planningHorizonHours,

      safetyFactor:
        control
          .forecastSafetyFactor,

      lossReferenceTemperature:
        control
          .pvTargetTemperature,
    });


  /*
   * ==============================================================
   * ABSOLUTE STORAGE BOUNDARIES
   * ==============================================================
   */

  const minimumStorageEnergy =
    calculateStorageEnergyAtTemperature(
      storage,
      storage.minTemperature,
    );


  const targetCapacity =
    calculateStorageEnergyAtTemperature(
      storage,
      control.pvTargetTemperature,
    );


  /*
   * ==============================================================
   * CONTROL MODEL
   * ==============================================================
   */

  const quantumGravityControlModel:
    HotWaterControlModel =
    (
      modelInput,
    ) => {

      const quantumPoint =
        quantumGravity
          .points[
            modelInput
              .currentIndex
          ];


      if (!quantumPoint) {

        throw new Error(
          `Quantum Gravity point missing at index ${modelInput.currentIndex}.`,
        );
      }


      /*
       * ----------------------------------------------------------
       * BASELINE FORECAST
       * ----------------------------------------------------------
       *
       * This gives us ALL fields required by
       * HotWaterForecastTarget.
       */

      const baselineForecast =
        calculateHotWaterForecastTarget({

          currentIndex:
            modelInput
              .currentIndex,

          weatherData,

          hotWaterData,

          heatPump,

          storage,

          control,

          timeStepHours,
        });


      /*
       * ----------------------------------------------------------
       * QUANTUM-GRAVITY TARGET
       * ----------------------------------------------------------
       */

      const forecastRequiredEnergy =
        Math.min(
          targetCapacity,

          minimumStorageEnergy +
            quantumPoint
              .requiredStorageEnergy,
        );


      const minimumReserveEnergy =
        minimumStorageEnergy;


      const targetEnergy =
        Math.min(
          targetCapacity,

          Math.max(
            minimumReserveEnergy,
            forecastRequiredEnergy,
          ),
        );


      /*
       * ----------------------------------------------------------
       * RETURN COMPLETE HotWaterForecastTarget
       * ----------------------------------------------------------
       *
       * Keep all common forecast metadata from
       * calculateHotWaterForecastTarget(), then replace
       * only the Quantum-Gravity control quantities.
       */

      return {

        ...baselineForecast,

        targetEnergy,

        minimumReserveEnergy,

        forecastRequiredEnergy,

        targetCapacity,
      };
    };


  return quantumGravityControlModel;
}