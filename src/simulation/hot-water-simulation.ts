// src/simulation/hot-water-simulation.ts


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
  createInitialStorageState,
} from "@/domain";

import {
  calculateThermalStorageStep,
} from "@/domain/thermal-storage-calculation";

import type {
  EnergyBalance,
} from "./simulation-types";

import type {
  HotWaterControlModel,
} from "./hot-water-control-model";


export interface HotWaterSimulationInput {
  weatherData: WeatherData;

  hotWaterData: HotWaterData;

  heatPump: HeatPump;

  storage: ThermalStorage;

  control: HotWaterControl;

  timeStepHours?: number;


  /*
   * Optional alternative controller.
   *
   * If omitted, the original SunOpt forecast
   * controller is used exactly as before.
   */
  controlModel?: HotWaterControlModel;
}


export interface HotWaterSimulationStep {
  timestamp: string;

  pvProduction: number;

  thermalDemand: number;

  storageEnergy: number;

  forecastTargetEnergy: number;

  forecastRequiredEnergy: number;

  minimumReserveEnergy: number;

  heatPumpThermalEnergy: number;

  heatPumpElectricalEnergy: number;

  pvUsedByHeatPump: number;

  gridImport: number;

  pvSurplus: number;

  storageLoss: number;

  unmetThermalDemand: number;

  chargingMode:
    | "off"
    | "reserve"
    | "forecast";
}


export interface HotWaterSimulationResult {
  balance: EnergyBalance;

  finalStorageState:
    ThermalStorageState;

  steps:
    HotWaterSimulationStep[];
}


export function simulateHotWaterSystem(
  input: HotWaterSimulationInput,
): HotWaterSimulationResult {

  const {
    weatherData,
    hotWaterData,
    heatPump,
    storage,
    control,
  } = input;


  const timeStepHours =
    input.timeStepHours ?? 1;


  if (
    weatherData.length !==
    hotWaterData.length
  ) {
    throw new Error(
      "Weather data and hot-water data must have the same length.",
    );
  }


  const storageCapacity =
    calculateStorageCapacity(
      storage,
    );


  let storageState =
    createInitialStorageState(
      storage,
    );


  const balance:
    EnergyBalance = {

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
    HotWaterSimulationStep[] = [];


  for (
    let index = 0;
    index < weatherData.length;
    index++
  ) {

    const weather =
      weatherData[index];


    const demand =
      hotWaterData[index]
        .thermalEnergy;


    /*
     * --------------------------------------------
     * 1. CURRENT DEMAND + STORAGE LOSS
     * --------------------------------------------
     */

    const storageEnergyAtStepStart =
    storageState.storedEnergy;

  /*
   * Preview:
   *
   * What remains after THIS step's passive loss
   * and DHW demand if the HP does not charge?
   *
   * This is PURE calculation.
   * storageState itself is NOT modified yet.
   */
  const storagePreview =
    calculateThermalStorageStep({

      storage,

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


  /*
   * Demand that could not be supplied by the storage
   * and therefore has to be supplied directly by the HP.
   */
  const directThermalDemand =
    storagePreview.unmetDemand;

    /*
     * --------------------------------------------
     * 2. FORECAST TARGET
     * --------------------------------------------
     */

    const forecast =
  input.controlModel

    ? input.controlModel({

        currentIndex:
          index,

        weatherData,

        hotWaterData,

        heatPump,

        storage,

        control,

        timeStepHours,

        /*
         * IMPORTANT:
         *
         * This is the real storage state of THIS
         * simulation after current DHW consumption
         * and storage losses.
         */
        storageState,
      })

    : calculateHotWaterForecastTarget({

        currentIndex:
          index,

        weatherData,

        hotWaterData,

        heatPump,

        storage,

        control,

        timeStepHours,
      });

 
      /*
       * --------------------------------------------
       * 3. CHARGING DECISION
       * --------------------------------------------
       */
      
      /*
       * Storage energy required for the mandatory
       * short-term reserve.
       *
       * This may be charged from the grid if necessary.
       */
      const missingToReserve =
        Math.max(
          0,
          forecast.minimumReserveEnergy -
          storageStateAfterDemand.storedEnergy,
        );
      
      
      /*
       * Physical free space in the storage.
       */
      const availableStorageCapacity =
        Math.max(
          0,
          storageCapacity -
          storageStateAfterDemand.storedEnergy,
        );
      
      
      const mandatoryStorageCharge =
        Math.min(
          missingToReserve,
          availableStorageCapacity,
        );
      
      
        /*
        * Total PV production.
        */
       const pvEnergy =
         weather.pvPower *
         timeStepHours;
       
       
       /*
        * PV power available to the heat pump
        * after reserving household consumption.
        */
       const availablePvPower =
         Math.max(
           0,
           weather.pvPower -
             control.householdReservePower,
         );
      
         const availablePvEnergy =
  availablePvPower *
  timeStepHours;

      /*
       * Determine current COP.
       *
       * Currently this still uses our constant-COP
       * model. Later this becomes temperature dependent.
       */
      const referenceHeatPumpStep =
        calculateHeatPumpStep(
          heatPump,
          heatPump.thermalPower *
            timeStepHours,
          timeStepHours,
        );
      
      
      const cop =
        referenceHeatPumpStep.cop;
      
      
      /*
       * Maximum thermal energy that can be produced
       * from PV during this hour.
       */
      const maxThermalEnergyFromPv =
      Math.min(
        heatPump.thermalPower *
          timeStepHours,
    
        availablePvEnergy *
          cop,
      );
      
      
      /*
       * Mandatory thermal production consists of:
       *
       * 1. direct unmet DHW demand
       * 2. minimum storage reserve
       *
       * This production is allowed to use grid power.
       */
      const mandatoryThermalEnergy =
        directThermalDemand +
        mandatoryStorageCharge;
      
      
      /*
       * PV remaining after mandatory production.
       *
       * Forecast charging is ONLY allowed from this
       * remaining PV energy.
       */
      const remainingPvThermalEnergy =
        Math.max(
          0,
          maxThermalEnergyFromPv -
            mandatoryThermalEnergy,
        );
      
      
      /*
       * Storage level after the mandatory reserve
       * has been restored.
       */
      const storageEnergyAfterMandatoryCharge =
      storageStateAfterDemand.storedEnergy +
        mandatoryStorageCharge;
      
      
      /*
       * Additional storage energy requested by
       * the long-term forecast.
       */
      const missingToForecastTarget =
        Math.max(
          0,
          forecast.targetEnergy -
            storageEnergyAfterMandatoryCharge,
        );
      
      
      /*
       * Forecast charging is strictly PV-only.
       */
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
      
      
      /*
       * Total heat-pump request.
       */
      const requestedThermalEnergy =
        mandatoryThermalEnergy +
        forecastPvCharge;
    /*
     * --------------------------------------------
     * 4. HEAT PUMP
     * --------------------------------------------
     */

    const heatPumpStep =
      calculateHeatPumpStep(
        heatPump,
        requestedThermalEnergy,
        timeStepHours,
      );


    /*
     * First satisfy direct demand.
     */
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


    /*
     * Remaining heat goes into storage.
     */
    const thermalEnergyForStorage =
      Math.max(
        0,
        heatPumpStep
          .deliveredThermalEnergy -
          directThermalSupply,
      );


     /*
     * ================================================================
     * FINAL STORAGE STEP
     * ================================================================
     *
     * Start again from the TRUE state at the beginning of this
     * 15-minute interval.
     *
     * The central storage function performs exactly:
     *
     *   passive loss
     *   -> DHW output
     *   -> HP storage input
     *
     * exactly once.
     */
    const storageStep =
      calculateThermalStorageStep({

        storage,

        currentEnergy:
          storageEnergyAtStepStart,

        thermalInputEnergy:
          thermalEnergyForStorage,

        thermalOutputEnergy:
          demand,

        timeStepHours,

      });


    storageState =
      storageStep.state;
      
      /*
       * --------------------------------------------
       * 5. ELECTRICAL BALANCE
       * --------------------------------------------
       */
      
      const pvUsedByHeatPump =
      Math.min(
        availablePvEnergy,
        heatPumpStep
          .electricalEnergy,
      );
    
    
    const gridImport =
      Math.max(
        0,
        heatPumpStep
          .electricalEnergy -
          availablePvEnergy,
      );
    
    
    const pvSurplus =
      Math.max(
        0,
        availablePvEnergy -
          heatPumpStep
            .electricalEnergy,
      );


        /*
         * --------------------------------------------
         * 6. CHARGING MODE
         * --------------------------------------------
         */
        
        let chargingMode:
        HotWaterSimulationStep[
          "chargingMode"
        ] = "off";
      
      
      if (
        directThermalDemand > 0
      ) {
      
        /*
         * Emergency operation:
         * current demand could not be supplied
         * from storage.
         */
        chargingMode =
          "reserve";
      
      } else if (
        forecastPvCharge > 0
      ) {
      
        /*
         * Planned PV precharge.
         */
        chargingMode =
          "forecast";
      }


    /*
     * --------------------------------------------
     * 7. ANNUAL BALANCE
     * --------------------------------------------
     */

    balance.pvProduction +=
      pvEnergy;


    balance.pvUsedByHeatPump +=
      pvUsedByHeatPump;


    balance.gridImport +=
      gridImport;


    balance.pvSurplus +=
      pvSurplus;


    balance.heatPumpConsumption +=
      heatPumpStep
        .electricalEnergy;


    balance.thermalProduction +=
      heatPumpStep
        .deliveredThermalEnergy;


    balance.thermalDemand +=
      demand;


    balance.unmetThermalDemand +=
      finalUnmetThermalDemand;


      balance.storageLosses +=
      storageStep.heatLoss;


    /*
     * --------------------------------------------
     * 8. STEP RESULT
     * --------------------------------------------
     */

    steps.push({

      timestamp:
        weather.timestamp,

      pvProduction:
        pvEnergy,

      thermalDemand:
        demand,

      storageEnergy:
        storageState.storedEnergy,

      forecastTargetEnergy:
        forecast.targetEnergy,

      forecastRequiredEnergy:
        forecast.forecastRequiredEnergy,

      minimumReserveEnergy:
        forecast.minimumReserveEnergy,

      heatPumpThermalEnergy:
        heatPumpStep
          .deliveredThermalEnergy,

      heatPumpElectricalEnergy:
        heatPumpStep
          .electricalEnergy,

      pvUsedByHeatPump,

      gridImport,

      pvSurplus,

      storageLoss:
      storageStep.heatLoss,

      unmetThermalDemand:
        finalUnmetThermalDemand,

      chargingMode,
    });
  }


  return {
    balance,

    finalStorageState:
      storageState,

    steps,
  };
}