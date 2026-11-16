// src/app/api/model-comparison/route.ts

import {
  NextRequest,
  NextResponse,
} from "next/server";

import {
  calculateAnnualHotWaterDemand,
  DEFAULT_HOT_WATER_PROFILE,

  type HeatPump,
  type HotWaterControl,
  type HotWaterDemand,
  type PVSystem,
  type ThermalStorage,
  type WeatherData,
} from "@/domain";

import {
  resampleToQuarterHour,
} from "@/simulation/resample-to-quarter-hour";

import {
  simulateHotWaterSystem,
} from "@/simulation/hot-water-simulation";

import {
  calculateRollingHotWaterDemandForecast,
} from "@/simulation/hot-water-demand-forecast";

import {
  createQuantumGravityControlModel,
} from "@/simulation/control-model-quantum-gravity";

import {
  createPlasmaWaveControlModel,
} from "@/simulation/control-model-plasma-wave";


interface WeatherApiResponse {
  weatherData: WeatherData;
}


interface RequestBody {

  postalCode:
    string;

  year:
    number;

  pvSystem:
    PVSystem;

  hotWaterDemand:
    HotWaterDemand;

  heatPump:
    HeatPump;

  storage:
    ThermalStorage;

  control:
    HotWaterControl;
}


const TIME_STEP_HOURS =
  0.25;


/*
 * ================================================================
 * API
 * ================================================================
 */
export async function POST(
  request: NextRequest,
): Promise<NextResponse> {

  try {

    const body =
      (await request.json()) as
        RequestBody;


    /*
     * ============================================================
     * WEATHER
     * ============================================================
     */

    const weatherResponse =
      await fetch(
        `${request.nextUrl.origin}/api/weather`,
        {
          method:
            "POST",

          headers: {
            "Content-Type":
              "application/json",
          },

          body:
            JSON.stringify({

              postalCode:
                body.postalCode,

              pvSystem:
                body.pvSystem,

              startYear:
                body.year,

              endYear:
                body.year,
            }),

          cache:
            "no-store",
        },
      );


    if (
      !weatherResponse.ok
    ) {

      const errorText =
        await weatherResponse.text();


      throw new Error(
        `Weather API failed: ${errorText}`,
      );
    }


    const weatherResult =
      (await weatherResponse.json()) as
        WeatherApiResponse;


    /*
     * ============================================================
     * HOT WATER
     * ============================================================
     */

    const hourlyHotWaterData =
      calculateAnnualHotWaterDemand(

        body.hotWaterDemand,

        DEFAULT_HOT_WATER_PROFILE,

        weatherResult.weatherData,
      );


    /*
     * Convert weather and DHW data to the common
     * 15-minute simulation resolution.
     */
    const quarterHour =
      resampleToQuarterHour(

        weatherResult.weatherData,

        hourlyHotWaterData,
      );


    /*
     * ============================================================
     * PHYSICAL FORECAST HORIZON
     * ============================================================
     *
     * This is currently still used by Quantum Gravity.
     *
     * Plasma Wave deliberately uses its own 48 h user-demand
     * horizon.
     */

    const rollingDemandForecast =
      calculateRollingHotWaterDemandForecast({

        hotWaterData:
          quarterHour.hotWaterData,

        storage:
          body.storage,

        timeStepHours:
          TIME_STEP_HOURS,

        targetTemperature:
          75,

        minimumTemperature:
          60,

        safetyFactor:
          1.25,

        maximumForecastHours:
          168,
      });


    /*
     * ============================================================
     * 1. EXISTING CONTROLLER
     * ============================================================
     *
     * No alternative controlModel supplied.
     *
     * Therefore simulateHotWaterSystem()
     * automatically uses the current SunOpt controller.
     */

    const existingSimulation =
      simulateHotWaterSystem({

        weatherData:
          quarterHour.weatherData,

        hotWaterData:
          quarterHour.hotWaterData,

        heatPump:
          body.heatPump,

        storage:
          body.storage,

        control:
          body.control,

        timeStepHours:
          TIME_STEP_HOURS,
      });


    /*
     * ============================================================
     * 2. QUANTUM GRAVITY CONTROLLER
     * ============================================================
     */

    const quantumGravityControlModel =
      createQuantumGravityControlModel({

        weatherData:
          quarterHour.weatherData,

        hotWaterData:
          quarterHour.hotWaterData,

        heatPump:
          body.heatPump,

        storage:
          body.storage,

        control:
          body.control,

        timeStepHours:
          TIME_STEP_HOURS,

        planningHorizonHours:
          rollingDemandForecast
            .planningHorizonHours,
      });


    const quantumGravitySimulation =
      simulateHotWaterSystem({

        weatherData:
          quarterHour.weatherData,

        hotWaterData:
          quarterHour.hotWaterData,

        heatPump:
          body.heatPump,

        storage:
          body.storage,

        control:
          body.control,

        timeStepHours:
          TIME_STEP_HOURS,

        controlModel:
          quantumGravityControlModel,
      });


    /*
     * ============================================================
     * 3. PLASMA WAVE ACCELERATOR
     * ============================================================
     *
     * Plasma Wave:
     *
     * 1. looks 48 h forward,
     * 2. includes ALL future DHW packages,
     * 3. corrects them for storage losses,
     * 4. compares the required inventory with the ACTUAL
     *    current storage state,
     * 5. respects the 75 °C tank limit,
     * 6. places the required HP energy into the highest
     *    PV-export peaks using the water-fill algorithm.
     */

    const plasmaWaveControlModel =
    createPlasmaWaveControlModel({
      weatherData:
        quarterHour.weatherData,
  
      hotWaterData:
        quarterHour.hotWaterData,
  
      heatPump:
        body.heatPump,
  
      storage:
        body.storage,
  
      control:
        body.control,
  
      timeStepHours:
        TIME_STEP_HOURS,
  
      planningHorizonHours:
        48,
  
      targetTemperature:
        75,
    });


    const plasmaWaveSimulation =
      simulateHotWaterSystem({

        weatherData:
          quarterHour.weatherData,

        hotWaterData:
          quarterHour.hotWaterData,

        heatPump:
          body.heatPump,

        storage:
          body.storage,

        control:
          body.control,

        timeStepHours:
          TIME_STEP_HOURS,

        controlModel:
          plasmaWaveControlModel,
      });


    /*
     * ============================================================
     * TIMELINE
     * ============================================================
     *
     * Same timestamps for all models.
     *
     * This gives the frontend real simulation traces instead of
     * diagnostic forecast curves.
     */

    const timeline =
      existingSimulation.steps.map(
        (
          existingStep,
          index,
        ) => {

          const quantumGravityStep =
            quantumGravitySimulation
              .steps[index];


          const plasmaWaveStep =
            plasmaWaveSimulation
              .steps[index];


          if (
            !quantumGravityStep ||
            !plasmaWaveStep
          ) {

            throw new Error(
              `Model comparison step missing at index ${index}.`,
            );
          }


          return {

            timestamp:
              existingStep.timestamp,


            /*
             * ----------------------------------
             * PV
             * ----------------------------------
             *
             * Same physical PV input for all models.
             *
             * kW_el
             */
            pvPower:
              existingStep
                .pvProduction /
              TIME_STEP_HOURS,


            /*
             * ----------------------------------
             * STORAGE ENERGY
             * ----------------------------------
             *
             * kWh_th
             */

            existingStorageEnergy:
              existingStep
                .storageEnergy,

            quantumGravityStorageEnergy:
              quantumGravityStep
                .storageEnergy,

            plasmaWaveStorageEnergy:
              plasmaWaveStep
                .storageEnergy,


            /*
             * ----------------------------------
             * HEAT-PUMP ELECTRICAL POWER
             * ----------------------------------
             *
             * kW_el
             */

            existingHeatPumpPower:
              existingStep
                .heatPumpElectricalEnergy /
              TIME_STEP_HOURS,

            quantumGravityHeatPumpPower:
              quantumGravityStep
                .heatPumpElectricalEnergy /
              TIME_STEP_HOURS,

            plasmaWaveHeatPumpPower:
              plasmaWaveStep
                .heatPumpElectricalEnergy /
              TIME_STEP_HOURS,


            /*
             * ----------------------------------
             * GRID IMPORT POWER
             * ----------------------------------
             *
             * kW_el
             */

            existingGridImportPower:
              existingStep
                .gridImport /
              TIME_STEP_HOURS,

            quantumGravityGridImportPower:
              quantumGravityStep
                .gridImport /
              TIME_STEP_HOURS,

            plasmaWaveGridImportPower:
              plasmaWaveStep
                .gridImport /
              TIME_STEP_HOURS,


            /*
             * ----------------------------------
             * PV USED BY HP
             * ----------------------------------
             *
             * kW_el
             */

            existingPvUsedByHeatPumpPower:
              existingStep
                .pvUsedByHeatPump /
              TIME_STEP_HOURS,

            quantumGravityPvUsedByHeatPumpPower:
              quantumGravityStep
                .pvUsedByHeatPump /
              TIME_STEP_HOURS,

            plasmaWavePvUsedByHeatPumpPower:
              plasmaWaveStep
                .pvUsedByHeatPump /
              TIME_STEP_HOURS,


            /*
             * ----------------------------------
             * HOT-WATER DEMAND
             * ----------------------------------
             *
             * Same physical load for all models.
             *
             * kW_th
             */

            hotWaterDemandPower:
              existingStep
                .thermalDemand /
              TIME_STEP_HOURS,
          };
        },
      );


    /*
     * ============================================================
     * REAL ANNUAL RESULTS
     * ============================================================
     *
     * IMPORTANT:
     *
     * These are NOT diagnostic forecast values.
     *
     * Every result comes from a complete annual run through the
     * same physical simulator.
     */

    return NextResponse.json({

      year:
        body.year,

      storageVolume:
        body.storage.volume,


      /*
       * Quantum Gravity planning horizon.
       */
      planningHorizonHours:
        rollingDemandForecast
          .planningHorizonHours,


      /*
       * Plasma Wave planning horizon.
       */
      plasmaWavePlanningHorizonHours:
        48,


      /*
       * ==========================================================
       * ANNUAL MODEL RESULTS
       * ==========================================================
       */

      models: [

        /*
         * --------------------------------------------------------
         * EXISTING
         * --------------------------------------------------------
         */
        {

          modelId:
            "existing",

          modelName:
            "Existing Controller",

          status:
            "simulated",

          gridImport:
            existingSimulation
              .balance
              .gridImport,

          pvUsedByHeatPump:
            existingSimulation
              .balance
              .pvUsedByHeatPump,

          heatPumpConsumption:
            existingSimulation
              .balance
              .heatPumpConsumption,

          storageLosses:
            existingSimulation
              .balance
              .storageLosses,

          unmetThermalDemand:
            existingSimulation
              .balance
              .unmetThermalDemand,
        },


        /*
         * --------------------------------------------------------
         * QUANTUM GRAVITY
         * --------------------------------------------------------
         */
        {

          modelId:
            "quantum-gravity",

          modelName:
            "Quantum Gravity",

          status:
            "simulated",

          gridImport:
            quantumGravitySimulation
              .balance
              .gridImport,

          pvUsedByHeatPump:
            quantumGravitySimulation
              .balance
              .pvUsedByHeatPump,

          heatPumpConsumption:
            quantumGravitySimulation
              .balance
              .heatPumpConsumption,

          storageLosses:
            quantumGravitySimulation
              .balance
              .storageLosses,

          unmetThermalDemand:
            quantumGravitySimulation
              .balance
              .unmetThermalDemand,
        },


        /*
         * --------------------------------------------------------
         * PLASMA WAVE
         * --------------------------------------------------------
         */
        {

          modelId:
            "plasma-wave",

          modelName:
            "Plasma Wave Accelerator",

          status:
            "simulated",

          gridImport:
            plasmaWaveSimulation
              .balance
              .gridImport,

          pvUsedByHeatPump:
            plasmaWaveSimulation
              .balance
              .pvUsedByHeatPump,

          heatPumpConsumption:
            plasmaWaveSimulation
              .balance
              .heatPumpConsumption,

          storageLosses:
            plasmaWaveSimulation
              .balance
              .storageLosses,

          unmetThermalDemand:
            plasmaWaveSimulation
              .balance
              .unmetThermalDemand,
        },


        /*
         * --------------------------------------------------------
         * ZEROPOINT
         * --------------------------------------------------------
         *
         * Still diagnostic only.
         */
        {

          modelId:
            "zeropoint",

          modelName:
            "Zeropoint Generator",

          status:
            "diagnostic",

          gridImport:
            null,

          pvUsedByHeatPump:
            null,

          heatPumpConsumption:
            null,

          storageLosses:
            null,

          unmetThermalDemand:
            null,
        },
      ],


      /*
       * ==========================================================
       * TIME-RESOLVED MODEL COMPARISON
       * ==========================================================
       */

      timeline,
    });


  } catch (error) {

    console.error(
      "Model comparison error:",
      error,
    );


    return NextResponse.json(
      {

        error:
          error instanceof Error
            ? error.message
            : "Model comparison failed.",
      },
      {
        status:
          400,
      },
    );
  }
}