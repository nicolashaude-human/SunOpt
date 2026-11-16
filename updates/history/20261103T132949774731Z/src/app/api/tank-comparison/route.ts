// src/app/api/tank-comparison/route.ts

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
  type WeatherData,
} from "@/domain";

import {
  TANK_COMPARISON_CONFIG,
  HOT_WATER_TEST_CONFIG,
} from "@/config/config";

import {
  resampleToQuarterHour,
} from "@/simulation/resample-to-quarter-hour";

import type {
  TankComparisonResponse,
} from "@/types/tank-comparison";

import {
  runTankComparisonProcessPool,
} from "@/simulation/tank-comparison-process-pool";


export const runtime =
  "nodejs";

export const dynamic =
  "force-dynamic";


interface RequestBody {
  heatPumpThermalPower: number;
  pvPeakPower: number;

  gridPriceChfPerKWh: number;
  feedInPriceChfPerKWh: number;
}


interface WeatherApiResponse {
  weatherData: WeatherData;
}


const YEAR =
  2023;

const POSTAL_CODE =
  "8001";

const TIME_STEP_HOURS =
  HOT_WATER_TEST_CONFIG
    .timeStepHours;


/*
 * ================================================================
 * API
 * ================================================================
 */
export async function POST(
  request: NextRequest,
): Promise<NextResponse> {

  const totalStart =
    performance.now();


  try {

    const body =
      (await request.json()) as
        RequestBody;


    /*
     * ============================================================
     * COMMON SIMULATION INPUT
     * ============================================================
     *
     * These objects remain identical for every tank.
     */

    const pvSystem:
      PVSystem = {

      peakPower:
        body.pvPeakPower,

      azimuth:
        0,

      inclination:
        30,

      systemLoss:
        0.14,
    };


    const hotWaterDemand:
      HotWaterDemand = {

      persons:
        2,

      dailyConsumptionPerPerson:
        50,

      coldWaterTemperature:
        10,

      deliveryTemperature:
        60,
    };


    const heatPump:
      HeatPump = {

      thermalPower:
        body.heatPumpThermalPower,

      maxFlowTemperature:
        75,

      copModel: {
        type:
          "constant",

        cop:
          3,
      },
    };


    const control:
      HotWaterControl = {

      minimumReserveHours:
        2,

      forecastHorizonHours:
        72,

      pvTargetTemperature:
        75,

      householdReservePower:
        HOT_WATER_TEST_CONFIG
          .householdReservePower,

      forecastSafetyFactor:
        HOT_WATER_TEST_CONFIG
          .forecastSafetyFactor,
    };


    /*
     * ============================================================
     * PREPARE HISTORICAL DATA ONCE
     * ============================================================
     *
     * weatherData and hotWaterData are historical constants.
     *
     * They must NOT be rebuilt for every storage volume.
     */

    const preparationStart =
      performance.now();


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
                POSTAL_CODE,

              pvSystem,

              startYear:
                YEAR,

              endYear:
                YEAR,
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


    const hourlyHotWaterData =
      calculateAnnualHotWaterDemand(

        hotWaterDemand,

        DEFAULT_HOT_WATER_PROFILE,

        weatherResult.weatherData,
      );


    const quarterHour =
      resampleToQuarterHour(

        weatherResult.weatherData,

        hourlyHotWaterData,
      );


    /*
     * Site-level annual PV production.
     *
     * This is identical for every tank.
     */
    let pvProduction =
      0;


    for (
      const weather of
      quarterHour.weatherData
    ) {

      pvProduction +=
        Math.max(
          0,
          weather.pvPower,
        ) *
        TIME_STEP_HOURS;
    }


    const preparationDurationMs =
      performance.now() -
      preparationStart;


      /*
      * ============================================================
      * STORAGE SWEEP
      * ============================================================
      *
      * 300 ... 3000 litres
      * 100 litre steps
      *
      * 28 complete annual Plasma-Wave simulations.
      */
 
     const storageVolumes =
       Array.from(
         {
           length:
             ((3000 - 300) / 100) +
             1,
         },
         (_, index) =>
           300 + index * 100,
       );
 
 
     /*
      * ------------------------------------------------------------
      * PARALLEL BENCHMARK
      * ------------------------------------------------------------
      *
      * Start with two worker processes.
      *
      * After numerical parity has been confirmed,
      * only WORKER_COUNT will be changed to 4 and 8.
      */
 
     const WORKER_COUNT =
       TANK_COMPARISON_CONFIG.workerCount;
 
 
     const sweepStart =
       performance.now();
 
 
     const points =
       await runTankComparisonProcessPool({
         workerCount:
           WORKER_COUNT,
 
         storageVolumes,
 
         init: {
           weatherData:
             quarterHour.weatherData,
 
           hotWaterData:
             quarterHour.hotWaterData,
 
           heatPump,
 
           control,
 
           timeStepHours:
             TIME_STEP_HOURS,
 
           pvProduction,
 
           gridPriceChfPerKWh:
             body.gridPriceChfPerKWh,
 
           feedInPriceChfPerKWh:
             body.feedInPriceChfPerKWh,
         },
       });
 
 
     const sweepDurationMs =
       performance.now() -
       sweepStart;


    /*
     * ============================================================
     * PERFORMANCE STATISTICS
     * ============================================================
     */

    const durations =
      points.map(
        (
          point,
        ) =>
          point.simulationDurationMs,
      );


    const durationSum =
      durations.reduce(
        (
          total,
          value,
        ) =>
          total + value,
        0,
      );


    const averageSimulationDurationMs =
      durations.length > 0

        ? durationSum /
          durations.length

        : 0;


    const minimumSimulationDurationMs =
      durations.length > 0

        ? Math.min(
            ...durations,
          )

        : 0;


    const maximumSimulationDurationMs =
      durations.length > 0

        ? Math.max(
            ...durations,
          )

        : 0;


    const totalDurationMs =
      performance.now() -
      totalStart;


    const result:
      TankComparisonResponse = {

      fake:
        false,

      year:
        YEAR,

      heatPumpThermalPower:
        body.heatPumpThermalPower,

      pvPeakPower:
        body.pvPeakPower,

      gridPriceChfPerKWh:
        body.gridPriceChfPerKWh,

      feedInPriceChfPerKWh:
        body.feedInPriceChfPerKWh,

      preparationDurationMs,

      sweepDurationMs,

      totalDurationMs,

      averageSimulationDurationMs,

      minimumSimulationDurationMs,

      maximumSimulationDurationMs,

      points,
    };


    console.log(
      "[SunOpt] Tank comparison complete:",
      {
        tanks:
          points.length,

        preparationDurationMs:
          preparationDurationMs
            .toFixed(1),

        sweepDurationMs:
          sweepDurationMs
            .toFixed(1),

        totalDurationMs:
          totalDurationMs
            .toFixed(1),

        averageSimulationDurationMs:
          averageSimulationDurationMs
            .toFixed(1),

        minimumSimulationDurationMs:
          minimumSimulationDurationMs
            .toFixed(1),

        maximumSimulationDurationMs:
          maximumSimulationDurationMs
            .toFixed(1),
      },
    );


    return NextResponse.json(
      result,
    );

  } catch (
    error
  ) {

    console.error(
      "[SunOpt] Tank comparison failed.",
      error,
    );


    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : String(error),
      },
      {
        status:
          500,
      },
    );
  }
}
