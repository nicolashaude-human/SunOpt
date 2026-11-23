// src/app/api/tank-comparison/route.ts
// Default tank-comparison endpoint now uses the central single-context GPU lockstep sweep.

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
  HOT_WATER_TEST_CONFIG,
} from "@/config/config";

import {
  resampleToQuarterHour,
} from "@/simulation/resample-to-quarter-hour";

import {
  runTankComparisonGpuLockstep,
} from "@/simulation/tank-comparison-gpu-lockstep";

import {
  availableParallelism,
} from "node:os";

import {
  runTankComparisonProcessPool,
} from "@/workers/tank-comparison-process-pool";


import type {
  TankComparisonCausalPath,
  TankComparisonRequest,
  TankComparisonResponse,
} from "@/types/tank-comparison";


export const runtime =
  "nodejs";

export const dynamic =
  "force-dynamic";


interface WeatherApiResponse {
  weatherData:
    WeatherData;
}


const YEAR =
  2023;

const POSTAL_CODE =
  "8001";

const TIME_STEP_HOURS =
  HOT_WATER_TEST_CONFIG
    .timeStepHours;


function normalizeCausalPath(
  value:
    unknown,
): TankComparisonCausalPath {

  return value ===
    "A"
      ? "A"
      : "B";
}


export async function POST(
  request:
    NextRequest,
): Promise<NextResponse> {

  const totalStart =
    performance.now();


  try {

    const body =
      (
        await request.json()
      ) as TankComparisonRequest;


    const causalPath =
      normalizeCausalPath(
        body.causalPath,
      );


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
      (
        await weatherResponse.json()
      ) as WeatherApiResponse;


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


    const storageVolumes =
      Array.from(
        {
          length:
            (
              (
                3000 -
                300
              ) /
              100
            ) +
            1,
        },

        (
          _,
          index,
        ) =>
          300 +
          index *
          100,
      );


    const sweepStart =
      performance.now();


    if (causalPath === "A") {
      /*
       * Approach A: consolidated native A1+A2 via the original
       * sequential Plasma-Wave controller for EACH tank.
       * Distinct tanks can run in separate processes; the annual
       * feedback chain inside any tank remains strictly sequential.
       * The worker requires A1+A2 native capabilities and rejects
       * any silent TS fallback.
       */
      const workerCount = Math.max(1, Math.min(6, availableParallelism()));
      const approachA = await runTankComparisonProcessPool({
        workerCount,
        storageVolumes,
        init: {
          weatherData: quarterHour.weatherData,
          hotWaterData: quarterHour.hotWaterData,
          heatPump,
          control,
          timeStepHours: TIME_STEP_HOURS,
          pvProduction,
          gridPriceChfPerKWh: body.gridPriceChfPerKWh,
          feedInPriceChfPerKWh: body.feedInPriceChfPerKWh,
        },
      });
      const sweepDurationMs = performance.now() - sweepStart;
      const totalDurationMs = performance.now() - totalStart;
      const durations = approachA.points.map((p) => p.simulationDurationMs);
      const result: TankComparisonResponse = {
        fake: false,
        causalPath: "A",
        year: YEAR,
        heatPumpThermalPower: body.heatPumpThermalPower,
        pvPeakPower: body.pvPeakPower,
        gridPriceChfPerKWh: body.gridPriceChfPerKWh,
        feedInPriceChfPerKWh: body.feedInPriceChfPerKWh,
        preparationDurationMs,
        sweepDurationMs,
        totalDurationMs,
        averageSimulationDurationMs: durations.reduce((a, b) => a + b, 0) / durations.length,
        minimumSimulationDurationMs: Math.min(...durations),
        maximumSimulationDurationMs: Math.max(...durations),
        points: approachA.points,
        approachA: {
          backend: "A1+A2 Rust/Cranelift native process pool",
          workerCount: approachA.workerCount,
          nativeScopes: approachA.nativeScopes,
        },
      };
      return NextResponse.json(result);
    }

    // Approach B is intentionally unchanged: FP32 B14P CUDA production solver.
    const lockstep = runTankComparisonGpuLockstep({
      weatherData: quarterHour.weatherData,
      hotWaterData: quarterHour.hotWaterData,
      heatPump,
      control,
      timeStepHours: TIME_STEP_HOURS,
      pvProduction,
      gridPriceChfPerKWh: body.gridPriceChfPerKWh,
      feedInPriceChfPerKWh: body.feedInPriceChfPerKWh,
      storageVolumes,
      causalPath: "B",
    });

    const sweepDurationMs = performance.now() - sweepStart;
    const totalDurationMs = performance.now() - totalStart;
    const result: TankComparisonResponse & {
      gpuLockstep: typeof lockstep.profile;
    } = {
      fake: false,
      causalPath: "B",
      year: YEAR,
      heatPumpThermalPower: body.heatPumpThermalPower,
      pvPeakPower: body.pvPeakPower,
      gridPriceChfPerKWh: body.gridPriceChfPerKWh,
      feedInPriceChfPerKWh: body.feedInPriceChfPerKWh,
      preparationDurationMs,
      sweepDurationMs,
      totalDurationMs,
      averageSimulationDurationMs: sweepDurationMs,
      minimumSimulationDurationMs: sweepDurationMs,
      maximumSimulationDurationMs: sweepDurationMs,
      points: lockstep.points,
      gpuLockstep: lockstep.profile,
    };


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
