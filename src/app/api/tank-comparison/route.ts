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
  startSunoptGpuProfileGuard,
  type SunoptGpuProfileReport,
} from "@/simulation/sunopt-gpu-profile-guard";

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


    // Fail closed: the proven B14P/Q NVAPI guard must set the 1050 mV
    // V/F point, +625 MHz curve offset and manual 70% fan COMMAND before
    // productive A/B. NVML here only reads telemetry. The legacy guard
    // restores the original clock lock, curve and fan control on STOP.
    // Observed fan speed is allowed to ramp: it is not the PWM setpoint.
    const gpuGuardPreparationStart = performance.now();
    const gpuGuard = await startSunoptGpuProfileGuard(causalPath, request.signal);
    const gpuQueueDurationMs = gpuGuard.queueDurationMs;
    const gpuGuardPreparationDurationMs = performance.now() - gpuGuardPreparationStart - gpuQueueDurationMs;
    const sweepStart = performance.now();
    let sweepEnd = sweepStart;

    /*
     * Approach A production winner (2026-11-13): DIRECT FP64.
     * The single-step CUDA Graph candidate (AG) was parity-exact,
     * but did not improve the end-to-end median.  The native worker
     * checks SUNOPT_APPROACH_A_STEP_GRAPH on every causal call,
     * so force it OFF for this synchronous production A sweep.
     * Restore the original value even if the sweep throws.
     * Approach B is intentionally untouched.
     */
    const graphEnvironmentKey =
      "SUNOPT_APPROACH_A_STEP_GRAPH";

    const previousGraphSetting =
      process.env[graphEnvironmentKey];

    if (causalPath === "A") {
      process.env[graphEnvironmentKey] = "0";
      console.info(
        "[SunOpt] Approach A selected: direct FP64 CUDA (final benchmark winner).",
      );
    }

    let lockstep:
      ReturnType<typeof runTankComparisonGpuLockstep>;

    let gpuClockProfile: SunoptGpuProfileReport;
    let gpuGuardRestoreDurationMs = 0;
    try {
      try {
        lockstep =
        runTankComparisonGpuLockstep(
        {
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

          storageVolumes,

          causalPath,
        },
      );
        sweepEnd = performance.now();
      } finally {
        if (causalPath === "A") {
          if (previousGraphSetting === undefined) {
            delete process.env[graphEnvironmentKey];
          } else {
            process.env[graphEnvironmentKey] = previousGraphSetting;
          }
        }
      }
    } finally {
      // Await telemetry and restoration; rejected clock/fan readings
      // invalidate the production measurement rather than mislabel it.
      const restoreStart = performance.now();
      gpuClockProfile = await gpuGuard.stop();
      gpuGuardRestoreDurationMs = performance.now() - restoreStart;
    }

    if (lockstep.profile.causalPath !== causalPath) {
      throw new Error(
        `GPU solver path mismatch: requested ${causalPath}, reported ${lockstep.profile.causalPath}.`,
      );
    }

    const sweepDurationMs =
      sweepEnd -
      sweepStart;


    const totalDurationMs =
      performance.now() -
      totalStart;


    const result:
      TankComparisonResponse & {
        gpuLockstep:
          typeof lockstep.profile;
        gpuClockProfile: SunoptGpuProfileReport;
        gpuGuardPreparationDurationMs: number;
        gpuGuardRestoreDurationMs: number;
        gpuQueueDurationMs: number;
      } = {

      fake:
        false,

      causalPath:
        lockstep.profile
          .causalPath,

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

      // Die Laufzeit des eigentlichen CUDA-Sweeps wird NICHT mit dem
      // Hochfahren von GPU-Takt und Lueftern vermischt.
      gpuGuardPreparationDurationMs,
      gpuGuardRestoreDurationMs,
      gpuQueueDurationMs,

      totalDurationMs,

      averageSimulationDurationMs:
        sweepDurationMs,

      minimumSimulationDurationMs:
        sweepDurationMs,

      maximumSimulationDurationMs:
        sweepDurationMs,

      points:
        lockstep.points,
      thermalReserve: lockstep.thermalReserve,

      gpuLockstep:
        lockstep.profile,
      gpuClockProfile,
    };


    return NextResponse.json(
      result,
    );

  } catch (
    error
  ) {

    console.error(
      "[SunOpt] GPU lockstep tank comparison failed.",
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
