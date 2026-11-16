// src/app/api/optimization/hot-water/route.ts

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
  optimizeHotWaterStorage,
} from "@/optimization/hot-water-storage-optimizer";

import type {
  OptimizationTarget,
  StorageOptimizationRange,
} from "@/optimization/optimization-types";


interface WeatherApiResponse {
  weatherData: WeatherData;
}


interface RequestBody {
  postalCode: string;

  pvSystem: PVSystem;

  startYear: number;
  endYear: number;

  hotWaterDemand:
    HotWaterDemand;

  heatPump:
    HeatPump;

  storageTemplate:
    Omit<ThermalStorage, "volume">;

  control:
    HotWaterControl;

  range:
    StorageOptimizationRange;

  target:
    OptimizationTarget;
}


export async function POST(
  request: NextRequest,
): Promise<NextResponse> {

  try {

    const body =
      (await request.json()) as RequestBody;


    /*
     * --------------------------------------------
     * 1. BASIC VALIDATION
     * --------------------------------------------
     */

    if (
      !body.postalCode ||
      !/^\d{4}$/.test(body.postalCode)
    ) {
      return NextResponse.json(
        {
          error:
            "Eine gültige vierstellige Schweizer PLZ ist erforderlich.",
        },
        {
          status: 400,
        },
      );
    }


    /*
     * --------------------------------------------
     * 2. LOAD WEATHER + PV DATA
     * --------------------------------------------
     *
     * Reuse our existing /api/weather endpoint.
     */

    const weatherResponse =
      await fetch(
        `${request.nextUrl.origin}/api/weather`,
        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json",
          },

          body: JSON.stringify({
            postalCode:
              body.postalCode,

            pvSystem:
              body.pvSystem,

            startYear:
              body.startYear,

            endYear:
              body.endYear,
          }),

          cache: "no-store",
        },
      );


    if (!weatherResponse.ok) {

      const weatherError =
        await weatherResponse.text();

      throw new Error(
        `Weather API failed: ${weatherError}`,
      );
    }


    const weatherResult =
      (await weatherResponse.json()) as WeatherApiResponse;


    const weatherData =
      weatherResult.weatherData;


    if (
      !Array.isArray(weatherData) ||
      weatherData.length === 0
    ) {
      throw new Error(
        "Weather API returned no weather data.",
      );
    }


    /*
     * --------------------------------------------
     * 3. HOT-WATER DEMAND
     * --------------------------------------------
     */

    const hotWaterData =
      calculateAnnualHotWaterDemand(
        body.hotWaterDemand,
        DEFAULT_HOT_WATER_PROFILE,
        weatherData,
      );


    /*
     * --------------------------------------------
     * 4. STORAGE OPTIMIZATION
     * --------------------------------------------
     */

    const result =
      optimizeHotWaterStorage({
        weatherData,

        hotWaterData,

        heatPump:
          body.heatPump,

        storageTemplate:
          body.storageTemplate,

        control:
          body.control,

        range:
          body.range,

        target:
          body.target,
      });


    /*
     * --------------------------------------------
     * 5. RESULT
     * --------------------------------------------
     */

    return NextResponse.json({
      postalCode:
        body.postalCode,

      startYear:
        body.startYear,

      endYear:
        body.endYear,

      weatherDataPoints:
        weatherData.length,

      ...result,
    });

  } catch (error) {

    console.error(
      "Hot-water optimization error:",
      error,
    );


    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Optimization failed.",
      },
      {
        status: 400,
      },
    );
  }
}