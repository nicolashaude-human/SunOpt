// src/app/api/weather/route.ts

import {
  NextRequest,
  NextResponse,
} from "next/server";

import {
  getCachedWeatherData,
  getReferenceWeatherCacheStatus,

  type CachedWeatherRequest,
} from "@/server/reference-weather-cache";


export const runtime =
  "nodejs";

export const dynamic =
  "force-dynamic";


export async function GET():
  Promise<NextResponse> {
  return NextResponse.json(
    getReferenceWeatherCacheStatus(),
  );
}


export async function POST(
  request: NextRequest,
): Promise<NextResponse> {

  try {
    const body =
      await request.json() as
        CachedWeatherRequest;

    const weatherData =
      getCachedWeatherData(
        body,
      ).map(point => ({
        ...point,
        outdoorTemperature: point.temperature,
      }));

    return NextResponse.json(
      {
        weatherData,
        cache: {
          kind:
            "startup-memory",
          ...getReferenceWeatherCacheStatus(),
        },
      },
      {
        headers: {
          "Cache-Control":
            "no-store",
          "X-SunOpt-Weather-Cache":
            "STARTUP",
        },
      },
    );

  } catch (error) {
    console.error(
      "[SunOpt] Cached weather request failed.",
      error,
    );

    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : String(error),
        cache:
          getReferenceWeatherCacheStatus(),
      },
      {
        status: 409,
      },
    );
  }
}
