// src/server/pvgis.ts

import type {
  Location,
  PVSystem,
  WeatherData,
  WeatherDataPoint,
} from "@/domain";

interface PVGISHourlyPoint {
  time: string;
  P: number;
  "G(i)": number;
  T2m: number;
}

interface PVGISResponse {
  outputs: {
    hourly: PVGISHourlyPoint[];
  };
}

const PVGIS_BASE_URL =
  "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc";

export async function getPVGISData(
  location: Location,
  pvSystem: PVSystem,
  startYear: number,
  endYear: number,
): Promise<WeatherData> {

  try {
    const params = new URLSearchParams({
      lat: location.latitude.toString(),
      lon: location.longitude.toString(),

      startyear: startYear.toString(),
      endyear: endYear.toString(),

      pvcalculation: "1",

      peakpower: pvSystem.peakPower.toString(),

      // PVGIS expects percentage, e.g. 14 instead of 0.14
      loss: (pvSystem.systemLoss * 100).toString(),

      angle: pvSystem.inclination.toString(),
      aspect: pvSystem.azimuth.toString(),

      outputformat: "json",
    });

    const response = await fetch(
      `${PVGIS_BASE_URL}?${params.toString()}`,
    );

    if (!response.ok) {
      throw new Error(
        `PVGIS request failed: ${response.status} ${response.statusText}`,
      );
    }

    const data = (await response.json()) as PVGISResponse;

    if (
      !data.outputs ||
      !Array.isArray(data.outputs.hourly)
    ) {
      throw new Error(
        "PVGIS response contains no hourly data.",
      );
    }

    const weatherData: WeatherData =
      data.outputs.hourly.map(
        (point): WeatherDataPoint => ({
          timestamp: point.time,

          outdoorTemperature: point.T2m,

          irradiance: point["G(i)"],

          // PVGIS returns W; our domain uses kW.
          pvPower: point.P / 1000,
        }),
      );

    return weatherData;

  } catch (error) {
    console.error(
      "Failed to retrieve PVGIS data:",
      error,
    );

    throw error;
  }
}