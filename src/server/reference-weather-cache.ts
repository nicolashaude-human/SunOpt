// src/server/reference-weather-cache.ts

import {
  mkdir,
  readFile,
  rename,
  writeFile,
} from "node:fs/promises";

import {
  dirname,
  join,
} from "node:path";


export const REFERENCE_WEATHER_CONFIG = {
  postalCode: "8001",
  latitude: 47.3769,
  longitude: 8.5417,
  year: 2023,
  azimuth: 0,
  inclination: 30,
  systemLoss: 0.14,
  basePeakPower: 1,
  presentationPeakPower: 10,
} as const;


interface CachedWeatherPoint {
  timestamp: string;
  pvPowerPerKwp: number;
  temperature?: number;
  irradiance?: number;
  sunHeight?: number;
  windSpeed?: number;
}


interface DiskWeatherCache {
  format: 1;
  source: typeof REFERENCE_WEATHER_CONFIG;
  fetchedAt: string;
  points: CachedWeatherPoint[];
}


interface WeatherCacheState {
  status: "empty" | "loading" | "ready" | "error";
  loadingPromise: Promise<void> | null;
  points: CachedWeatherPoint[] | null;
  loadedAt: string | null;
  sourceMode: "network" | "disk" | null;
  error: string | null;
}


type GlobalWithWeatherCache = typeof globalThis & {
  __sunoptReferenceWeatherCache?: WeatherCacheState;
};


const globalWithWeatherCache =
  globalThis as GlobalWithWeatherCache;


const state: WeatherCacheState =
  globalWithWeatherCache.__sunoptReferenceWeatherCache ?? {
    status: "empty",
    loadingPromise: null,
    points: null,
    loadedAt: null,
    sourceMode: null,
    error: null,
  };


globalWithWeatherCache.__sunoptReferenceWeatherCache =
  state;


const CACHE_PATH =
  join(
    process.cwd(),
    "data",
    "cache",
    "weather",
    "reference-8001-2023-az0-inc30-loss14.json",
  );


const PRESENTATION_DATA_PATH =
  join(
    process.cwd(),
    "public",
    "presentation",
    "presentation-data.js",
  );


function finiteNumber(
  value: unknown,
): number | undefined {
  return typeof value === "number" &&
    Number.isFinite(value)
    ? value
    : undefined;
}


function pvgisTimestampToIso(
  value: unknown,
): string {
  if (typeof value !== "string") {
    throw new Error(
      "PVGIS hourly point has no timestamp.",
    );
  }

  const match =
    value.match(
      /^(\d{4})(\d{2})(\d{2}):(\d{2})(\d{2})$/,
    );

  if (!match) {
    const parsed = new Date(value);

    if (!Number.isFinite(parsed.getTime())) {
      throw new Error(
        `Unsupported PVGIS timestamp: ${value}`,
      );
    }

    return parsed.toISOString();
  }

  return new Date(
    Date.UTC(
      Number(match[1]),
      Number(match[2]) - 1,
      Number(match[3]),
      Number(match[4]),
      Number(match[5]),
    ),
  ).toISOString();
}


async function fetchReferenceWeatherFromPvgis():
  Promise<DiskWeatherCache> {

  const config =
    REFERENCE_WEATHER_CONFIG;

  const url =
    new URL(
      "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc",
    );

  const params = {
    lat: String(config.latitude),
    lon: String(config.longitude),
    startyear: String(config.year),
    endyear: String(config.year),
    pvcalculation: "1",
    peakpower: String(config.basePeakPower),
    loss: String(config.systemLoss * 100),
    angle: String(config.inclination),
    aspect: String(config.azimuth),
    outputformat: "json",
  };

  for (const [key, value] of Object.entries(params)) {
    url.searchParams.set(key, value);
  }

  console.info(
    "[SunOpt] Loading reference weather from PVGIS at server startup...",
  );

  const response =
    await fetch(
      url,
      {
        signal:
          AbortSignal.timeout(
            45_000,
          ),
        cache: "no-store",
      },
    );

  if (!response.ok) {
    throw new Error(
      `PVGIS startup request failed with HTTP ${response.status}.`,
    );
  }

  const payload =
    await response.json() as {
      outputs?: {
        hourly?: Array<Record<string, unknown>>;
      };
    };

  const hourly =
    payload.outputs?.hourly;

  if (!Array.isArray(hourly) || hourly.length < 8_000) {
    throw new Error(
      `PVGIS startup response contains only ${hourly?.length ?? 0} hourly points.`,
    );
  }

  const points: CachedWeatherPoint[] =
    hourly.map(
      (row) => {
        const pvPowerW =
          finiteNumber(row.P) ?? 0;

        return {
          timestamp:
            pvgisTimestampToIso(
              row.time,
            ),
          pvPowerPerKwp:
            Math.max(
              0,
              pvPowerW / 1000,
            ),
          temperature:
            finiteNumber(row.T2m),
          irradiance:
            finiteNumber(row["G(i)"]),
          sunHeight:
            finiteNumber(row.H_sun),
          windSpeed:
            finiteNumber(row.WS10m),
        };
      },
    );

  return {
    format: 1,
    source: config,
    fetchedAt: new Date().toISOString(),
    points,
  };
}


async function readDiskWeatherCache():
  Promise<DiskWeatherCache> {

  const text =
    await readFile(
      CACHE_PATH,
      "utf8",
    );

  const parsed =
    JSON.parse(text) as DiskWeatherCache;

  if (
    parsed.format !== 1 ||
    !Array.isArray(parsed.points) ||
    parsed.points.length < 8_000
  ) {
    throw new Error(
      "Stored SunOpt weather cache is invalid.",
    );
  }

  return parsed;
}


async function atomicWrite(
  path: string,
  content: string,
): Promise<void> {
  await mkdir(
    dirname(path),
    { recursive: true },
  );

  const temporaryPath =
    `${path}.tmp-${process.pid}`;

  await writeFile(
    temporaryPath,
    content,
    "utf8",
  );

  await rename(
    temporaryPath,
    path,
  );
}


async function writeDiskWeatherCache(
  cache: DiskWeatherCache,
): Promise<void> {
  await atomicWrite(
    CACHE_PATH,
    `${JSON.stringify(cache)}\n`,
  );
}


function scaledWeatherData(
  points: CachedWeatherPoint[],
  peakPower: number,
) {
  return points.map(
    (point) => ({
      timestamp:
        point.timestamp,
      pvPower:
        point.pvPowerPerKwp *
        peakPower,
      temperature:
        point.temperature,
      irradiance:
        point.irradiance,
      sunHeight:
        point.sunHeight,
      windSpeed:
        point.windSpeed,
    }),
  );
}


async function writePresentationSnapshot(
  cache: DiskWeatherCache,
): Promise<void> {
  const presentationPeakPower =
    REFERENCE_WEATHER_CONFIG
      .presentationPeakPower;

  const snapshot = {
    format: 1,
    generatedAt:
      new Date().toISOString(),
    source: {
      postalCode:
        REFERENCE_WEATHER_CONFIG
          .postalCode,
      year:
        REFERENCE_WEATHER_CONFIG
          .year,
      pvSystem: {
        peakPower:
          presentationPeakPower,
        azimuth:
          REFERENCE_WEATHER_CONFIG
            .azimuth,
        inclination:
          REFERENCE_WEATHER_CONFIG
            .inclination,
        systemLoss:
          REFERENCE_WEATHER_CONFIG
            .systemLoss,
      },
      cacheSource:
        state.sourceMode,
    },
    weatherData:
      scaledWeatherData(
        cache.points,
        presentationPeakPower,
      ),
  };

  const javascript = [
    "// public/presentation/presentation-data.js",
    "// Generated automatically by SunOpt at Node server startup.",
    "window.SUNOPT_PRESENTATION_DATA = ",
    JSON.stringify(snapshot),
    ";",
    "",
  ].join("");

  await atomicWrite(
    PRESENTATION_DATA_PATH,
    javascript,
  );
}


const WEATHER_RETRY_INITIAL_MS =
  5_000;

const WEATHER_RETRY_MAX_MS =
  5 * 60_000;


function delay(
  milliseconds: number,
): Promise<void> {
  return new Promise(
    (resolve) => {
      setTimeout(
        resolve,
        milliseconds,
      );
    },
  );
}


async function tryDiskCache():
  Promise<DiskWeatherCache | null> {
  try {
    const cache =
      await readDiskWeatherCache();

    state.sourceMode =
      "disk";

    return cache;

  } catch {
    return null;
  }
}


async function initializeInternal():
  Promise<void> {

  const deployed = process.env.VERCEL === "1";
  let retryDelayMs =
    WEATHER_RETRY_INITIAL_MS;

  let attempt =
    1;

  let cache:
    DiskWeatherCache | null =
    deployed ? await tryDiskCache() : null;

  if (deployed && cache === null) {
    throw new Error("The deployed reference weather cache is missing or invalid.");
  }

  while (cache === null) {
    try {
      cache =
        await fetchReferenceWeatherFromPvgis();

      state.sourceMode =
        "network";

      await writeDiskWeatherCache(
        cache,
      );

    } catch (networkError) {
      state.error =
        networkError instanceof Error
          ? networkError.message
          : String(networkError);

      console.warn(
        `[SunOpt] PVGIS startup attempt ${attempt} failed; checking disk cache.`,
        networkError,
      );

      cache =
        await tryDiskCache();

      if (cache !== null) {
        console.warn(
          "[SunOpt] Using the last valid weather disk cache.",
        );
        break;
      }

      console.warn(
        `[SunOpt] No weather data available. Retrying PVGIS in ${Math.round(retryDelayMs / 1000)} s.`,
      );

      state.status =
        "loading";

      await delay(
        retryDelayMs,
      );

      retryDelayMs =
        Math.min(
          WEATHER_RETRY_MAX_MS,
          retryDelayMs * 2,
        );

      attempt +=
        1;
    }
  }

  state.points =
    cache.points;

  state.loadedAt =
    new Date().toISOString();

  state.status =
    "ready";

  state.error =
    null;

  if (!deployed) {
    await writePresentationSnapshot(cache);
  }

  console.info(
    "[SunOpt] Reference weather cache ready:",
    {
      points:
        cache.points.length,
      mode:
        state.sourceMode,
      year:
        REFERENCE_WEATHER_CONFIG.year,
      postalCode:
        REFERENCE_WEATHER_CONFIG.postalCode,
    },
  );
}


export async function initializeReferenceWeatherCache():
  Promise<void> {

  if (state.status === "ready") {
    return;
  }

  if (state.loadingPromise) {
    return state.loadingPromise;
  }

  state.status =
    "loading";

  state.loadingPromise =
    initializeInternal()
      .catch(
        (error) => {
          state.status =
            "error";

          state.error =
            error instanceof Error
              ? error.message
              : String(error);

          throw error;
        },
      )
      .finally(
        () => {
          state.loadingPromise =
            null;
        },
      );

  return state.loadingPromise;
}


export interface CachedWeatherRequest {
  postalCode: string;
  startYear: number;
  endYear: number;
  pvSystem: {
    peakPower: number;
    azimuth: number;
    inclination: number;
    systemLoss: number;
  };
}


function nearlyEqual(
  left: number,
  right: number,
): boolean {
  return Math.abs(left - right) < 1e-9;
}


function validateRequest(
  request: CachedWeatherRequest,
): void {

  const config =
    REFERENCE_WEATHER_CONFIG;

  if (
    request.postalCode !== config.postalCode ||
    request.startYear !== config.year ||
    request.endYear !== config.year ||
    !nearlyEqual(request.pvSystem.azimuth, config.azimuth) ||
    !nearlyEqual(request.pvSystem.inclination, config.inclination) ||
    !nearlyEqual(request.pvSystem.systemLoss, config.systemLoss)
  ) {
    throw new Error(
      "Requested weather configuration is not part of the startup cache. " +
      "Restart SunOpt with the desired reference weather configuration instead of fetching weather during a simulation.",
    );
  }

  if (
    !Number.isFinite(request.pvSystem.peakPower) ||
    request.pvSystem.peakPower <= 0
  ) {
    throw new Error(
      "PV peakPower must be greater than zero.",
    );
  }
}


export function getCachedWeatherData(
  request: CachedWeatherRequest,
) {
  validateRequest(request);

  if (
    state.status !== "ready" ||
    state.points === null
  ) {
    throw new Error(
      state.error
        ? `Weather startup cache is unavailable: ${state.error}`
        : "Weather startup cache is not ready. Restart the Node server so instrumentation can preload it.",
    );
  }

  return scaledWeatherData(
    state.points,
    request.pvSystem.peakPower,
  );
}


export function getReferenceWeatherCacheStatus() {
  return {
    status:
      state.status,
    loadedAt:
      state.loadedAt,
    sourceMode:
      state.sourceMode,
    points:
      state.points?.length ?? 0,
    error:
      state.error,
    reference:
      REFERENCE_WEATHER_CONFIG,
  };
}
