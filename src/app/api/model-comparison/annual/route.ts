// src/app/api/model-comparison/annual/route.ts

import {
  NextRequest,
  NextResponse,
} from "next/server";

import type {
  HeatPump,
  HotWaterControl,
  HotWaterDemand,
  PVSystem,
  ThermalStorage,
} from "@/domain";

import type {
  ComparisonModelId,
  ModelComparisonResponse,
  MonthlyEnergyValues,
} from "@/types/model-comparison";

import { gunzipSync } from "node:zlib";
import { POST as simulateAnnualHotWater } from "../../debug/hot-water/route";
import { getCachedWeatherData, initializeReferenceWeatherCache } from "@/server/reference-weather-cache";

export const maxDuration = 300;
export const runtime = "nodejs";

interface RequestBody {
  postalCode: string;
  year: number;
  pvSystem: PVSystem;
  hotWaterDemand: HotWaterDemand;
  heatPump: HeatPump;
  storage: ThermalStorage;
  control: HotWaterControl;
}

interface ModelPoint {
  gridImportEnergy: number;
  exportAfterHpPower: number;
}

interface DebugPoint {
  timestamp: string;
  existing: ModelPoint;
  quantumGravity: ModelPoint;
  plasmaWave: ModelPoint;
}

interface DebugResponse {
  year: number;
  storageVolume: number;
  simulationDurationMs?: number;
  timeStepMinutes: number;
  data: DebugPoint[];
}

interface WeatherPoint {
  timestamp: string;
  pvPower: number;
}

const MODEL_IDS: ComparisonModelId[] = [
  "existing",
  "quantumGravity",
  "plasmaWave",
];

const MODEL_NAMES: Record<ComparisonModelId, string> = {
  existing: "Existing",
  quantumGravity: "Quantum Gravity",
  plasmaWave: "Plasma Wave",
};

function createEmptyMonthly(): MonthlyEnergyValues[] {
  return Array.from(
    { length: 12 },
    (_, index) => ({
      month: index + 1,
      gridImport: 0,
      feedIn: 0,
      selfConsumption: 0,
    }),
  );
}

function parseSunOptTimestamp(
  timestamp:
    string,
): Date {

  /*
   * PVGIS / SunOpt format:
   *
   * YYYYMMDD:HHMM
   *
   * Example:
   * 20230101:0010
   *
   * Means:
   * 2023-01-01 00:10 UTC
   */
  const match =
    /^(\d{4})(\d{2})(\d{2}):(\d{2})(\d{2})$/
      .exec(
        timestamp,
      );


  if (match) {

    const [
      ,
      year,
      month,
      day,
      hour,
      minute,
    ] =
      match;


    return new Date(
      Date.UTC(
        Number(
          year,
        ),
        Number(
          month,
        ) -
          1,
        Number(
          day,
        ),
        Number(
          hour,
        ),
        Number(
          minute,
        ),
      ),
    );
  }


  /*
   * Fallback for existing ISO timestamps.
   */
  const date =
    new Date(
      timestamp,
    );


  if (
    Number.isNaN(
      date.getTime(),
    )
  ) {

    throw new Error(
      `Invalid timestamp: ${timestamp}`,
    );
  }


  return date;
}


function getUtcMonth(
  timestamp:
    string,
): number {

  return (
    parseSunOptTimestamp(
      timestamp,
    )
      .getUTCMonth() +
    1
  );
}


function estimateWeatherStepHours(
  weatherData:
    WeatherPoint[],
): number {

  for (
    let index =
      1;

    index <
      weatherData.length;

    index++
  ) {

    const previous =
      weatherData[
        index -
        1
      ];

    const current =
      weatherData[
        index
      ];


    if (
      !previous ||
      !current
    ) {
      continue;
    }


    const previousTime =
      parseSunOptTimestamp(
        previous.timestamp,
      )
        .getTime();


    const currentTime =
      parseSunOptTimestamp(
        current.timestamp,
      )
        .getTime();


    const deltaHours =
      (
        currentTime -
        previousTime
      ) /
      3_600_000;


    if (
      Number.isFinite(
        deltaHours,
      ) &&
      deltaHours >
        0 &&
      deltaHours <=
        24
    ) {

      return deltaHours;
    }
  }


  throw new Error(
    "Could not determine weather timestep.",
  );
}

function sum(
  values:
    number[],
): number {

  return values.reduce(
    (
      total,
      value,
    ) =>
      total +
      value,
    0,
  );
}

export async function POST(
  request: NextRequest,
): Promise<NextResponse> {
  try {
    const body =
      (await request.clone().json()) as RequestBody;

    /*
     * Keep the expensive annual controller calculation in the
     * existing canonical simulation route.
     *
     * This endpoint is only a compact aggregation adapter.
     */
    await initializeReferenceWeatherCache();
    const weatherData = getCachedWeatherData({
      postalCode: body.postalCode, pvSystem: body.pvSystem,
      startYear: body.year, endYear: body.year,
    });
    const simulationResponse = await simulateAnnualHotWater(request);
    if (!simulationResponse.ok) {
      throw new Error(`Annual simulation failed with HTTP ${simulationResponse.status}: ${await simulationResponse.text()}`);
    }
    const payload = Buffer.from(await simulationResponse.arrayBuffer());
    const decoded = simulationResponse.headers.get("Content-Encoding") === "gzip"
      ? gunzipSync(payload)
      : payload;
    const simulation = JSON.parse(decoded.toString("utf8")) as DebugResponse;

    const weatherStepHours =
      estimateWeatherStepHours(
        weatherData,
      );

    const monthlyPvProduction =
      Array.from(
        { length: 12 },
        () => 0,
      );

    for (
      const point of
      weatherData
    ) {
      const month =
        getUtcMonth(
          point.timestamp,
        );

      monthlyPvProduction[
        month - 1
      ] +=
        Math.max(
          0,
          point.pvPower,
        ) *
        weatherStepHours;
    }

    const simulationStepHours =
      simulation.timeStepMinutes /
      60;

    const models =
      MODEL_IDS.map(
        (modelId) => {
          const monthly =
            createEmptyMonthly();

          for (
            const point of
            simulation.data
          ) {
            const month =
              getUtcMonth(
                point.timestamp,
              );

            const target =
              monthly[
                month - 1
              ];

            if (!target) {
              continue;
            }

            const modelPoint =
              point[modelId];

            target.gridImport +=
              modelPoint
                .gridImportEnergy;

            target.feedIn +=
              Math.max(
                0,
                modelPoint
                  .exportAfterHpPower,
              ) *
              simulationStepHours;
          }

          for (
            const month of
            monthly
          ) {
            const pvProduction =
              monthlyPvProduction[
                month.month - 1
              ] ?? 0;

            /*
             * Site-level PV self-consumption:
             *
             * PV production - grid feed-in.
             *
             * This includes both the household-reserved PV share
             * and PV directly consumed by the heat pump.
             */
            month.selfConsumption =
              Math.max(
                0,
                pvProduction -
                  month.feedIn,
              );
          }

          const gridImport =
            sum(
              monthly.map(
                (month) =>
                  month.gridImport,
              ),
            );

          const feedIn =
            sum(
              monthly.map(
                (month) =>
                  month.feedIn,
              ),
            );

          const selfConsumption =
            sum(
              monthly.map(
                (month) =>
                  month.selfConsumption,
              ),
            );

          const pvProduction =
            sum(
              monthlyPvProduction,
            );

          return {
            id:
              modelId,
            name:
              MODEL_NAMES[
                modelId
              ],
            monthly,
            annual: {
              gridImport,
              feedIn,
              selfConsumption,
              pvProduction,
              selfConsumptionRate:
                pvProduction > 0
                  ? (
                      selfConsumption /
                      pvProduction
                    ) * 100
                  : 0,
            },
          };
        },
      );


    const result:
      ModelComparisonResponse = {
      fake: false,
      year:
        simulation.year,
      storageVolume:
        simulation.storageVolume,
      simulationDurationMs:
        simulation
          .simulationDurationMs ??
        null,
      generatedAt:
        new Date()
          .toISOString(),
      models,
    };

    return NextResponse.json(
      result,
    );
  } catch (error) {
    console.error(
      "[SunOpt] Annual model comparison failed.",
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
        status: 500,
      },
    );
  }
}
