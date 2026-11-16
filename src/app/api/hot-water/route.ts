// src/app/api/hot-water/route.ts

import {
  calculateDailyHotWaterDemand,
  DEFAULT_HOT_WATER_PROFILE,
  type HotWaterDemand,
} from "@/domain";

import {
  NextRequest,
  NextResponse,
} from "next/server";



export async function POST(
  request: NextRequest,
) {
  try {
    const demand =
      (await request.json()) as HotWaterDemand;

    if (
      !Number.isFinite(demand.persons) ||
      demand.persons <= 0 ||
      !Number.isFinite(
        demand.dailyConsumptionPerPerson,
      ) ||
      demand.dailyConsumptionPerPerson <= 0
    ) {
      return NextResponse.json(
        {
          error:
            "Ungültige Warmwasserdaten.",
        },
        {
          status: 400,
        },
      );
    }

    const data =
      calculateDailyHotWaterDemand(
        demand,
        DEFAULT_HOT_WATER_PROFILE,
      );

    const dailyVolume =
      data.reduce(
        (sum, point) =>
          sum + point.waterVolume,
        0,
      );

    const dailyThermalEnergy =
      data.reduce(
        (sum, point) =>
          sum + point.thermalEnergy,
        0,
      );
    

    return NextResponse.json({
      dailyVolume,
      dailyThermalEnergy,
      data,
    });

  } catch (error) {
    console.error(
      "Hot-water API error:",
      error,
    );

    return NextResponse.json(
      {
        error:
          "Warmwasserbedarf konnte nicht berechnet werden.",
      },
      {
        status: 500,
      },
    );
  }
}