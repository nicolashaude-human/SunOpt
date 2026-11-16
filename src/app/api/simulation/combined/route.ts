// src/app/api/simulation/combined/route.ts

import {
  NextRequest,
  NextResponse,
} from "next/server";

import {
  calculateAnnualHotWaterDemand,
  calculateSpaceHeatingProfile,
  combineHeatDemand,
  DEFAULT_HOT_WATER_PROFILE,
  type Building,
  type CombinedThermalStorage,
  type HeatPump,
  type HotWaterDemand,
  type WeatherData,
} from "@/domain";

import {
  simulateCombinedSystem,
} from "@/simulation/combined-simulation";


interface RequestBody {
  weatherData: WeatherData;

  hotWaterDemand: HotWaterDemand;
  building: Building;

  heatPump: HeatPump;
  storage: CombinedThermalStorage;
}


export async function POST(
  request: NextRequest,
): Promise<NextResponse> {

  try {
    const body =
      (await request.json()) as RequestBody;


    // --------------------------------
    // Eingabedaten prüfen
    // --------------------------------

    if (
      !Array.isArray(body.weatherData) ||
      body.weatherData.length === 0
    ) {
      return NextResponse.json(
        {
          error:
            "weatherData fehlt oder enthält keine Wetterdaten.",
        },
        {
          status: 400,
        },
      );
    }


    // --------------------------------
    // Warmwasser
    // --------------------------------

    const hotWaterData =
      calculateAnnualHotWaterDemand(
        body.hotWaterDemand,
        DEFAULT_HOT_WATER_PROFILE,
        body.weatherData,
      );


    // --------------------------------
    // Raumheizung
    // --------------------------------

    const heatingData =
      calculateSpaceHeatingProfile(
        body.building,
        body.weatherData,
      );


    // --------------------------------
    // Wärmebedarf kombinieren
    // --------------------------------

    const heatDemand =
      combineHeatDemand(
        hotWaterData,
        heatingData,
      );


    // --------------------------------
    // Simulation
    // --------------------------------

    const result =
      simulateCombinedSystem({
        weatherData:
          body.weatherData,

        heatDemand,

        heatPump:
          body.heatPump,

        storage:
          body.storage,
      });


    // --------------------------------
    // Erfolgreiche Antwort
    // --------------------------------

    return NextResponse.json({
      balance:
        result.balance,

      finalHotWaterStorageState:
        result.finalHotWaterStorageState,

      finalHeatingStorageState:
        result.finalHeatingStorageState,

      dataPoints:
        result.steps.length,

      steps:
        result.steps,
    });

  } catch (error) {

    console.error(
      "Combined simulation error:",
      error,
    );


    // --------------------------------
    // Fehlerantwort
    // --------------------------------

    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Simulation failed.",
      },
      {
        status: 400,
      },
    );
  }
}