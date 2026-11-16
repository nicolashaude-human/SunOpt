// src/app/api/simulation/space-heating/route.ts

import {
  NextRequest,
  NextResponse,
} from "next/server";

import {
  calculateSpaceHeatingProfile,
  type Building,
  type HeatPump,
  type ThermalStorage,
  type WeatherData,
} from "@/domain";

import {
  simulateSpaceHeatingSystem,
} from "@/simulation/space-heating-simulation";


interface RequestBody {
  weatherData: WeatherData;

  building: Building;

  heatPump: HeatPump;

  storage: ThermalStorage;
}


export async function POST(
  request: NextRequest,
) {
  try {

    const body =
      (await request.json()) as RequestBody;


    const heatingData =
      calculateSpaceHeatingProfile(
        body.building,
        body.weatherData,
      );


    const result =
      simulateSpaceHeatingSystem({
        weatherData:
          body.weatherData,

        heatingData,

        heatPump:
          body.heatPump,

        storage:
          body.storage,
      });


    return NextResponse.json({
      balance:
        result.balance,

      finalStorageState:
        result.finalStorageState,

      dataPoints:
        result.steps.length,

      steps:
        result.steps,
    });

  } catch (error) {

    console.error(
      "Space-heating simulation error:",
      error,
    );

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