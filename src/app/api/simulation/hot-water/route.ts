// src/app/api/simulation/hot-water/route.ts

import {
  NextRequest,
  NextResponse,
} from "next/server";

import {
  calculateAnnualHotWaterDemand,
  DEFAULT_HOT_WATER_PROFILE,
  type HeatPump,
  type HotWaterDemand,
  type ThermalStorage,
  type WeatherData,
} from "@/domain";

import {
  simulateHotWaterSystem,
} from "@/simulation/hot-water-simulation";

import {
  CONFIG,
} from "@/config/config";


interface RequestBody {
  weatherData: WeatherData;

  hotWaterDemand: HotWaterDemand;

  heatPump: HeatPump;

  storage: ThermalStorage;
}


export async function POST(
  request: NextRequest,
) {
  try {

    const body =
      (await request.json()) as RequestBody;


    const hotWaterData =
      calculateAnnualHotWaterDemand(
        body.hotWaterDemand,
        DEFAULT_HOT_WATER_PROFILE,
        body.weatherData,
      );


    const result =
      simulateHotWaterSystem({
        weatherData:
          body.weatherData,

        hotWaterData,

        heatPump:
          body.heatPump,

        storage:
          body.storage,

        control: {
          minimumReserveHours:
            CONFIG.control.minimumReserveHours,

          forecastHorizonHours:
            CONFIG.control.forecastHorizonHours,

          pvTargetTemperature:
            CONFIG.control.pvTargetTemperature,

          householdReservePower:
            CONFIG.control.householdReservePower,

          forecastSafetyFactor:
            CONFIG.control.forecastSafetyFactor,
        },

        timeStepHours:
          CONFIG.simulation.timeStepHours,
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
      "Hot-water simulation error:",
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