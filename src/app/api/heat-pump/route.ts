// src/app/api/heat-pump/route.ts

import {
  NextRequest,
  NextResponse,
} from "next/server";

import {
  calculateHeatPumpStep,
  type HeatPump,
} from "@/domain";

interface RequestBody {
  heatPump: HeatPump;

  requestedThermalEnergy: number;
  timeStepHours?: number;
}

export async function POST(
  request: NextRequest,
) {
  try {
    const body =
      (await request.json()) as RequestBody;

    const result =
      calculateHeatPumpStep(
        body.heatPump,
        body.requestedThermalEnergy,
        body.timeStepHours ?? 1,
      );

    return NextResponse.json(result);

  } catch (error) {
    console.error(
      "Heat-pump API error:",
      error,
    );

    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Heat-pump calculation failed.",
      },
      {
        status: 400,
      },
    );
  }
}