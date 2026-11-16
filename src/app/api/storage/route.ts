// src/app/api/storage/route.ts

import {
  NextRequest,
  NextResponse,
} from "next/server";

import {
  calculateStorageCapacity,
  createInitialStorageState,
  simulateStorageStep,
  type ThermalStorage,
} from "@/domain";

interface RequestBody {
  storage: ThermalStorage;

  chargeEnergy: number;
  demandEnergy: number;

  timeStepHours?: number;
}

export async function POST(
  request: NextRequest,
) {
  try {
    const body =
      (await request.json()) as RequestBody;

    const state =
      createInitialStorageState(
        body.storage,
      );

    const result =
      simulateStorageStep(
        body.storage,
        state,
        body.chargeEnergy,
        body.demandEnergy,
        body.timeStepHours ?? 1,
      );

    return NextResponse.json({
      capacity:
        calculateStorageCapacity(
          body.storage,
        ),

      initialState: state,

      result,
    });

  } catch (error) {
    console.error(
      "Storage API error:",
      error,
    );

    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Storage calculation failed.",
      },
      {
        status: 400,
      },
    );
  }
}