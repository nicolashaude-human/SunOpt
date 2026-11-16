// src/app/api/debug/project-cleanup/apply/route.ts

import {
  NextResponse,
} from "next/server";

import {
  renameCleanupTargets,
} from "@/lib/project-cleanup/cleanup-plan";

import type {
  CleanupPlan,
} from "@/lib/project-cleanup/types";


export const runtime =
  "nodejs";

export const dynamic =
  "force-dynamic";


export async function POST(
  request: Request,
) {

  try {

    const plan =
      await request.json() as
        CleanupPlan;


    const state =
      renameCleanupTargets(
        plan,
      );


    return NextResponse.json({
      ok:
        true,
      state,
    });

  } catch (error) {

    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : String(
                error,
              ),
      },
      {
        status:
          400,
      },
    );
  }
}
