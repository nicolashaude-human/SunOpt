// src/app/api/debug/project-cleanup/delete/route.ts

import {
  NextResponse,
} from "next/server";

import {
  deleteCommittedOldFiles,
} from "@/lib/project-cleanup/cleanup-plan";


export const runtime =
  "nodejs";

export const dynamic =
  "force-dynamic";


export async function POST() {

  try {

    const result =
      deleteCommittedOldFiles();


    return NextResponse.json({
      ok:
        true,
      ...result,
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
