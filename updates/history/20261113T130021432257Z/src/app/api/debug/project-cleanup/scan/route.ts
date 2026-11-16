// src/app/api/debug/project-cleanup/scan/route.ts

import {
  NextResponse,
} from "next/server";

import {
  createProjectScanResponse,
} from "@/lib/project-cleanup/project-tree";

import {
  scanProject,
} from "@/lib/project-cleanup/scanner";


export const runtime =
  "nodejs";

export const dynamic =
  "force-dynamic";


export async function POST() {

  if (
    process.env.NODE_ENV ===
    "production"
  ) {

    return NextResponse.json(
      {
        error:
          "Project Cleanup is disabled in production.",
      },
      {
        status:
          403,
      },
    );
  }


  try {

    const scan =
      scanProject();


    return NextResponse.json(
      createProjectScanResponse(
        scan,
      ),
    );

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
          500,
      },
    );
  }
}
