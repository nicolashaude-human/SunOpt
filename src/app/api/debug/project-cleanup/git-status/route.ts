// src/app/api/debug/project-cleanup/git-status/route.ts

import {
  NextResponse,
} from "next/server";

import {
  getGitCleanupStatus,
} from "@/lib/project-cleanup/git-safety";


export const runtime =
  "nodejs";

export const dynamic =
  "force-dynamic";


export async function GET() {

  try {

    return NextResponse.json(
      getGitCleanupStatus(),
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
