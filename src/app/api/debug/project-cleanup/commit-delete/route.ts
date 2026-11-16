// src/app/api/debug/project-cleanup/commit-delete/route.ts

import {
  NextResponse,
} from "next/server";

import {
  commitOldDeletion,
} from "@/lib/project-cleanup/git-safety";


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

    return NextResponse.json({
      ok:
        true,
      ...commitOldDeletion(),
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
