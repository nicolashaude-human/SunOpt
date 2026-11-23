// src/app/api/debug/project-cleanup/scan/route.ts

import {
  existsSync,
  mkdirSync,
  renameSync,
} from "node:fs";

import {
  randomUUID,
} from "node:crypto";

import {
  join,
} from "node:path";

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


const EXTERNAL_REVIEW_ARTIFACTS = [
  "cleanup-analysis.json",
  "sunopt-project-manifest.llm.txt",
  "sunopt-project-candidates.llm.txt",
] as const;


type ParkedArtifact = {
  originalPath:
    string;
  parkedPath:
    string;
};


function parkExternalReviewArtifacts():
  ParkedArtifact[] {

  const projectRoot =
    process.cwd();

  const parkingDirectory =
    join(
      projectRoot,
      "updates",
      "outbox",
      ".project-cleanup-scan-ignore",
    );

  mkdirSync(
    parkingDirectory,
    {
      recursive:
        true,
    },
  );

  const parked:
    ParkedArtifact[] =
    [];

  for (
    const filename of
      EXTERNAL_REVIEW_ARTIFACTS
  ) {

    const originalPath =
      join(
        projectRoot,
        filename,
      );

    if (
      !existsSync(
        originalPath,
      )
    ) {
      continue;
    }

    const parkedPath =
      join(
        parkingDirectory,
        `${randomUUID()}-${filename}`,
      );

    renameSync(
      originalPath,
      parkedPath,
    );

    parked.push({
      originalPath,
      parkedPath,
    });
  }

  return parked;
}


function restoreExternalReviewArtifacts(
  parked:
    ParkedArtifact[],
): void {

  for (
    const artifact of
      [...parked].reverse()
  ) {

    if (
      existsSync(
        artifact.parkedPath,
      )
    ) {
      renameSync(
        artifact.parkedPath,
        artifact.originalPath,
      );
    }
  }
}


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


  const parked =
    parkExternalReviewArtifacts();


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

  } finally {

    restoreExternalReviewArtifacts(
      parked,
    );
  }
}
