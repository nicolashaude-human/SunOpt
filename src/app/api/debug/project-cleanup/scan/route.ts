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
  mkdir,
  open,
  rename,
  rm,
} from "node:fs/promises";

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


/** Write a complete file to the outbox, exposing it only after close + rename. */
async function writeOutboxFile(
  destination: string,
  contents: string,
): Promise<void> {
  const temporary = `${destination}.${randomUUID()}.tmp`;

  try {
    const handle = await open(temporary, "wx");
    try {
      await handle.writeFile(contents, "utf8");
      await handle.sync();
    } finally {
      await handle.close();
    }
    await rename(temporary, destination);
  } catch (error) {
    await rm(temporary, { force: true }).catch(() => undefined);
    throw error;
  }
}


/** The scanner response is already prepared; write its LLM exports to disk. */
async function saveScanToOutbox(
  result: ReturnType<typeof createProjectScanResponse>,
) {
  const directory = "updates/outbox";
  const outbox = join(process.cwd(), "updates", "outbox");
  await mkdir(outbox, { recursive: true });

  // UTC timestamp plus UUID: concurrent scans will never overwrite each other.
  const stamp = new Date().toISOString().replace(/\D/g, "").slice(0, 14);
  const suffix = `${stamp}-${randomUUID().slice(0, 8)}`;
  const llmManifest = `llm-project-cleanup-${suffix}.txt`;
  const candidateManifest = `llm-project-candidates-${suffix}.txt`;
  const jsonReport = `project-cleanup-scan-${suffix}.json`;

  const artifacts = [
    { name: llmManifest, text: result.llmManifest },
    { name: candidateManifest, text: result.candidateManifest },
    {
      name: jsonReport,
      text: JSON.stringify({
        generatedAt: new Date().toISOString(),
        scanner: "SunOpt Project Cleanup",
        result,
      }, null, 2) + "\n",
    },
  ];

  const committed: string[] = [];
  try {
    for (const artifact of artifacts) {
      const target = join(outbox, artifact.name);
      await writeOutboxFile(target, artifact.text);
      committed.push(target);
    }
  } catch (error) {
    // Do not leave a partially complete bundle marked as successful.
    await Promise.all(
      committed.map((file) => rm(file, { force: true }).catch(() => undefined)),
    );
    throw error;
  }

  return {
    saved: true as const,
    directory,
    llmManifest: `${directory}/${llmManifest}`,
    candidateManifest: `${directory}/${candidateManifest}`,
    jsonReport: `${directory}/${jsonReport}`,
  };
}


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


/**
 * The source tree contains a few review files outside its stable source set.
 * Move them temporarily so the scan stays deterministic.  A partial move is
 * rolled back before reporting an error.
 */
function parkExternalReviewArtifacts(): ParkedArtifact[] {
  const projectRoot = process.cwd();
  const parkingDirectory = join(
    projectRoot,
    "updates",
    "outbox",
    ".project-cleanup-scan-ignore",
  );
  mkdirSync(parkingDirectory, { recursive: true });

  const parked: ParkedArtifact[] = [];
  try {
    for (const filename of EXTERNAL_REVIEW_ARTIFACTS) {
      const originalPath = join(projectRoot, filename);
      if (!existsSync(originalPath)) continue;
      const parkedPath = join(
        parkingDirectory,
        `${randomUUID()}-${filename}`,
      );
      renameSync(originalPath, parkedPath);
      parked.push({ originalPath, parkedPath });
    }
    return parked;
  } catch (error) {
    const rollbackFailures = restoreExternalReviewArtifacts(parked);
    if (rollbackFailures.length > 0) {
      throw new Error(
        `Preparing scan failed; restoring moved files also failed: ${rollbackFailures.join("; ")}`,
        { cause: error },
      );
    }
    throw error;
  }
}

/** Attempt every restore, reporting failures without losing other restores. */
function restoreExternalReviewArtifacts(
  parked: ParkedArtifact[],
): string[] {
  const failures: string[] = [];
  for (const artifact of [...parked].reverse()) {
    try {
      if (!existsSync(artifact.parkedPath)) continue;
      if (existsSync(artifact.originalPath)) {
        throw new Error(
          `Original path already exists; leaving parked file untouched: ${artifact.originalPath}`,
        );
      }
      renameSync(artifact.parkedPath, artifact.originalPath);
    } catch (error) {
      failures.push(
        `${artifact.originalPath}: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }
  return failures;
}


export async function POST() {
  if (process.env.NODE_ENV === "production") {
    return NextResponse.json(
      { error: "Project Cleanup is disabled in production." },
      { status: 403 },
    );
  }

  let stage = "park";
  let parked: ParkedArtifact[] = [];
  let restored = false;

  try {
    // This must be inside try: Windows can fail a rename with EACCES/EPERM.
    parked = parkExternalReviewArtifacts();

    stage = "scan";
    const scan = scanProject();

    stage = "prepare-report";
    const result = createProjectScanResponse(scan);

    stage = "outbox";
    const outbox = await saveScanToOutbox(result);

    // Do not acknowledge success until all temporarily moved files return.
    stage = "restore";
    const restoreFailures = restoreExternalReviewArtifacts(parked);
    if (restoreFailures.length > 0) {
      throw new Error(restoreFailures.join("; "));
    }
    restored = true;

    stage = "response";
    return NextResponse.json({ ...result, outbox });
  } catch (error) {
    const restorationFailures = restored
      ? []
      : restoreExternalReviewArtifacts(parked);

    const message = error instanceof Error ? error.message : String(error);
    console.error("[SunOpt project-cleanup scan]", {
      stage,
      message,
      error,
      restorationFailures,
    });

    return NextResponse.json(
      {
        error: message,
        stage,
        ...(restorationFailures.length > 0
          ? { restorationFailures }
          : {}),
      },
      { status: 500 },
    );
  }
}
