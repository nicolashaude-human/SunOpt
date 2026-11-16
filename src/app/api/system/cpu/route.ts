// src/app/api/system/cpu/route.ts

import os from "node:os";

import {
  NextResponse,
} from "next/server";


export const runtime =
  "nodejs";


export async function GET() {

  const cpus =
    os.cpus();


  return NextResponse.json({
    platform:
      os.platform(),

    architecture:
      os.arch(),

    availableParallelism:
      os.availableParallelism(),

    logicalCpuCount:
      cpus.length,

    cpuModel:
      cpus[0]?.model ??
      "unknown",

    cpuSpeedMHz:
      cpus[0]?.speed ??
      null,
  });
}