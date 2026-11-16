import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here =
  path.dirname(
    fileURLToPath(import.meta.url)
  );

const root =
  path.resolve(
    here,
    "..",
    "..",
    ".."
  );

function read(rel) {
  return fs.readFileSync(
    path.join(root, rel),
    "utf8"
  );
}

const types =
  read("src/types/tank-comparison.ts");

const route =
  read("src/app/api/tank-comparison/route.ts");

const lockstep =
  read("src/simulation/tank-comparison-gpu-lockstep.ts");

const page =
  read("src/app/tank-comparison/page.tsx");

const checks = {
  sharedType:
    types.includes(
      "export type TankComparisonCausalPath"
    ),

  sharedRequest:
    types.includes(
      "export interface TankComparisonRequest"
    ),

  responseCarriesPath:
    types.includes(
      "causalPath:\n    TankComparisonCausalPath;"
    ),

  routeNormalizes:
    route.includes(
      "normalizeCausalPath("
    ),

  routePassesPath:
    route.includes(
      "          causalPath,"
    ),

  routeReturnsActualPath:
    route.includes(
      "lockstep.profile\n          .causalPath"
    ),

  lockstepUsesSharedPath:
    lockstep.includes(
      "const causalPath:\n    TankComparisonCausalPath ="
    ),

  lockstepWorkerSelection:
    lockstep.includes(
      "createCudaCausalWorker("
    ) &&
    lockstep.includes(
      "      causalPath,\n    );"
    ),

  lockstepProfileCarriesPath:
    lockstep.includes(
      "    causalPath,\n\n    tanks:"
    ),

  uiSendsPath:
    page.includes(
      "                  causalPath,"
    ),

  uiDisplaysResultPath:
    page.includes(
      "Pfad {result.causalPath}"
    ),

  noShadowResultPathState:
    !page.includes(
      "resultCausalPath"
    ),
};

const passed =
  Object.values(checks)
    .every(Boolean);

const report = {
  stage:
    "PATH_INTEGRATION_VERIFY",

  checks,

  passed,
};

console.log(
  JSON.stringify(
    report,
    null,
    2
  )
);

if (!passed) {
  process.exitCode = 1;
}
