// scripts/test-global-acceleration-parity.mjs

import {
  readFileSync,
} from "node:fs";

import {
  resolve,
} from "node:path";

import native from "sunopt-node";

import {
  runRustNoSunshineScope,
  runRustPeakShavingCausalScope,
} from "../src/simulation/rust-acceleration-scopes.ts";


const ABSOLUTE_TOLERANCE =
  1e-9;


function assertClose(
  name,
  actual,
  expected,
  tolerance =
    ABSOLUTE_TOLERANCE,
) {

  const difference =
    Math.abs(
      actual -
      expected,
    );


  if (
    !Number.isFinite(
      actual,
    ) ||
    !Number.isFinite(
      expected,
    ) ||
    difference >
      tolerance
  ) {

    throw new Error(
      `${name} differs: native=${actual}, ts=${expected}, diff=${difference}`,
    );
  }
}


function assertEqual(
  name,
  actual,
  expected,
) {

  if (
    actual !==
    expected
  ) {

    throw new Error(
      `${name} differs: native=${actual}, ts=${expected}`,
    );
  }
}


function printComparison(
  title,
  nativeResult,
  tsResult,
) {

  console.log(
    `\n${title}`,
  );


  for (
    const key of Object.keys(
      tsResult,
    )
  ) {

    console.log(
      `  ${key}:`,
      {
        native:
          nativeResult[key],

        ts:
          tsResult[key],
      },
    );
  }
}


const source =
  readFileSync(
    resolve(
      "src/simulation/rust-acceleration-scopes.ts",
    ),
    "utf8",
  );


/*
 * ================================================================
 * SHARED TEST DATA
 * ================================================================
 */

const length =
  1024;


const pvPower =
  new Float64Array(
    length,
  );


const loadEnergy =
  new Float64Array(
    length,
  );


for (
  let index = 0;
  index < length;
  index++
) {

  const quarter =
    index %
    96;


  const hour =
    quarter /
    4;


  if (
    hour >=
      6 &&
    hour <=
      18
  ) {

    const phase =
      (
        hour -
        6
      ) /
      12 *
      Math.PI;


    pvPower[
      index
    ] =
      8.5 *
      Math.max(
        0,
        Math.sin(
          phase,
        ),
      );
  }


  /*
   * Two realistic daily DHW draw events.
   */
  if (
    quarter ===
    28
  ) {

    loadEnergy[
      index
    ] =
      0.35;

  } else if (
    quarter ===
    80
  ) {

    loadEnergy[
      index
    ] =
      0.28;
  }
}


const storageParameters = {

  storageEnergyPerKelvin:
    3000 *
    4.186 /
    3600,

  storageConductanceWPerKelvin:
    3.617,

  storageMinTemperature:
    60,

  ambientTemperature:
    20,

  dailyHeatLoss:
    -1,
};


/*
 * ================================================================
 * 1. rust_noSunshine
 * ================================================================
 */

const noSunshineInput = {

  pvPower,

  loadEnergy,

  currentIndex:
    173,

  horizonSteps:
    672,

  protectedSteps:
    192,

  timeStepHours:
    0.25,

  householdReservePower:
    2,

  maximumHeatPumpElectricalPower:
    5 /
    3,

  cop:
    3,

  initialWaterlinePower:
    10,

  targetStorageEnergy:
    52.325,

  currentStorageEnergy:
    10,

  ...storageParameters,
};


const noSunshineTs =
  runRustNoSunshineScope(
    noSunshineInput,
  );


const noSunshineNative =
  native
    .runNoSunshineKernel(
      source,

      "rust_noSunshine",

      pvPower,

      loadEnergy,

      {
        currentIndex:
          noSunshineInput
            .currentIndex,

        horizonSteps:
          noSunshineInput
            .horizonSteps,

        protectedSteps:
          noSunshineInput
            .protectedSteps,

        timeStepHours:
          noSunshineInput
            .timeStepHours,

        householdReservePower:
          noSunshineInput
            .householdReservePower,

        maximumHeatPumpElectricalPower:
          noSunshineInput
            .maximumHeatPumpElectricalPower,

        cop:
          noSunshineInput
            .cop,

        initialWaterlinePower:
          noSunshineInput
            .initialWaterlinePower,

        targetStorageEnergy:
          noSunshineInput
            .targetStorageEnergy,

        currentStorageEnergy:
          noSunshineInput
            .currentStorageEnergy,

        storageEnergyPerKelvin:
          noSunshineInput
            .storageEnergyPerKelvin,

        storageConductanceWPerKelvin:
          noSunshineInput
            .storageConductanceWPerKelvin,

        storageMinTemperature:
          noSunshineInput
            .storageMinTemperature,

        ambientTemperature:
          noSunshineInput
            .ambientTemperature,

        dailyHeatLoss:
          noSunshineInput
            .dailyHeatLoss,
      },
    );


printComparison(
  "rust_noSunshine parity",
  noSunshineNative,
  noSunshineTs,
);


for (
  const key of Object.keys(
    noSunshineTs,
  )
) {

  assertClose(
    `rust_noSunshine.${key}`,
    noSunshineNative[
      key
    ],
    noSunshineTs[
      key
    ],
  );
}


/*
 * ================================================================
 * 2. rust_peakShavingCausal
 * ================================================================
 */

const householdReservePower =
  2;


const exportPower =
  Float64Array.from(
    pvPower,
    (power) =>
      Math.max(
        0,

        power -
        householdReservePower,
      ),
  );


const causalInput = {

  exportPower,

  demandEnergy:
    loadEnergy,

  currentIndex:
    173,

  horizonSteps:
    192,

  currentStorageEnergy:
    10,

  minimumReserveEnergy:
    2,

  targetStorageEnergy:
    52.325,

  maximumHeatPumpElectricalPower:
    5 /
    3,

  cop:
    3,

  energyCutLevel:
    4,

  timeStepHours:
    0.25,

  ...storageParameters,
};


const causalTs =
  runRustPeakShavingCausalScope(
    causalInput,
  );


const causalNative =
  native
    .runPeakShavingCausalKernel(
      source,

      "rust_peakShavingCausal",

      exportPower,

      loadEnergy,

      {
        currentIndex:
          causalInput
            .currentIndex,

        horizonSteps:
          causalInput
            .horizonSteps,

        currentStorageEnergy:
          causalInput
            .currentStorageEnergy,

        minimumReserveEnergy:
          causalInput
            .minimumReserveEnergy,

        targetStorageEnergy:
          causalInput
            .targetStorageEnergy,

        maximumHeatPumpElectricalPower:
          causalInput
            .maximumHeatPumpElectricalPower,

        cop:
          causalInput
            .cop,

        energyCutLevel:
          causalInput
            .energyCutLevel,

        timeStepHours:
          causalInput
            .timeStepHours,

        storageEnergyPerKelvin:
          causalInput
            .storageEnergyPerKelvin,

        storageConductanceWPerKelvin:
          causalInput
            .storageConductanceWPerKelvin,

        storageMinTemperature:
          causalInput
            .storageMinTemperature,

        ambientTemperature:
          causalInput
            .ambientTemperature,

        dailyHeatLoss:
          causalInput
            .dailyHeatLoss,
      },
    );


printComparison(
  "rust_peakShavingCausal parity",
  causalNative,
  causalTs,
);


assertClose(
  "rust_peakShavingCausal.cutLevelPower",
  causalNative
    .cutLevelPower,
  causalTs
    .cutLevelPower,
);


assertEqual(
  "rust_peakShavingCausal.normalPlanSurvives",
  causalNative
    .normalPlanSurvives,
  causalTs
    .normalPlanSurvives,
);


assertEqual(
  "rust_peakShavingCausal.maximumPvPlanSurvives",
  causalNative
    .maximumPvPlanSurvives,
  causalTs
    .maximumPvPlanSurvives,
);


assertClose(
  "rust_peakShavingCausal.survivalChecks",
  causalNative
    .survivalChecks,
  causalTs
    .survivalChecks,
);


assertClose(
  "rust_peakShavingCausal.binarySearchIterations",
  causalNative
    .binarySearchIterations,
  causalTs
    .binarySearchIterations,
);


/*
 * ================================================================
 * GLOBAL CACHE
 * ================================================================
 */

const cacheSize =
  native
    .getAccelerationScopeCacheSize();


console.log(
  "\nglobal scope cache size:",
  cacheSize,
);


if (
  cacheSize !==
  2
) {

  throw new Error(
    `Expected exactly 2 compiled acceleration scopes, got ${cacheSize}.`,
  );
}


console.log(
  "\nOK: TypeScript and native Oxc/Cranelift results are numerically identical for both acceleration scopes.",
);
