// scripts/test-global-acceleration-scopes.mjs

import {
  readFileSync,
} from "node:fs";

import {
  resolve,
} from "node:path";

import native from "sunopt-node";


const source =
  readFileSync(
    resolve(
      "src/simulation/rust-acceleration-scopes.ts",
    ),
    "utf8",
  );


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
    index % 96;

  const hour =
    quarter / 4;


  if (
    hour >= 6 &&
    hour <= 18
  ) {

    const phase =
      (
        hour - 6
      ) /
      12 *
      Math.PI;


    pvPower[index] =
      8.5 *
      Math.max(
        0,
        Math.sin(
          phase,
        ),
      );
  }


  if (
    quarter === 28
  ) {

    loadEnergy[index] =
      0.35;

  } else if (
    quarter === 80
  ) {

    loadEnergy[index] =
      0.28;
  }
}


const commonStorage = {
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


native.clearAccelerationScopeCache();


const noSunshine =
  native.runNoSunshineKernel(
    source,

    "rust_noSunshine",

    pvPower,

    loadEnergy,

    {
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
        5 / 3,

      cop:
        3,

      initialWaterlinePower:
        10,

      targetStorageEnergy:
        52.325,

      currentStorageEnergy:
        10,

      ...commonStorage,
    },
  );


const exportPower =
  Float64Array.from(
    pvPower,
    (power) =>
      Math.max(
        0,
        power - 2,
      ),
  );


const causal =
  native.runPeakShavingCausalKernel(
    source,

    "rust_peakShavingCausal",

    exportPower,

    loadEnergy,

    {
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
        5 / 3,

      cop:
        3,

      energyCutLevel:
        4,

      timeStepHours:
        0.25,

      ...commonStorage,
    },
  );


const cacheSize =
  native.getAccelerationScopeCacheSize();


console.log(
  "rust_noSunshine:",
  noSunshine,
);


console.log(
  "rust_peakShavingCausal:",
  causal,
);


console.log(
  "global scope cache size:",
  cacheSize,
);


if (
  cacheSize !== 2
) {

  throw new Error(
    `Expected two compiled scope IDs in one global cache, got ${cacheSize}.`,
  );
}


for (
  const [
    name,
    value,
  ] of [
    [
      "noSunshine.currentWaterlinePower",
      noSunshine.currentWaterlinePower,
    ],
    [
      "causal.cutLevelPower",
      causal.cutLevelPower,
    ],
  ]
) {

  if (
    !Number.isFinite(
      value,
    )
  ) {

    throw new Error(
      `${name} is not finite.`,
    );
  }
}


console.log(
  "OK: one TypeScript source -> two scope IDs -> one Oxc/Cranelift cache.",
);
