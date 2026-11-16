// scripts/test-rust-no-sunshine.mjs

import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { performance } from "node:perf_hooks";


const require =
  createRequire(
    import.meta.url,
  );


const projectRoot =
  process.cwd();


const native =
  require(
    path.join(
      projectRoot,
      "rust",
      "sunopt-node",
    ),
  );


const sourcePath =
  path.join(
    projectRoot,
    "src",
    "simulation",
    "plasma-wave-no-sunshine-simulation.ts",
  );


const typescriptSource =
  fs.readFileSync(
    sourcePath,
    "utf8",
  );


if (
  !typescriptSource.includes(
    "rust_noSunshine:",
  )
) {
  throw new Error(
    "The real TypeScript source has no rust_noSunshine label.",
  );
}


const dataLength = 1024;

const pvPower =
  new Float64Array(
    dataLength,
  );

const loadEnergy =
  new Float64Array(
    dataLength,
  );


for (
  let index = 0;
  index < dataLength;
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


  if (quarter === 28) {
    loadEnergy[index] =
      0.35;
  }

  if (quarter === 80) {
    loadEnergy[index] =
      0.28;
  }
}


const parameters = {
  currentIndex: 173,

  horizonSteps: 672,

  protectedSteps: 192,

  timeStepHours: 0.25,

  householdReservePower: 2,

  maximumHeatPumpElectricalPower:
    5 / 3,

  cop: 3,

  initialWaterlinePower: 10,

  targetStorageEnergy: 52.325,

  storageEnergyPerKelvin:
    3000 *
    4.186 /
    3600,

  storageConductanceWPerKelvin:
    3.617,

  storageMinTemperature: 60,

  ambientTemperature: 20,

  /*
   * -1 selects the physical insulation model.
   */
  dailyHeatLoss: -1,

  currentStorageEnergy: 10,
};


native.clearNoSunshineKernelCache();


const firstStart =
  performance.now();


const first =
  native.runNoSunshineKernel(
    typescriptSource,
    pvPower,
    loadEnergy,
    parameters,
  );


const firstEnd =
  performance.now();


const cacheAfterFirst =
  native.getNoSunshineKernelCacheSize();


const secondStart =
  performance.now();


const second =
  native.runNoSunshineKernel(
    typescriptSource,
    pvPower,
    loadEnergy,
    parameters,
  );


const secondEnd =
  performance.now();


const cacheAfterSecond =
  native.getNoSunshineKernelCacheSize();


console.log(
  "First Node -> Oxc -> Cranelift -> JIT call:",
  `${(
    firstEnd -
    firstStart
  ).toFixed(3)} ms`,
);


console.log(
  "Second cached native 672-bucket call:",
  `${(
    secondEnd -
    secondStart
  ).toFixed(3)} ms`,
);


console.log(
  "Kernel cache:",
  cacheAfterFirst,
  "->",
  cacheAfterSecond,
);


console.log(
  "Result:",
  first,
);


const keys =
  Object.keys(
    first,
  );


for (const key of keys) {

  const a =
    first[key];

  const b =
    second[key];


  if (
    typeof a === "number" &&
    typeof b === "number" &&
    Math.abs(
      a - b,
    ) > 1e-12
  ) {

    throw new Error(
      `Cached run differs for ${key}: ${a} vs ${b}`,
    );
  }
}


if (
  cacheAfterFirst !== 1 ||
  cacheAfterSecond !== 1
) {
  throw new Error(
    "Expected exactly one cached JIT kernel.",
  );
}


console.log(
  "OK: Node native bridge works and reuses the JIT kernel.",
);
