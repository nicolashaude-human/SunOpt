// rust/sunopt-node/scripts/blackwell-b12-phase3c-b.mjs
//
// B12 Phase 3C — A side.
//
// A reads the synchronized research files and converts them to the fixed,
// parameter-only B ABI.  B receives no file paths and no SunOpt objects.

import {
  createRequire,
} from "node:module";

import {
  readFileSync,
  writeFileSync,
} from "node:fs";

import path from "node:path";

import {
  performance,
} from "node:perf_hooks";

import {
  fileURLToPath,
} from "node:url";


const require =
  createRequire(
    import.meta.url,
  );

const here =
  path.dirname(
    fileURLToPath(
      import.meta.url,
    ),
  );

const root =
  path.resolve(
    here,
    "..",
  );

const addon =
  require(
    path.join(
      root,
      "sunopt_node.win32-x64-msvc.node",
    ),
  );


const snapshotPath =
  process.argv[2]
    ? path.resolve(
        process.argv[2],
      )
    : path.join(
        root,
        "blackwell-b12-sync-blackwell.bin",
      );

const physicalPath =
  process.argv[3]
    ? path.resolve(
        process.argv[3],
      )
    : path.join(
        root,
        "blackwell-physical-reference.bin",
      );

const deviceId =
  Number.parseInt(
    process.argv[4] ??
      "0",
    10,
  );

const chunkSteps =
  Number.parseInt(
    process.argv[5] ??
      "4096",
    10,
  );

const searchIterations =
  Number.parseInt(
    process.argv[6] ??
      "8",
    10,
  );


if (
  typeof addon
    .runB12Phase3cCrossParityB !==
  "function"
) {
  throw new Error(
    "sunopt-node does not export runB12Phase3cCrossParityB. Rebuild rust/sunopt-node with `npm run build` after installing this update.",
  );
}


function fail(
  message,
) {
  throw new Error(
    message,
  );
}


function readF64Array(
  buffer,
  state,
  count,
  label,
) {
  const bytes =
    count *
    8;

  if (
    state.offset +
      bytes >
    buffer.length
  ) {
    fail(
      `${label} exceeds file length.`,
    );
  }

  const result =
    new Float64Array(
      count,
    );

  for (
    let index = 0;
    index <
      count;
    index +=
      1
  ) {
    result[
      index
    ] =
      buffer.readDoubleLE(
        state.offset +
        index *
          8,
      );
  }

  state.offset +=
    bytes;

  return result;
}


function parseBlackwell(
  filePath,
) {
  const buffer =
    readFileSync(
      filePath,
    );

  const magic =
    Buffer.from(
      "SUNOPT-BW-BENCH\0",
      "ascii",
    );

  if (
    buffer.length <
      36 ||
    !buffer
      .subarray(
        0,
        16,
      )
      .equals(
        magic,
      )
  ) {
    fail(
      "Blackwell snapshot magic does not match.",
    );
  }


  const version =
    buffer.readUInt32LE(
      16,
    );

  const dataLength =
    buffer.readUInt32LE(
      20,
    );

  const tankCount =
    buffer.readUInt32LE(
      24,
    );

  const stepCount =
    buffer.readUInt32LE(
      28,
    );

  const configStride =
    buffer.readUInt32LE(
      32,
    );

  const dynamicStride =
    buffer.readUInt32LE(
      36,
    );


  if (
    version !==
      1 ||
    dataLength ===
      0 ||
    tankCount ===
      0 ||
    stepCount ===
      0 ||
    configStride !==
      12 ||
    dynamicStride !==
      3
  ) {
    fail(
      `Invalid synchronized Blackwell header: version=${version}, dataLength=${dataLength}, tanks=${tankCount}, steps=${stepCount}, configStride=${configStride}, dynamicStride=${dynamicStride}.`,
    );
  }


  const state = {
    offset:
      40,
  };


  const exportPower =
    readF64Array(
      buffer,
      state,
      dataLength,
      "exportPower",
    );

  const demandEnergy =
    readF64Array(
      buffer,
      state,
      dataLength,
      "demandEnergy",
    );

  const packedConfig =
    readF64Array(
      buffer,
      state,
      tankCount *
        configStride,
      "packedConfig",
    );


  const currentIndices =
    new Float64Array(
      stepCount,
    );

  const horizonSteps =
    new Float64Array(
      stepCount,
    );

  const dynamicValues =
    new Float64Array(
      stepCount *
      tankCount *
      dynamicStride,
    );


  for (
    let step = 0;
    step <
      stepCount;
    step +=
      1
  ) {
    if (
      state.offset +
        8 >
      buffer.length
    ) {
      fail(
        `Blackwell step ${step} header exceeds file length.`,
      );
    }


    currentIndices[
      step
    ] =
      buffer.readUInt32LE(
        state.offset,
      );

    horizonSteps[
      step
    ] =
      buffer.readUInt32LE(
        state.offset +
          4,
      );

    state.offset +=
      8;


    const count =
      tankCount *
      dynamicStride;

    const base =
      step *
      count;


    for (
      let index = 0;
      index <
        count;
      index +=
        1
    ) {
      dynamicValues[
        base +
        index
      ] =
        buffer.readDoubleLE(
          state.offset +
          index *
            8,
        );
    }


    state.offset +=
      count *
      8;
  }


  if (
    state.offset !==
    buffer.length
  ) {
    fail(
      `Blackwell snapshot has ${buffer.length - state.offset} trailing bytes.`,
    );
  }


  return {
    dataLength,
    tankCount,
    stepCount,
    packedConfig,
    exportPower,
    demandEnergy,
    currentIndices,
    horizonSteps,
    dynamicValues,
  };
}


function makeBTankParameters(
  snapshot,
) {
  const {
    tankCount,
    packedConfig,
  } =
    snapshot;

  const result =
    new Float64Array(
      tankCount *
      7,
    );


  for (
    let tank = 0;
    tank <
      tankCount;
    tank +=
      1
  ) {
    const source =
      tank *
      12;

    const targetStorageEnergy =
      packedConfig[
        source +
        2
      ];

    const maximumHeatPumpElectricalPower =
      packedConfig[
        source +
        3
      ];

    const cop =
      packedConfig[
        source +
        4
      ];

    const timeStepHours =
      packedConfig[
        source +
        6
      ];

    const storageEnergyPerKelvin =
      packedConfig[
        source +
        7
      ];

    const storageConductanceWPerKelvin =
      packedConfig[
        source +
        8
      ];

    const storageMinTemperature =
      packedConfig[
        source +
        9
      ];

    const ambientTemperature =
      packedConfig[
        source +
        10
      ];

    const dailyHeatLoss =
      packedConfig[
        source +
        11
      ];


    let heatLossRetention;
    let heatLossIntercept;
    let affineSupported;


    if (
      dailyHeatLoss >=
      0
    ) {
      heatLossRetention =
        1;

      heatLossIntercept =
        dailyHeatLoss *
        timeStepHours /
        24;

      affineSupported =
        true;

    } else if (
      storageEnergyPerKelvin >
      0
    ) {
      const conductanceEnergy =
        storageConductanceWPerKelvin *
        timeStepHours /
        1000;

      const slope =
        conductanceEnergy /
        storageEnergyPerKelvin;

      const intercept =
        conductanceEnergy *
        (
          storageMinTemperature -
          ambientTemperature
        );

      heatLossRetention =
        1 -
        slope;

      heatLossIntercept =
        intercept;

      affineSupported =
        slope >=
          0 &&
        slope <
          1 &&
        intercept >=
          0;

    } else {
      heatLossRetention =
        0;

      heatLossIntercept =
        0;

      affineSupported =
        false;
    }


    const destination =
      tank *
      7;


    result[
      destination +
      0
    ] =
      targetStorageEnergy;

    result[
      destination +
      1
    ] =
      maximumHeatPumpElectricalPower;

    result[
      destination +
      2
    ] =
      maximumHeatPumpElectricalPower *
      cop *
      timeStepHours;

    result[
      destination +
      3
    ] =
      heatLossRetention;

    result[
      destination +
      4
    ] =
      heatLossIntercept;

    result[
      destination +
      5
    ] =
      affineSupported
        ? 1
        : 0;

    result[
      destination +
      6
    ] =
      maximumHeatPumpElectricalPower *
      timeStepHours;
  }


  return result;
}


function makePresortedExportWindows(
  snapshot,
) {
  const {
    stepCount,
    dataLength,
    exportPower,
    currentIndices,
    horizonSteps,
  } =
    snapshot;


  const result =
    new Float64Array(
      stepCount *
      256,
    );

  result.fill(
    Number.NEGATIVE_INFINITY,
  );


  for (
    let step = 0;
    step <
      stepCount;
    step +=
      1
  ) {
    const horizon =
      horizonSteps[
        step
      ];


    if (
      !Number.isInteger(
        horizon,
      ) ||
      horizon <
        1 ||
      horizon >
        256
    ) {
      fail(
        `Invalid horizonSteps[${step}]=${horizon}.`,
      );
    }


    const current =
      currentIndices[
        step
      ];


    if (
      !Number.isInteger(
        current,
      ) ||
      current <
        0 ||
      current >=
        dataLength
    ) {
      fail(
        `Invalid currentIndices[${step}]=${current}.`,
      );
    }


    const base =
      step *
      256;

    const first =
      base +
      (
        256 -
        horizon
      );


    for (
      let offset = 0;
      offset <
        horizon;
      offset +=
        1
    ) {
      const index =
        (
          current +
          offset
        ) %
        dataLength;


      result[
        first +
        offset
      ] =
        exportPower[
          index
        ];
    }


    result
      .subarray(
        first,
        base +
          256,
      )
      .sort();
  }


  return result;
}


function parsePhysicalWaterlines(
  filePath,
  expectedSteps,
  expectedTanks,
) {
  const buffer =
    readFileSync(
      filePath,
    );


  if (
    buffer.length <
      40 ||
    buffer
      .subarray(
        0,
        15,
      )
      .toString(
        "ascii",
      ) !==
      "SUNOPT_B12_PHY1"
  ) {
    fail(
      "B12 physical reference magic does not match.",
    );
  }


  const version =
    buffer.readUInt32LE(
      16,
    );

  const stepCount =
    buffer.readUInt32LE(
      20,
    );

  const tankCount =
    buffer.readUInt32LE(
      24,
    );

  const fieldCount =
    buffer.readUInt32LE(
      28,
    );


  if (
    version !==
      1 ||
    stepCount !==
      expectedSteps ||
    tankCount !==
      expectedTanks ||
    fieldCount !==
      24
  ) {
    fail(
      `B12 physical dimensions do not match synchronized Blackwell snapshot: version=${version}, steps=${stepCount}, tanks=${tankCount}, fields=${fieldCount}.`,
    );
  }


  const dataOffset =
    40 +
    tankCount *
    8;

  const expectedBytes =
    dataOffset +
    tankCount *
    stepCount *
    fieldCount *
    8;


  if (
    buffer.length !==
    expectedBytes
  ) {
    fail(
      `B12 physical size mismatch: expected ${expectedBytes}, got ${buffer.length}.`,
    );
  }


  const jobCount =
    stepCount *
    tankCount;

  const userForecastWaterline =
    new Float64Array(
      jobCount,
    );

  const noSunshineWaterline =
    new Float64Array(
      jobCount,
    );

  const effectiveWaterline =
    new Float64Array(
      jobCount,
    );


  const USER_FIELD =
    10;

  const NO_SUNSHINE_FIELD =
    11;

  const EFFECTIVE_FIELD =
    12;


  for (
    let step = 0;
    step <
      stepCount;
    step +=
      1
  ) {
    for (
      let tank = 0;
      tank <
        tankCount;
      tank +=
        1
    ) {
      const job =
        step *
        tankCount +
        tank;

      const physicalBase =
        dataOffset +
        (
          (
            tank *
            stepCount +
            step
          ) *
          fieldCount
        ) *
        8;


      userForecastWaterline[
        job
      ] =
        buffer.readDoubleLE(
          physicalBase +
          USER_FIELD *
            8,
        );

      noSunshineWaterline[
        job
      ] =
        buffer.readDoubleLE(
          physicalBase +
          NO_SUNSHINE_FIELD *
            8,
        );

      effectiveWaterline[
        job
      ] =
        buffer.readDoubleLE(
          physicalBase +
          EFFECTIVE_FIELD *
            8,
        );
    }
  }


  return {
    userForecastWaterline,
    noSunshineWaterline,
    effectiveWaterline,
  };
}


const overallStart =
  performance.now();

const parseStart =
  performance.now();

const snapshot =
  parseBlackwell(
    snapshotPath,
  );

const physical =
  parsePhysicalWaterlines(
    physicalPath,
    snapshot.stepCount,
    snapshot.tankCount,
  );

const parseMs =
  performance.now() -
  parseStart;


const prepareStart =
  performance.now();

const tankParameters =
  makeBTankParameters(
    snapshot,
  );

const presortedExportWindows =
  makePresortedExportWindows(
    snapshot,
  );

const preparationMs =
  performance.now() -
  prepareStart;


const bStart =
  performance.now();

const result =
  addon
    .runB12Phase3cCrossParityB(
      snapshot.exportPower,
      snapshot.demandEnergy,
      tankParameters,
      presortedExportWindows,
      snapshot.currentIndices,
      snapshot.horizonSteps,
      snapshot.dynamicValues,
      physical.userForecastWaterline,
      physical.noSunshineWaterline,
      physical.effectiveWaterline,
      searchIterations,
      chunkSteps,
      deviceId,
    );

const bCallMs =
  performance.now() -
  bStart;

const totalMs =
  performance.now() -
  overallStart;


const report = {
  architecture:
    "A-files-and-mapping__B-flat-parameter-ABI",

  snapshotPath,
  physicalPath,

  aParseMs:
    parseMs,

  aPreparationMs:
    preparationMs,

  bCallMs,

  totalMs,

  ...result,
};


console.log(
  `SUNOPT_B12_PHASE3C_B_JSON=${JSON.stringify(report)}`,
);


console.log();
console.log(
  "--- B12 PHASE 3C: A/B CROSS PARITY ---",
);

console.log(
  `steps=${result.annualStepCount}`,
);

console.log(
  `tanks=${result.tankCount}`,
);

console.log(
  `jobs=${result.causalJobs}`,
);

console.log(
  `searchIterations=${result.searchIterations}`,
);

console.log(
  `chunkSteps=${result.chunkSteps}`,
);

console.log(
  `presortedMiB=${result.presortedMiB.toFixed(3)}`,
);


console.log();
console.log(
  "A PREPARATION",
);

console.log(
  `parseMs=${parseMs.toFixed(3)}`,
);

console.log(
  `preparationMs=${preparationMs.toFixed(3)}`,
);


console.log();
console.log(
  "B CUDA",
);

console.log(
  `inputH2DMs=${result.inputH2DMs.toFixed(3)}`,
);

console.log(
  `kernelMs=${result.kernelMs.toFixed(3)}`,
);

console.log(
  `resultD2HMs=${result.resultD2HMs.toFixed(3)}`,
);

console.log(
  `wallMs=${result.wallMs.toFixed(3)}`,
);

console.log(
  `bCallMs=${bCallMs.toFixed(3)}`,
);

console.log(
  `checksum=${result.checksumHex}`,
);


console.log();
console.log(
  "CROSS PARITY",
);

console.log(
  `cut vs userForecast mismatches1e-12=${result.cutVsUserForecastMismatches1e12} mismatches1e-9=${result.cutVsUserForecastMismatches1e9} maxError=${result.cutVsUserForecastMaxAbsError.toExponential(6)} maxJob=${result.cutVsUserForecastMaxErrorJob}`,
);

console.log(
  `cut vs noSunshine mismatches1e-12=${result.cutVsNoSunshineMismatches1e12} mismatches1e-9=${result.cutVsNoSunshineMismatches1e9} maxError=${result.cutVsNoSunshineMaxAbsError.toExponential(6)} maxJob=${result.cutVsNoSunshineMaxErrorJob}`,
);

console.log(
  `cut vs effective mismatches1e-12=${result.cutVsEffectiveMismatches1e12} mismatches1e-9=${result.cutVsEffectiveMismatches1e9} maxError=${result.cutVsEffectiveMaxAbsError.toExponential(6)} maxJob=${result.cutVsEffectiveMaxErrorJob}`,
);

console.log(
  `nonFiniteCuts=${result.nonFiniteCuts}`,
);

console.log(
  `nonFinitePhysical=${result.nonFinitePhysical}`,
);


console.log();
console.log(
  "PHASE 3D MISMATCH CLASSIFICATION",
);

console.log(
  result.phase3dSummary,
);

console.log();
console.log(
  "MISMATCHES PER TANK (zero-based tank index)",
);

console.log(
  result.phase3dTankMismatchCounts,
);

console.log();
console.log(
  "TOP 12 USER-FORECAST MISMATCHES",
);

console.log(
  result.phase3dTopMismatches,
);


const reportPath =
  path.join(
    root,
    "blackwell-b12-phase3c-b.json",
  );

writeFileSync(
  reportPath,
  JSON.stringify(
    report,
    null,
    2,
  ) +
    "\n",
  "utf8",
);

console.log();
console.log(
  `report=${reportPath}`,
);

