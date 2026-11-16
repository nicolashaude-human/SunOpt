import { writeFileSync } from "node:fs";
import path from "node:path";

const baseUrl =
  process.argv[2] ??
  "http://localhost:3000";

const heatPumpThermalPower =
  Number.parseFloat(
    process.argv[3] ?? "5",
  );

const pvPeakPower =
  Number.parseFloat(
    process.argv[4] ?? "10",
  );

if (
  !Number.isFinite(
    heatPumpThermalPower,
  ) ||
  heatPumpThermalPower <= 0
) {
  throw new Error(
    "heatPumpThermalPower must be > 0",
  );
}

if (
  !Number.isFinite(
    pvPeakPower,
  ) ||
  pvPeakPower <= 0
) {
  throw new Error(
    "pvPeakPower must be > 0",
  );
}

const url =
  new URL(
    "/api/research/b12-physical-reference",
    baseUrl,
  );

console.log(
  `B12_REFERENCE_REQUEST=${url.href}`,
);

const response =
  await fetch(
    url,
    {
      method:
        "POST",

      headers: {
        "Content-Type":
          "application/json",
      },

      body:
        JSON.stringify({
          heatPumpThermalPower,
          pvPeakPower,
        }),
    },
  );

const text =
  await response.text();

let result;

try {
  result =
    JSON.parse(
      text,
    );
} catch {
  throw new Error(
    `B12 route returned invalid JSON (${response.status}):\n${text}`,
  );
}

console.log(
  `SUNOPT_B12_PHYSICAL_REFERENCE_JSON=${JSON.stringify(result)}`,
);

if (
  !response.ok ||
  !result.ok
) {
  throw new Error(
    result.error ??
    `B12 reference failed with HTTP ${response.status}`,
  );
}

const reportPath =
  path.resolve(
    "b12-physical-reference-result.json",
  );

writeFileSync(
  reportPath,
  JSON.stringify(
    result,
    null,
    2,
  ) +
    "\n",
  "utf8",
);

console.log();
console.log(
  "--- B12 PHYSICAL REFERENCE ---",
);

console.log(
  `steps=${result.stepCount}`,
);

console.log(
  `tanks=${result.tankCount}`,
);

console.log(
  `fields=${result.fieldCount}`,
);

console.log(
  `binaryBytes=${result.binaryBytes}`,
);

console.log(
  `binaryPath=${result.binaryPath}`,
);

console.log(
  `metadataPath=${result.metadataPath}`,
);

console.log(
  `coupledBytes=${result.coupledBytes}`,
);

console.log(
  `coupledPath=${result.coupledPath}`,
);

console.log(
  `coupledMetadataPath=${result.coupledMetadataPath}`,
);

console.log(
  `report=${reportPath}`,
);

