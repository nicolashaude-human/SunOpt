// rust/sunopt-node/scripts/blackwell-b12-state-handoff.mjs
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here =
  path.dirname(
    fileURLToPath(import.meta.url),
  );

const root =
  path.resolve(
    here,
    "..",
  );

const snapshot =
  process.argv[2]
    ? path.resolve(process.argv[2])
    : path.join(
        root,
        "blackwell-benchmark.bin",
      );

const physical =
  process.argv[3]
    ? path.resolve(process.argv[3])
    : path.join(
        root,
        "blackwell-physical-reference.bin",
      );

console.log(
  `B12_BLACKWELL_SNAPSHOT=${snapshot}`,
);

console.log(
  `B12_PHYSICAL_REFERENCE=${physical}`,
);

const result =
  spawnSync(
    "cargo",
    [
      "run",
      "--release",
      "--bin",
      "b12_state_handoff",
      "--",
      snapshot,
      physical,
    ],
    {
      cwd:
        root,

      stdio:
        "inherit",

      shell:
        false,
    },
  );

if (result.error) {
  throw result.error;
}

if (result.status !== 0) {
  throw new Error(
    `B12 state-handoff benchmark failed with exit code ${result.status}`,
  );
}

