// rust/sunopt-node/scripts/blackwell-b14p-child.mjs
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const nativeRoot = path.resolve(here, "..");
const projectRoot = path.resolve(nativeRoot, "..", "..");
const addon = require(
  path.join(
    nativeRoot,
    "sunopt_node.win32-x64-msvc.node"
  )
);

const args = new Map();
for (
  let i = 2;
  i + 1 < process.argv.length;
  i += 2
) {
  args.set(
    process.argv[i],
    process.argv[i + 1]
  );
}

const kind =
  args.get("--kind") ??
  "production-b";

const snapshot =
  args.get("--snapshot") ??
  path.join(
    nativeRoot,
    "blackwell-b12-sync-blackwell.bin"
  );

const reference =
  args.get("--reference") ??
  path.join(
    nativeRoot,
    "benchmarks",
    "references",
    "blackwell-b14m-strict-reference.bin"
  );

function telemetry() {
  try {
    const raw = execFileSync(
      "nvidia-smi",
      [
        "--query-gpu=clocks.sm,temperature.gpu,pstate,power.draw,utilization.gpu",
        "--format=csv,noheader,nounits",
        "-i",
        "0",
      ],
      {
        encoding: "utf8",
        windowsHide: true,
      }
    ).trim();

    const [
      clock,
      temp,
      pstate,
      power,
      util,
    ] =
      raw
        .split(",")
        .map(
          value =>
            value.trim()
        );

    return {
      clockMHz:
        Number(clock),

      temperatureC:
        Number(temp),

      pstate,

      powerW:
        Number(power),

      utilizationPercent:
        Number(util),
    };
  } catch {
    return null;
  }
}

const before =
  telemetry();

let result;

if (kind === "reference") {
  fs.mkdirSync(
    path.dirname(reference),
    { recursive: true }
  );

  result =
    addon
      .runBlackwellB14MPrecision(
        snapshot,
        "strict-reference",
        reference,
        true,
        0
      );
} else if (kind === "b14l") {
  const pinned =
    (
      args.get("--pinned") ??
      "false"
    ) === "true";

  result =
    addon
      .runBlackwellB14LPackedTransport(
        snapshot,
        pinned,
        0
      );
} else {
  result =
    addon
      .runBlackwellB14PProductionReplay(
        snapshot,
        reference,
        0
      );
}

const after =
  telemetry();

process.stdout.write(
  JSON.stringify({
    kind,
    before,
    after,
    result,
  }) + "\n"
);
