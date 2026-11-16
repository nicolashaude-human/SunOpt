// rust/sunopt-node/scripts/blackwell-b14p-telemetry.mjs
import fs from "node:fs";
import { execFileSync } from "node:child_process";

const args = new Map();
for (let i = 2; i + 1 < process.argv.length; i += 2) {
  args.set(process.argv[i], process.argv[i + 1]);
}

const output =
  args.get("--output");

const stopFile =
  args.get("--stop-file");

const parentPid =
  Number.parseInt(
    args.get("--parent-pid") ?? "0",
    10
  );

const intervalMs =
  Number.parseInt(
    args.get("--interval-ms") ?? "200",
    10
  );

if (!output || !stopFile || !Number.isFinite(parentPid) || parentPid <= 0) {
  throw new Error("B14P telemetry arguments are invalid.");
}

const errorFile =
  output + ".errors.txt";

function sleepMs(ms) {
  const wait =
    new Int32Array(
      new SharedArrayBuffer(4)
    );

  Atomics.wait(
    wait,
    0,
    0,
    ms
  );
}

function parentAlive() {
  try {
    process.kill(parentPid, 0);
    return true;
  } catch {
    return false;
  }
}

function query() {
  return execFileSync(
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
      timeout: 5000,
    }
  ).trim();
}

try {
  fs.writeFileSync(output, "", "utf8");
  try {
    fs.unlinkSync(errorFile);
  } catch {}

  while (
    parentAlive() &&
    !fs.existsSync(stopFile)
  ) {
    const started =
      Date.now();

    try {
      const line =
        query();

      if (line) {
        fs.appendFileSync(
          output,
          line + "\n",
          "utf8"
        );
      }
    } catch (error) {
      fs.appendFileSync(
        errorFile,
        [
          `time=${new Date().toISOString()}`,
          `error=${error?.stack ?? error}`,
          "",
        ].join("\n"),
        "utf8"
      );
    }

    const elapsed =
      Date.now() -
      started;

    sleepMs(
      Math.max(
        10,
        intervalMs -
        elapsed
      )
    );
  }
} catch (error) {
  try {
    fs.appendFileSync(
      errorFile,
      [
        `fatalTime=${new Date().toISOString()}`,
        `fatalError=${error?.stack ?? error}`,
        "",
      ].join("\n"),
      "utf8"
    );
  } catch {}

  process.exitCode = 1;
}
