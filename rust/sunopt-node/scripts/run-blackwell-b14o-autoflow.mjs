// rust/sunopt-node/scripts/run-blackwell-b14o-autoflow.mjs
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..", "..", "..");
const nativeRoot = path.join(root, "rust", "sunopt-node");
const outbox = path.join(root, "updates", "outbox");

const workflow = path.join(
  nativeRoot,
  "scripts",
  "blackwell-b14o-roundtrip-workflow.mjs"
);

fs.mkdirSync(outbox, { recursive: true });

const failedPrompt = path.join(outbox, "failed-prompt.txt");
const failedBuild = path.join(outbox, "failed-build.txt");
const failedRun = path.join(outbox, "failed-run.txt");

for (const file of [failedPrompt, failedBuild, failedRun]) {
  try { fs.unlinkSync(file); } catch {}
}

function stringifySpawn(result) {
  const error = result.error
    ? [
        `spawnErrorName=${result.error.name ?? ""}`,
        `spawnErrorCode=${result.error.code ?? ""}`,
        `spawnErrorMessage=${result.error.message ?? ""}`,
        `spawnErrorStack=${result.error.stack ?? ""}`,
      ].join("\n")
    : "spawnError=none";

  return [
    `status=${result.status}`,
    `signal=${result.signal ?? ""}`,
    error,
    "",
    "=== STDOUT ===",
    result.stdout ?? "",
    "",
    "=== STDERR ===",
    result.stderr ?? "",
    "",
  ].join("\n");
}

function writeFailure(file, stage, message, result = null) {
  fs.writeFileSync(
    file,
    [
      `stage=${stage}`,
      `time=${new Date().toISOString()}`,
      `error=${message}`,
      result ? stringifySpawn(result) : "",
      "",
    ].join("\n"),
    "utf8"
  );
}

function runBuild() {
  const comspec =
    process.env.ComSpec ||
    process.env.COMSPEC ||
    "C:\\Windows\\System32\\cmd.exe";

  const result = spawnSync(
    comspec,
    ["/d", "/s", "/c", "npm run build"],
    {
      cwd: nativeRoot,
      encoding: "utf8",
      windowsHide: true,
      env: {
        ...process.env,
        SUNOPT_CUDA_FAST_MATH: "0",
      },
    }
  );

  fs.writeFileSync(
    path.join(outbox, "b14o-build.txt"),
    stringifySpawn(result),
    "utf8"
  );

  if (result.error || result.status !== 0) {
    writeFailure(
      failedBuild,
      "build",
      result.error
        ? `build process could not start: ${result.error.message}`
        : `build exited with status ${result.status}`,
      result
    );
    process.exit(2);
  }
}

function runWorkflow() {
  if (!fs.existsSync(workflow)) {
    writeFailure(
      failedPrompt,
      "prompt/package/install",
      `Missing installed B14O workflow: ${workflow}`
    );
    process.exit(1);
  }

  const result = spawnSync(
    process.execPath,
    [workflow],
    {
      cwd: root,
      encoding: "utf8",
      windowsHide: true,
      env: {
        ...process.env,
        SUNOPT_CUDA_FAST_MATH: "0",
      },
    }
  );

  fs.writeFileSync(
    path.join(outbox, "b14o-run.txt"),
    stringifySpawn(result),
    "utf8"
  );

  if (result.error || result.status !== 0) {
    writeFailure(
      failedRun,
      "run",
      result.error
        ? `run process could not start: ${result.error.message}`
        : `run exited with status ${result.status}`,
      result
    );
    process.exit(3);
  }

  process.stdout.write(result.stdout ?? "");
  process.stderr.write(result.stderr ?? "");
}

console.log("B14O_AUTOFLOW_BUILD_START");
runBuild();
console.log("B14O_AUTOFLOW_BUILD_OK");

console.log("B14O_AUTOFLOW_RUN_START");
runWorkflow();
console.log("B14O_AUTOFLOW_RUN_OK");
