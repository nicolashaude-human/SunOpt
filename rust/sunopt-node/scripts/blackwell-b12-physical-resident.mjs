// rust/sunopt-node/scripts/blackwell-b12-physical-resident.mjs
import { existsSync, statSync } from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here =
  path.dirname(
    fileURLToPath(import.meta.url),
  );

const root =
  path.resolve(here, "..");

const source =
  path.join(
    root,
    "research",
    "b12_physical_resident.cu",
  );

const executable =
  path.join(
    root,
    "research",
    "b12_physical_resident.exe",
  );

const reference =
  process.argv[2]
    ? path.resolve(process.argv[2])
    : path.join(
        root,
        "blackwell-physical-reference.bin",
      );

const deviceId =
  Number.parseInt(
    process.argv[3] ?? "0",
    10,
  );

const tolerance =
  Number.parseFloat(
    process.argv[4] ?? "1e-9",
  );

if (!existsSync(source)) {
  throw new Error(
    `B12 CUDA source missing: ${source}`,
  );
}

if (!existsSync(reference)) {
  throw new Error(
    `B12 physical reference missing: ${reference}`,
  );
}

const nvcc =
  process.env.CUDA_PATH
    ? path.join(
        process.env.CUDA_PATH,
        "bin",
        "nvcc.exe",
      )
    : "nvcc.exe";

const needsBuild =
  !existsSync(executable) ||
  statSync(executable).mtimeMs <
    statSync(source).mtimeMs;

if (needsBuild) {
  console.log(`B12_NVCC=${nvcc}`);

  const compile =
    spawnSync(
      nvcc,
      [
        "-O3",
        "-std=c++17",
        "-arch=sm_120f",
        "-Xcompiler",
        "/Zc:preprocessor",
        source,
        "-o",
        executable,
      ],
      {
        cwd: root,
        stdio: "inherit",
        shell: false,
      },
    );

  if (compile.error) {
    throw compile.error;
  }

  if (compile.status !== 0) {
    throw new Error(
      `B12 nvcc failed with exit code ${compile.status}`,
    );
  }
}

console.log(`B12_REFERENCE=${reference}`);
console.log(`B12_REFERENCE_BYTES=${statSync(reference).size}`);
console.log(`B12_DEVICE=${deviceId}`);
console.log(`B12_TOLERANCE=${tolerance}`);

const run =
  spawnSync(
    executable,
    [
      reference,
      String(deviceId),
      String(tolerance),
    ],
    {
      cwd: root,
      stdio: "inherit",
      shell: false,
    },
  );

if (run.error) {
  throw run.error;
}

if (run.status !== 0) {
  throw new Error(
    `B12 resident physics benchmark failed with exit code ${run.status}`,
  );
}

