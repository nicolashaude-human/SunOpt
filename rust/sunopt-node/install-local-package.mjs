// rust/sunopt-node/install-local-package.mjs

import { copyFile, mkdir, readFile, rm } from "node:fs/promises";
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(scriptDirectory, "..", "..");
const sourceDirectory = scriptDirectory;
const destinationDirectory = join(projectRoot, "node_modules", "sunopt-node");

const requiredFiles = [
  "index.js",
  "index.d.ts",
  "package.json",
  "sunopt_node.win32-x64-msvc.node",
];

async function sha256(path) {
  const data = await readFile(path);
  return createHash("sha256").update(data).digest("hex");
}

async function install() {
  for (const fileName of requiredFiles) {
    await readFile(join(sourceDirectory, fileName));
  }

  await rm(destinationDirectory, { recursive: true, force: true });
  await mkdir(destinationDirectory, { recursive: true });

  for (const fileName of requiredFiles) {
    await copyFile(
      join(sourceDirectory, fileName),
      join(destinationDirectory, fileName),
    );
  }

  const sourceBinary = join(sourceDirectory, "sunopt_node.win32-x64-msvc.node");
  const installedBinary = join(destinationDirectory, "sunopt_node.win32-x64-msvc.node");
  const [sourceHash, installedHash] = await Promise.all([
    sha256(sourceBinary),
    sha256(installedBinary),
  ]);

  if (sourceHash !== installedHash) {
    throw new Error("Installed native binary hash does not match rust/sunopt-node.");
  }

  const require = createRequire(join(projectRoot, "package.json"));
  const resolved = require.resolve("sunopt-node");
  const native = require("sunopt-node");

  for (const exportName of ["runNoSunshineKernel", "runPeakShavingCausalKernel"]) {
    if (typeof native?.[exportName] !== "function") {
      throw new Error(`sunopt-node is installed but ${exportName} is missing.`);
    }
  }

  console.info("[SunOpt] Local native package installed and verified.", {
    destination: destinationDirectory,
    resolved,
    binarySha256: installedHash,
    target: native.__napiBindingTarget ?? "unknown",
  });
}

install().catch((error) => {
  console.error("[SunOpt] Native package installation failed.", error);
  console.error("Stop the Next.js server before retrying if the .node file is locked.");
  process.exitCode = 1;
});
