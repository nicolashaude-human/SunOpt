// src/simulation/hot-water-simulation-cache.ts

import {
  createHash,
} from "node:crypto";

import {
  mkdir,
  readFile,
  rename,
  unlink,
  writeFile,
} from "node:fs/promises";

import {
  join,
} from "node:path";

import {
  gzip,
} from "node:zlib";

import {
  promisify,
} from "node:util";


const gzipAsync =
  promisify(
    gzip,
  );


/*
 * Increase this whenever a change to the simulation algorithm
 * makes existing cached results invalid.
 */
export const HOT_WATER_CACHE_VERSION = 1;


const CACHE_DIRECTORY =
  join(
    process.cwd(),
    "data",
    "cache",
    "hot-water",
  );


function stableStringify(
  value: unknown,
): string {

  if (
    value === null ||
    typeof value !== "object"
  ) {
    return JSON.stringify(
      value,
    );
  }


  if (
    Array.isArray(
      value,
    )
  ) {
    return (
      "[" +
      value
        .map(
          stableStringify,
        )
        .join(",") +
      "]"
    );
  }


  const record =
    value as Record<
      string,
      unknown
    >;


  const keys =
    Object.keys(
      record,
    )
      .sort();


  return (
    "{" +
    keys
      .map(
        (key) =>
          JSON.stringify(key) +
          ":" +
          stableStringify(
            record[key],
          ),
      )
      .join(",") +
    "}"
  );
}


export function createHotWaterCacheKey(
  input: unknown,
): string {

  const canonicalInput =
    stableStringify({
      cacheVersion:
        HOT_WATER_CACHE_VERSION,

      input,
    });


  return createHash(
    "sha256",
  )
    .update(
      canonicalInput,
      "utf8",
    )
    .digest(
      "hex",
    );
}


function getCachePath(
  cacheKey: string,
): string {

  return join(
    CACHE_DIRECTORY,
    `${cacheKey}.json.gz`,
  );
}


export async function readHotWaterCache(
  cacheKey: string,
): Promise<Buffer | null> {

  try {

    return await readFile(
      getCachePath(
        cacheKey,
      ),
    );

  } catch (
    error
  ) {

    if (
      error instanceof Error &&
      "code" in error &&
      error.code === "ENOENT"
    ) {
      return null;
    }


    throw error;
  }
}


export async function writeHotWaterCache(
  cacheKey: string,
  value: unknown,
): Promise<Buffer> {

  await mkdir(
    CACHE_DIRECTORY,
    {
      recursive: true,
    },
  );

  const json =
    JSON.stringify(value);

  const compressed =
    await gzipAsync(
      Buffer.from(json, "utf8"),
      {
        level: 9,
      },
    );

  const cachePath =
    getCachePath(cacheKey);

  const temporaryPath =
    `${cachePath}.${process.pid}.tmp`;

  try {

    await writeFile(
      temporaryPath,
      compressed,
    );

    await rename(
      temporaryPath,
      cachePath,
    );

  } catch (error) {

    try {
      await unlink(temporaryPath);
    } catch {
      // Nothing to clean up.
    }

    throw error;
  }

  return compressed;
}