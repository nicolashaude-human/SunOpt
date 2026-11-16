import fs from "fs/promises";
import path from "path";
import type { Location } from "@/domain";

const LOCALITIES_PATH = path.join(
  process.cwd(),
  "data",
  "localities",
  "localities.json",
);

let localityIndex: Map<string, Location[]> | null = null;

async function buildLocalityIndex(): Promise<Map<string, Location[]>> {
  const file = await fs.readFile(
    LOCALITIES_PATH,
    "utf8",
  );

  const localities = JSON.parse(file) as Location[];

  const index = new Map<string, Location[]>();

  for (const locality of localities) {
    const existing = index.get(locality.postalCode);

    if (existing) {
      existing.push(locality);
    } else {
      index.set(
        locality.postalCode,
        [locality],
      );
    }
  }

  console.log(
    `Loaded ${localities.length} localities with ${index.size} postal codes.`,
  );

  return index;
}

async function getLocalityIndex(): Promise<Map<string, Location[]>> {
  if (localityIndex === null) {
    localityIndex = await buildLocalityIndex();
  }

  return localityIndex;
}

export async function findLocalitiesByPostalCode(
  postalCode: string,
): Promise<Location[]> {
  const index = await getLocalityIndex();

  return index.get(postalCode) ?? [];
}