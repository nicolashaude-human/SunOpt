import fs from "fs/promises";
import path from "path";
import { parse } from "csv-parse/sync";
import AdmZip from "adm-zip";

interface Locality {
  country: "CH";

  postalCode: string;
  additionalDigit: string;

  name: string;
  municipality: string;

  bfsNumber: number;
  canton: string;

  latitude: number;
  longitude: number;
}

interface StacAsset {
  href: string;
  type?: string;
  "proj:epsg"?: number;
}

interface StacItem {
  assets?: Record<string, StacAsset>;
}

interface StacCollection {
  features?: StacItem[];
}

const STAC_URL =
  "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo-vd.ortschaftenverzeichnis_plz/items";

const OUTPUT_PATH = path.join(
  process.cwd(),
  "data",
  "localities",
  "localities.json",
);


// --------------------------------------------------
// Aktuelle CSV-Datei über STAC finden
// --------------------------------------------------

async function findCsvUrl(): Promise<string> {
  const response = await fetch(STAC_URL);

  if (!response.ok) {
    throw new Error(
      `STAC request failed: ${response.status} ${response.statusText}`,
    );
  }

  const data = (await response.json()) as StacCollection;

  if (!data.features) {
    throw new Error("STAC response contains no features.");
  }

  for (const feature of data.features) {
    if (!feature.assets) {
      continue;
    }

    for (const asset of Object.values(feature.assets)) {
      if (
        asset["proj:epsg"] === 4326 &&
        asset.href.toLowerCase().endsWith(".csv.zip")
      ) {
        return asset.href;
      }
    }
  }

  throw new Error("No EPSG:4326 CSV asset found in STAC response.");
}


// --------------------------------------------------
// ZIP herunterladen und CSV extrahieren
// --------------------------------------------------

async function downloadCsv(url: string): Promise<string> {
  console.log(`Downloading:\n${url}`);

  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(
      `CSV download failed: ${response.status} ${response.statusText}`,
    );
  }

  const arrayBuffer = await response.arrayBuffer();

  const zip = new AdmZip(Buffer.from(arrayBuffer));

  const csvEntry = zip
    .getEntries()
    .find(
      (entry) =>
        !entry.isDirectory &&
        entry.entryName.toLowerCase().endsWith(".csv"),
    );

  if (!csvEntry) {
    throw new Error("ZIP archive contains no CSV file.");
  }

  console.log(`CSV in ZIP: ${csvEntry.entryName}`);

  return csvEntry.getData().toString("utf8");
}


// --------------------------------------------------
// Hauptfunktion
// --------------------------------------------------

async function updateLocalities() {
  try {
    console.log("Updating Swiss localities...");

    // 1. Aktuelle CSV-Datei suchen

    const csvUrl = await findCsvUrl();

    console.log("CSV asset found.");


    // 2. CSV herunterladen

    const csv = await downloadCsv(csvUrl);


    // 3. CSV parsen

    const rows = parse(csv, {
      columns: true,
      skip_empty_lines: true,
      delimiter: ";",
      bom: true,
      trim: true,
    }) as Record<string, string>[];

    console.log(`CSV contains ${rows.length} rows.`);


    // 4. CSV -> Locality[]

    const localities: Locality[] = rows
      .map((row): Locality | null => {
        const postalCode = row["PLZ4"];
        const name = row["Ortschaftsname"];

        const longitude = Number(row["E"]);
        const latitude = Number(row["N"]);

        const bfsNumber = Number(row["BFS-Nr"]);

        if (
          !postalCode ||
          !name ||
          !Number.isFinite(latitude) ||
          !Number.isFinite(longitude) ||
          !Number.isFinite(bfsNumber)
        ) {
          console.warn("Invalid locality row:", row);

          return null;
        }

        return {
          country: "CH",

          postalCode,
          additionalDigit: row["Zusatzziffer"],

          name,
          municipality: row["Gemeindename"],

          bfsNumber,
          canton: row["Kantonskürzel"],

          latitude,
          longitude,
        };
      })
      .filter(
        (locality): locality is Locality =>
          locality !== null,
      );


    // 5. Datensatz plausibilisieren

    if (localities.length < 3000) {
      throw new Error(
        `Unexpectedly small locality dataset: ${localities.length} entries.`,
      );
    }

    const invalidCoordinates = localities.filter(
      (locality) =>
        locality.latitude < 45 ||
        locality.latitude > 48 ||
        locality.longitude < 5 ||
        locality.longitude > 11,
    );

    if (invalidCoordinates.length > 0) {
      throw new Error(
        `${invalidCoordinates.length} localities contain implausible coordinates.`,
      );
    }


    // 6. Zielverzeichnis sicherstellen

    await fs.mkdir(
      path.dirname(OUTPUT_PATH),
      {
        recursive: true,
      },
    );


    // 7. Temporäre Datei schreiben

    const tempPath = `${OUTPUT_PATH}.tmp`;

    await fs.writeFile(
      tempPath,
      JSON.stringify(localities, null, 2),
      "utf8",
    );


    // 8. Geschriebene Datei nochmals prüfen

    const testContent = await fs.readFile(
      tempPath,
      "utf8",
    );

    const testData: unknown = JSON.parse(testContent);

    if (!Array.isArray(testData)) {
      throw new Error(
        "Generated JSON is invalid.",
      );
    }

    if (testData.length !== localities.length) {
      throw new Error(
        "Generated JSON contains an unexpected number of entries.",
      );
    }


    // 9. Alte Datei durch neue ersetzen

    try {
      await fs.rename(
        tempPath,
        OUTPUT_PATH,
      );
    } catch {
      await fs.rm(
        OUTPUT_PATH,
        {
          force: true,
        },
      );

      await fs.rename(
        tempPath,
        OUTPUT_PATH,
      );
    }


    // 10. Erfolg

    console.log(
      `Successfully wrote ${localities.length} localities.`,
    );

    console.log(OUTPUT_PATH);

  } catch (error) {
    console.error(
      "Failed to update localities.",
    );

    if (error instanceof Error) {
      console.error(error.message);
    } else {
      console.error(error);
    }

    process.exitCode = 1;
  }
}


updateLocalities();