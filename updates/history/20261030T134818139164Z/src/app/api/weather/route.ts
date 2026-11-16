// src/app/api/weather/route.ts

import { NextRequest, NextResponse } from "next/server";
import { findLocalitiesByPostalCode } from "@/server/localities";
import { getPVGISData } from "@/server/pvgis";
import {
  analyzePVProduction,
  type PVSystem,
} from "@/domain";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const {
      postalCode,
      pvSystem,
      startYear,
      endYear,
    } = body;

    // PLZ prüfen
    if (
      typeof postalCode !== "string" ||
      !/^\d{4}$/.test(postalCode)
    ) {
      return NextResponse.json(
        {
          error: "Ungültige Schweizer Postleitzahl.",
        },
        {
          status: 400,
        },
      );
    }

    // Jahre prüfen
    if (
      !Number.isInteger(startYear) ||
      !Number.isInteger(endYear) ||
      startYear > endYear
    ) {
      return NextResponse.json(
        {
          error: "Ungültiger Zeitraum.",
        },
        {
          status: 400,
        },
      );
    }

    // PV-System prüfen
    if (!isValidPVSystem(pvSystem)) {
      return NextResponse.json(
        {
          error: "Ungültige PV-Systemdaten.",
        },
        {
          status: 400,
        },
      );
    }

    // PLZ -> Location
    const locations =
      await findLocalitiesByPostalCode(postalCode);

    if (locations.length === 0) {
      return NextResponse.json(
        {
          error: "Postleitzahl nicht gefunden.",
        },
        {
          status: 404,
        },
      );
    }

    /*
     * Für den ersten Test nehmen wir den ersten Treffer.
     *
     * Später können wir bei mehreren Treffern
     * eine Auswahl durch den Benutzer verlangen.
     */
    const location = locations[0];

    // Location + PV-System -> PVGIS
    const weatherData = await getPVGISData(
      location,
      pvSystem,
      startYear,
      endYear,
    );

    // src/app/api/weather/route.ts

    const analysis = analyzePVProduction(
    weatherData,
    pvSystem.peakPower,
    );

    return NextResponse.json({
        location,
        pvSystem,
        startYear,
        endYear,
      
        dataPoints: weatherData.length,
      
        analysis,
      
        weatherData,
      });

  } catch (error) {
    console.error(
      "Weather API error:",
      error,
    );

    return NextResponse.json(
      {
        error: "Wetterdaten konnten nicht geladen werden.",
      },
      {
        status: 500,
      },
    );
  }
}


function isValidPVSystem(
  value: unknown,
): value is PVSystem {

  if (
    typeof value !== "object" ||
    value === null
  ) {
    return false;
  }

  const pv = value as Record<string, unknown>;

  return (
    typeof pv.peakPower === "number" &&
    pv.peakPower > 0 &&

    typeof pv.azimuth === "number" &&
    pv.azimuth >= -180 &&
    pv.azimuth <= 180 &&

    typeof pv.inclination === "number" &&
    pv.inclination >= 0 &&
    pv.inclination <= 90 &&

    typeof pv.systemLoss === "number" &&
    pv.systemLoss >= 0 &&
    pv.systemLoss < 1
  );
}