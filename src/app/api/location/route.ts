import { NextRequest, NextResponse } from "next/server";
import { findLocalitiesByPostalCode } from "@/server/localities";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const postalCode = body.postalCode;

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

    const matches =
      await findLocalitiesByPostalCode(postalCode);

    if (matches.length === 0) {
      return NextResponse.json(
        {
          error: "Postleitzahl nicht gefunden.",
        },
        {
          status: 404,
        },
      );
    }

    return NextResponse.json({
      results: matches,
    });

  } catch (error) {
    console.error(
      "Location API error:",
      error,
    );

    return NextResponse.json(
      {
        error: "Interner Serverfehler.",
      },
      {
        status: 500,
      },
    );
  }
}