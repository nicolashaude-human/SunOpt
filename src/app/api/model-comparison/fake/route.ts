// src/app/api/model-comparison/fake/route.ts

import {
  NextResponse,
} from "next/server";

import type {
  ComparisonModelId,
  ModelComparisonResponse,
  MonthlyEnergyValues,
} from "@/types/model-comparison";

export const dynamic =
  "force-dynamic";

const PV_PRODUCTION = [
  310,
  475,
  840,
  1120,
  1340,
  1440,
  1515,
  1370,
  1010,
  690,
  355,
  245,
] as const;

interface FakeModelDefinition {
  id: ComparisonModelId;
  name: string;
  gridImport: readonly number[];
  feedIn: readonly number[];
}

const FAKE_MODELS: FakeModelDefinition[] = [
  {
    id: "existing",
    name: "Existing",
    gridImport: [
      104, 86, 62, 31, 18, 12,
      10, 12, 27, 55, 83, 110,
    ],
    feedIn: [
      42, 95, 310, 525, 690, 805,
      870, 760, 495, 255, 85, 31,
    ],
  },
  {
    id: "quantumGravity",
    name: "Quantum Gravity",
    gridImport: [
      93, 75, 51, 25, 15, 10,
      9, 10, 22, 46, 71, 98,
    ],
    feedIn: [
      51, 111, 336, 553, 716, 832,
      895, 785, 520, 277, 98, 39,
    ],
  },
  {
    id: "plasmaWave",
    name: "Plasma Wave",
    gridImport: [
      72, 55, 31, 13, 7, 4,
      3, 4, 10, 27, 49, 79,
    ],
    feedIn: [
      39, 87, 292, 508, 674, 788,
      852, 742, 472, 231, 72, 25,
    ],
  },
];

function sum(
  values: readonly number[],
): number {
  return values.reduce(
    (total, value) =>
      total + value,
    0,
  );
}

function buildMonthly(
  model: FakeModelDefinition,
): MonthlyEnergyValues[] {
  return PV_PRODUCTION.map(
    (pvProduction, index) => ({
      month: index + 1,
      gridImport:
        model.gridImport[index] ?? 0,
      feedIn:
        model.feedIn[index] ?? 0,
      selfConsumption:
        Math.max(
          0,
          pvProduction -
            (model.feedIn[index] ?? 0),
        ),
    }),
  );
}

export async function GET() {
  const pvProduction =
    sum(PV_PRODUCTION);

  const models =
    FAKE_MODELS.map(
      (model) => {
        const monthly =
          buildMonthly(model);

        const gridImport =
          sum(
            monthly.map(
              (point) =>
                point.gridImport,
            ),
          );

        const feedIn =
          sum(
            monthly.map(
              (point) =>
                point.feedIn,
            ),
          );

        const selfConsumption =
          sum(
            monthly.map(
              (point) =>
                point.selfConsumption,
            ),
          );

        return {
          id: model.id,
          name: model.name,
          monthly,
          annual: {
            gridImport,
            feedIn,
            selfConsumption,
            pvProduction,
            selfConsumptionRate:
              selfConsumption /
              pvProduction *
              100,
            
          },
        };
      },
    );

    const response:
    Omit<ModelComparisonResponse, "fake"> & {
      fake: true;
    } = {
  
    fake: true,
  
    year: 2023,
  
    storageVolume: 1000,
  
    simulationDurationMs: 0,
  
    generatedAt:
      new Date().toISOString(),
  
    models,
  };

  return NextResponse.json(
    response,
  );
}
