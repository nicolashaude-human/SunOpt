// src/components/monthly-energy-bar-chart.tsx

"use client";

import { MODEL_CHART_COLORS as MODEL_COLORS } from "@/styles/chart-palette";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type {
  ModelComparisonResult,
} from "@/types/model-comparison";

type MonthlyMetric =
  | "gridImport"
  | "feedIn"
  | "selfConsumption";

interface MonthlyEnergyBarChartProps {
  title: string;
  description: string;
  metric: MonthlyMetric;
  models: ModelComparisonResult[];
}

const MONTH_LABELS = [
  "Jan",
  "Feb",
  "Mär",
  "Apr",
  "Mai",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Okt",
  "Nov",
  "Dez",
] as const;

function getValue(
  model: ModelComparisonResult | undefined,
  month: number,
  metric: MonthlyMetric,
): number {
  const point =
    model?.monthly.find(
      (entry) => entry.month === month,
    );

  return point?.[metric] ?? 0;
}

export function MonthlyEnergyBarChart({
  title,
  description,
  metric,
  models,
}: MonthlyEnergyBarChartProps) {
  const byId =
    new Map(
      models.map(
        (model) => [
          model.id,
          model,
        ] as const,
      ),
    );

  const data =
    MONTH_LABELS.map(
      (label, index) => {
        const month =
          index + 1;

        return {
          month: label,
          existing: getValue(
            byId.get("existing"),
            month,
            metric,
          ),
          quantumGravity: getValue(
            byId.get("quantumGravity"),
            month,
            metric,
          ),
          plasmaWave: getValue(
            byId.get("plasmaWave"),
            month,
            metric,
          ),
        };
      },
    );

  return (
    <section className="rounded-2xl border border-[var(--sunopt-border-soft)] bg-[var(--sunopt-card)] p-5 shadow-sm">
      <div className="mb-5">
        <h2 className="text-lg font-bold text-[var(--sunopt-ink)]">
          {title}
        </h2>

        <p className="mt-1 text-sm leading-6 text-[var(--sunopt-soft-text)]">
          {description}
        </p>
      </div>

      <div className="h-[340px] w-full">
        <ResponsiveContainer
          width="100%"
          height="100%"
        >
          <BarChart
            data={data}
            margin={{
              top: 8,
              right: 10,
              left: 8,
              bottom: 0,
            }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              vertical={false}
            />

            <XAxis
              dataKey="month"
              tickLine={false}
            />

            <YAxis
              unit=" kWh"
              width={82}
              tickLine={false}
            />

            <Tooltip
              formatter={(value) => [
                `${Number(value ?? 0).toFixed(1)} kWh`,
                "",
              ]}
            />

            <Legend />

            <Bar
              dataKey="existing"
              name="Existing"
              fill={MODEL_COLORS.existing}
              radius={[4, 4, 0, 0]}
            />

            <Bar
              dataKey="quantumGravity"
              name="Quantum Gravity"
              fill={MODEL_COLORS.quantumGravity}
              radius={[4, 4, 0, 0]}
            />

            <Bar
              dataKey="plasmaWave"
              name="Plasma Wave"
              fill={MODEL_COLORS.plasmaWave}
              radius={[4, 4, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

export default MonthlyEnergyBarChart;
