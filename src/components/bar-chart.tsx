// src/components/bar-chart.tsx

"use client";

import {
  Bar,
  BarChart as RechartsBarChart,
  CartesianGrid,
  LabelList,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";


export interface BarChartSeries {
  dataKey:
    string;

  name:
    string;

  color:
    string;
}


interface BarChartProps {
  title:
    string;

  description?:
    string;

  data:
    Array<Record<string, string | number>>;

  xAxisKey:
    string;

  series:
    BarChartSeries[];

  unit?:
    string;

  height?:
    number;
  referenceValue?: number;
}


export function BarChart({
  title,
  description,
  data,
  xAxisKey,
  series,
  unit = "",
  height = 340,
  referenceValue,
}: BarChartProps) {

  return (
    <section
      className="rounded-2xl border border-[var(--sunopt-border)] bg-[var(--sunopt-bg)] p-5 shadow-sm"
    >
      <div
        className="mb-5"
      >
        <h2
          className="text-lg font-bold text-[var(--sunopt-ink)]"
        >
          {title}
        </h2>

        {description && (
          <p
            className="mt-1 text-sm leading-6 text-[var(--sunopt-muted)]"
          >
            {description}
          </p>
        )}
      </div>

      <div
        className="w-full"
        style={{
          height,
        }}
      >
        <ResponsiveContainer
          width="100%"
          height="100%"
        >
          <RechartsBarChart
            data={data}
            margin={{
              top:
                8,
              right:
                10,
              left:
                8,
              bottom:
                0,
            }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              vertical={false}
              stroke="#ded1bd"
            />

            <XAxis
              dataKey={xAxisKey}
              tickLine={false}
            />

            <YAxis
              unit={unit ? ` ${unit}` : undefined}
              width={82}
              tickLine={false}
            />

            <Tooltip
              formatter={(value) => [
                `${Number(value ?? 0).toFixed(1)}${unit ? ` ${unit}` : ""}`,
                "",
              ]}
            />

            <Legend />

            {referenceValue !== undefined && (
              <ReferenceLine
                y={referenceValue}
                stroke="#00538a"
                strokeWidth={2}
                strokeDasharray="6 4"
                ifOverflow="extendDomain"
              />
            )}

            {series.map(
              (
                item,
              ) => (
                <Bar
  key={item.dataKey}
  dataKey={item.dataKey}
  name={item.name}
  fill={item.color}
  radius={[4, 4, 0, 0]}
  minPointSize={1}
>
  <LabelList
    dataKey={item.dataKey}
    position="top"
    fill="#000000"
    fontSize={20}
    fontWeight={700}
    formatter={(value) => Number(value) === 0 ? "0" : ""}
  />
</Bar>
              ),
            )}
          </RechartsBarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}


export default BarChart;
