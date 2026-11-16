// src/components/model-annual-summary-row.tsx

import {
  MetricBox,
} from "@/components/metric-box";

import type {
  ModelComparisonResult,
} from "@/types/model-comparison";

interface ModelAnnualSummaryRowProps {
  model: ModelComparisonResult;
}

function formatEnergy(
  value: number,
): string {
  return `${value.toFixed(1)} kWh`;
}

function formatPercent(
  value: number,
): string {
  return `${value.toFixed(1)} %`;
}

export function ModelAnnualSummaryRow({
  model,
}: ModelAnnualSummaryRowProps) {
  return (
    <section className="rounded-2xl border border-gray-200 bg-white p-4 shadow-sm">
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-lg font-bold text-gray-950">
          {model.name}
        </h3>

        <div className="text-xs font-semibold uppercase tracking-wide text-gray-400">
          Annual summary
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
        <MetricBox
          label="Netzbezug"
          value={formatEnergy(model.annual.gridImport)}
        />

        <MetricBox
          label="Einspeisung"
          value={formatEnergy(model.annual.feedIn)}
        />

        <MetricBox
          label="Eigenverbrauch"
          value={formatEnergy(model.annual.selfConsumption)}
        />

        <MetricBox
          label="PV-Erzeugung"
          value={formatEnergy(model.annual.pvProduction)}
        />

        <MetricBox
          label="Eigenverbrauchsquote"
          value={formatPercent(model.annual.selfConsumptionRate)}
        />

        <MetricBox
        label="Eigenverbrauchsquote"
        value={formatPercent(model.annual.selfConsumptionRate)}
        />
      </div>
    </section>
  );
}

export default ModelAnnualSummaryRow;
