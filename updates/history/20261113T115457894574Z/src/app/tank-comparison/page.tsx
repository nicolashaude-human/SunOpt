// src/app/tank-comparison/page.tsx

"use client";

import {
  useMemo,
  useState,
} from "react";

import {
  BarChart,
  type BarChartSeries,
} from "@/components/bar-chart";

import {
  MetricBox,
} from "@/components/metric-box";

import type {
  TankComparisonResponse,
} from "@/types/tank-comparison";

const HEAT_PUMP_POWER_OPTIONS = [
  3,
  5,
  7,
  10,
  12,
  15,
] as const;

const PV_POWER_OPTIONS = [
  5,
  10,
  15,
  20,
  25,
  30,
] as const;

const SINGLE_SERIES:
  BarChartSeries[] = [
  {
    dataKey:
      "value",
    name:
      "Wert",
    color:
      "#803E75",
  },
];

function formatEnergy(
  value: number,
): string {
  return `${value.toFixed(1)} kWh`;
}

function formatMoney(
  value: number,
): string {
  return `${value.toFixed(2)} CHF`;
}

export default function TankComparisonPage() {
  const [
    heatPumpThermalPower,
    setHeatPumpThermalPower,
  ] =
    useState(7);

  const [
    pvPeakPower,
    setPvPeakPower,
  ] =
    useState(10);

  const [
    gridPrice,
    setGridPrice,
  ] =
    useState(0.15);

  const [
    feedInPrice,
    setFeedInPrice,
  ] =
    useState(0.05);

  const [
    result,
    setResult,
  ] =
    useState<
      TankComparisonResponse | null
    >(null);

  const [
    loading,
    setLoading,
  ] =
    useState(false);

  const [
    error,
    setError,
  ] =
    useState<
      string | null
    >(null);

  async function runComparison():
    Promise<void> {
    setLoading(true);
    setError(null);

    try {
      const response =
        await fetch(
          "/api/tank-comparison",
          {
            method:
              "POST",
            headers: {
              "Content-Type":
                "application/json",
            },
            body:
              JSON.stringify({
                heatPumpThermalPower,
                pvPeakPower,
                gridPriceChfPerKWh:
                  gridPrice,
                feedInPriceChfPerKWh:
                  feedInPrice,
              }),
          },
        );

      const text =
        await response.text();

      if (
        !response.ok
      ) {
        throw new Error(
          `Tank comparison failed with HTTP ${response.status}: ${text}`,
        );
      }

      setResult(
        JSON.parse(
          text,
        ) as
          TankComparisonResponse,
      );
    } catch (
      caught
    ) {
      setError(
        caught instanceof Error
          ? caught.message
          : String(caught),
      );
    } finally {
      setLoading(false);
    }
  }

  const bestPoint =
    useMemo(
      () => {
        if (
          !result ||
          result.points.length ===
            0
        ) {
          return null;
        }

        return result.points.reduce(
          (
            best,
            current,
          ) =>
            current.netEnergyCost <
            best.netEnergyCost
              ? current
              : best,
        );
      },
      [
        result,
      ],
    );

function formatDuration(
  value: number,
): string {

  if (
    value >= 1000
  ) {
    return (
      `${value.toFixed(0)} ms ` +
      `(${(value / 1000).toFixed(2)} s)`
    );
  }

  return `${value.toFixed(1)} ms`;
}

// src/app/tank-comparison/page.tsx

function chartData(
  metric:
    | "gridImport"
    | "feedIn"
    | "selfConsumption"
    | "storageLosses"
    | "netEnergyCost"
    | "simulationDurationMs",
):
  Array<
    Record<
      string,
      string | number
    >
  > {

  return (
    result?.points.map(
      (
        point,
      ) => ({

        tank:
          `${point.storageVolume} L`,

        value:
          point[
            metric
          ],
      }),
    ) ??
    []
  );
}

  return (
    <main
      className="
        min-h-screen
        bg-[#ffe1a6]
        px-4 py-8
        sm:px-6
      "
    >
      <div
        className="
          mx-auto max-w-7xl
        "
      >
        <header
          className="mb-6"
        >
          <h1
            className="
              text-3xl font-bold
              tracking-tight
              text-gray-950
            "
          >
            Tankvergleich
          </h1>

          <p
            className="
              mt-3 max-w-4xl
              leading-7
              text-[#675640]
            "
          >
            Vergleich der
            Warmwasserspeicher von
            300 bis 3000 Litern bei
            identischer Wärmepumpe,
            PV-Anlage und
            Tarifannahme.
          </p>
        </header>

        <section
          className="
            mb-6 rounded-2xl
            border border-[#d8c5a5]
            bg-[#ffedca]
            p-5 shadow-sm
          "
        >
          <h2
            className="
              mb-4 text-xl
              font-bold
              text-gray-950
            "
          >
            Randbedingungen
          </h2>

          <div
            className="
              grid gap-4
              sm:grid-cols-2
              lg:grid-cols-4
            "
          >
            <label
              className="
                flex flex-col gap-2
              "
            >
              <span
                className="
                  text-sm font-semibold
                  text-[#675640]
                "
              >
                Wärmepumpe
              </span>

              <select
                value={
                  heatPumpThermalPower
                }
                onChange={
                  (
                    event,
                  ) =>
                    setHeatPumpThermalPower(
                      Number(
                        event.target.value,
                      ),
                    )
                }
                className="
                  rounded-lg
                  border border-[#d8c5a5]
                  bg-[#fffaf0]
                  px-3 py-2.5
                  font-semibold
                "
              >
                {HEAT_PUMP_POWER_OPTIONS.map(
                  (
                    power,
                  ) => (
                    <option
                      key={power}
                      value={power}
                    >
                      {power} kW_th
                    </option>
                  ),
                )}
              </select>
            </label>

            <label
              className="
                flex flex-col gap-2
              "
            >
              <span
                className="
                  text-sm font-semibold
                  text-[#675640]
                "
              >
                PV-Anlage
              </span>

              <select
                value={
                  pvPeakPower
                }
                onChange={
                  (
                    event,
                  ) =>
                    setPvPeakPower(
                      Number(
                        event.target.value,
                      ),
                    )
                }
                className="
                  rounded-lg
                  border border-[#d8c5a5]
                  bg-[#fffaf0]
                  px-3 py-2.5
                  font-semibold
                "
              >
                {PV_POWER_OPTIONS.map(
                  (
                    power,
                  ) => (
                    <option
                      key={power}
                      value={power}
                    >
                      {power} kWp
                    </option>
                  ),
                )}
              </select>
            </label>

            <label
              className="
                flex flex-col gap-2
              "
            >
              <span
                className="
                  text-sm font-semibold
                  text-[#675640]
                "
              >
                Netzbezug
              </span>

              <input
                type="number"
                min="0"
                step="0.01"
                value={gridPrice}
                onChange={
                  (
                    event,
                  ) =>
                    setGridPrice(
                      Number(
                        event.target.value,
                      ),
                    )
                }
                className="
                  rounded-lg
                  border border-[#d8c5a5]
                  bg-[#fffaf0]
                  px-3 py-2.5
                  font-semibold
                "
              />
            </label>

            <label
              className="
                flex flex-col gap-2
              "
            >
              <span
                className="
                  text-sm font-semibold
                  text-[#675640]
                "
              >
                Einspeisung
              </span>

              <input
                type="number"
                min="0"
                step="0.01"
                value={feedInPrice}
                onChange={
                  (
                    event,
                  ) =>
                    setFeedInPrice(
                      Number(
                        event.target.value,
                      ),
                    )
                }
                className="
                  rounded-lg
                  border border-[#d8c5a5]
                  bg-[#fffaf0]
                  px-3 py-2.5
                  font-semibold
                "
              />
            </label>
          </div>

          <div
            className="
              mt-5 flex flex-wrap
              items-center
              justify-between
              gap-4
            "
          >
            <p
              className="
                max-w-4xl text-sm
                leading-6
                text-[#675640]
              "
            >
              Preise in CHF/kWh.
              Nicht-sonnenbedingter
              Wärmepumpenbetrieb wird
              derzeit noch nicht auf
              Niedertarif optimiert.
            </p>

            <button
              type="button"
              onClick={
                runComparison
              }
              disabled={
                loading
              }
              className="
                rounded-lg
                bg-gray-900
                px-4 py-2.5
                text-sm font-semibold
                text-white
                hover:bg-gray-700
                disabled:bg-gray-400
              "
            >
              {loading
                ? "Vergleich läuft …"
                : "Tankvergleich starten"}
            </button>
          </div>
        </section>

        {error && (
          <div
            className="
              mb-6 rounded-xl
              border border-red-300
              bg-red-50
              p-4 text-sm
              text-red-900
            "
          >
            {error}
          </div>
        )}

        {result && (
          <div
            className="space-y-6"
          >
            <section
              className="
                rounded-2xl
                border border-[#d8c5a5]
                bg-[#ffedca]
                p-5 shadow-sm
              "
            >
              <h2
                className="
                  mb-4 text-2xl
                  font-bold
                  text-gray-950
                "
              >
                Vergleich {result.year}
              </h2>

              {bestPoint && (
                <div
                  className="
                    grid gap-3
                    sm:grid-cols-2
                    xl:grid-cols-5
                  "
                >
                  <MetricBox
                    label="Niedrigste Energiekosten"
                    value={
                      `${bestPoint.storageVolume} L`
                    }
                  />

                  <MetricBox
                    label="Netzbezug"
                    value={
                      formatEnergy(
                        bestPoint.gridImport,
                      )
                    }
                  />

                  <MetricBox
                    label="Einspeisung"
                    value={
                      formatEnergy(
                        bestPoint.feedIn,
                      )
                    }
                  />

                  <MetricBox
                    label="Netzkosten"
                    value={
                      formatMoney(
                        bestPoint.gridCost,
                      )
                    }
                  />

                  <MetricBox
                    label="Nettoenergiekosten"
                    value={
                      formatMoney(
                        bestPoint.netEnergyCost,
                      )
                    }
                  />
                </div>
              )}

              <p
                className="
                  mt-4 text-sm
                  leading-6
                  text-[#675640]
                "
              >
                Noch ohne
                Investitionskosten des
                Speichers. Die
                niedrigsten jährlichen
                Energiekosten sind daher
                noch kein vollständiges
                wirtschaftliches Optimum.
              </p>
            </section>

            <BarChart
              title="Netzbezug nach Tankgrösse"
              data={
                chartData(
                  "gridImport",
                )
              }
              series={
                SINGLE_SERIES
              }
              xAxisKey="tank"
              unit=" kWh"
            />

            <BarChart
              title="Einspeisung nach Tankgrösse"
              data={
                chartData(
                  "feedIn",
                )
              }
              series={
                SINGLE_SERIES
              }
              xAxisKey="tank"
              unit=" kWh"
            />

            <BarChart
              title="Eigenverbrauch nach Tankgrösse"
              data={
                chartData(
                  "selfConsumption",
                )
              }
              series={
                SINGLE_SERIES
              }
              xAxisKey="tank"
              unit=" kWh"
            />

            <BarChart
              title="Speicherverluste nach Tankgrösse"
              data={
                chartData(
                  "storageLosses",
                )
              }
              series={
                SINGLE_SERIES
              }
              xAxisKey="tank"
              unit=" kWh"
            />

            <BarChart
              title="Nettoenergiekosten nach Tankgrösse"
              data={
                chartData(
                  "netEnergyCost",
                )
              }
              series={
                SINGLE_SERIES
              }
              xAxisKey="tank"
              unit=" CHF"
            />

{/* src/app/tank-comparison/page.tsx */}

<section
  className="
    rounded-2xl
    border border-[#d8c5a5]
    bg-[#ffedca]
    p-5 shadow-sm
  "
>
  <h2
    className="
      mb-4 text-2xl
      font-bold
      text-gray-950
    "
  >
    Simulationsperformance
  </h2>

  <div
    className="
      grid gap-3
      sm:grid-cols-2
      xl:grid-cols-5
    "
  >
    <MetricBox
      label="Gesamtlaufzeit"
      value={
        formatDuration(
          result.totalDurationMs,
        )
      }
    />

    <MetricBox
      label="Tank-Sweep"
      value={
        formatDuration(
          result.sweepDurationMs,
        )
      }
    />

    <MetricBox
      label="Ø pro Tank"
      value={
        formatDuration(
          result
            .averageSimulationDurationMs,
        )
      }
    />

    <MetricBox
      label="Schnellster Tank"
      value={
        formatDuration(
          result
            .minimumSimulationDurationMs,
        )
      }
    />

    <MetricBox
      label="Langsamster Tank"
      value={
        formatDuration(
          result
            .maximumSimulationDurationMs,
        )
      }
    />
  </div>

  <p
    className="
      mt-4 text-sm
      leading-6
      text-[#675640]
    "
  >
    Historische Datenaufbereitung:{" "}
    {formatDuration(
      result.preparationDurationMs,
    )}
  </p>
</section>


<BarChart
  title="Simulationszeit nach Tankgrösse"
  data={
    chartData(
      "simulationDurationMs",
    )
  }
  series={
    SINGLE_SERIES
  }
  xAxisKey="tank"
  unit=" ms"
/>

          </div>
        )}
      </div>
    </main>
  );
}
