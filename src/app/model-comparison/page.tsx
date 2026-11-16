// src/app/model-comparison/page.tsx

"use client";

import {
  useState,
} from "react";

import {
  BarChart,
  type BarChartSeries,
} from "@/components/bar-chart";

import {
  MetricBox,
} from "@/components/metric-box";

import {
  MetricBoxRowContainer,
} from "@/components/metric-box-row-container";

import {
  HOT_WATER_TEST_CONFIG,
} from "@/config/config";

import type {
  ComparisonModelId,
  ModelComparisonResponse,
} from "@/types/model-comparison";

const YEAR =
  2023;

const STORAGE_VOLUME =
  1000;

const MODEL_COLORS:
  Record<
    ComparisonModelId,
    string
  > = {
  existing:
    "#FFB300",
  quantumGravity:
    "#803E75",
  plasmaWave:
    "#FF6800",
};

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

const CHART_SERIES:
  BarChartSeries[] = [
  {
    dataKey:
      "existing",
    name:
      "Existing",
    color:
      MODEL_COLORS
        .existing,
  },
  {
    dataKey:
      "quantumGravity",
    name:
      "Quantum Gravity",
    color:
      MODEL_COLORS
        .quantumGravity,
  },
  {
    dataKey:
      "plasmaWave",
    name:
      "Plasma Wave",
    color:
      MODEL_COLORS
        .plasmaWave,
  },
];

function formatEnergy(
  value: number,
): string {
  return (
    `${value.toFixed(1)} kWh`
  );
}

function formatPercent(
  value: number,
): string {
  return (
    `${value.toFixed(1)} %`
  );
}

function errorToString(
  error: unknown,
): string {
  return error instanceof Error
    ? error.message
    : String(error);
}

export default function ModelComparisonPage() {
  const [
    result,
    setResult,
  ] =
    useState<
      ModelComparisonResponse | null
    >(
      null,
    );

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
    >(
      null,
    );

  async function runAnnualComparison():
    Promise<void> {
    setLoading(true);
    setError(null);

    try {
      const response =
        await fetch(
          "/api/model-comparison/annual",
          {
            method:
              "POST",
            headers: {
              "Content-Type":
                "application/json",
            },
            body:
              JSON.stringify({
                postalCode:
                  "8001",
                year:
                  YEAR,
                pvSystem: {
                  peakPower:
                    10,
                  azimuth:
                    0,
                  inclination:
                    30,
                  systemLoss:
                    0.14,
                },
                hotWaterDemand: {
                  persons:
                    2,
                  dailyConsumptionPerPerson:
                    50,
                  coldWaterTemperature:
                    10,
                  deliveryTemperature:
                    60,
                },
                heatPump: {
                  thermalPower:
                    5,
                  maxFlowTemperature:
                    75,
                  copModel: {
                    type:
                      "constant",
                    cop:
                      3,
                  },
                },
                storage: {
                  volume:
                    STORAGE_VOLUME,
                  minTemperature:
                    60,
                  maxTemperature:
                    75,
                  initialTemperature:
                    60,
                  insulation: {
                    thermalConductivity:
                      0.030,
                    thickness:
                      0.10,
                    cylinderHeight:
                      1.70,
                    ambientTemperature:
                      20,
                  },
                },
                control: {
                  minimumReserveHours:
                    2,
                  forecastHorizonHours:
                    72,
                  pvTargetTemperature:
                    75,
                  householdReservePower:
                    HOT_WATER_TEST_CONFIG
                      .householdReservePower,
                  forecastSafetyFactor:
                    HOT_WATER_TEST_CONFIG
                      .forecastSafetyFactor,
                },
              }),
          },
        );

      const text =
        await response.text();

      if (
        !response.ok
      ) {
        throw new Error(
          `Annual comparison failed with HTTP ${response.status}: ${text}`,
        );
      }

      setResult(
        JSON.parse(
          text,
        ) as
          ModelComparisonResponse,
      );
    } catch (caught) {
      setError(
        errorToString(
          caught,
        ),
      );
    } finally {
      setLoading(false);
    }
  }

  function buildMetricChartData(
    metric:
      "gridImport" |
      "feedIn" |
      "selfConsumption",
  ):
    Array<
      Record<
        string,
        string | number
      >
    > {
    return MONTH_LABELS.map(
      (
        monthLabel,
        index,
      ) => {
        const month =
          index + 1;

        const row:
          Record<
            string,
            string | number
          > = {
          month:
            monthLabel,
        };

        for (
          const model of
          result?.models ??
          []
        ) {
          row[
            model.id
          ] =
            model.monthly.find(
              (
                point,
              ) =>
                point.month ===
                month,
            )?.[
              metric
            ] ??
            0;
        }

        return row;
      },
    );
  }

 // #ffe1a6
 // #ffedca
//  #ffc144

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
          className="
            mb-6 flex flex-wrap
            items-start
            justify-between
            gap-4
          "
        >
          <div>
            <h1
              className="
                text-3xl font-bold
                tracking-tight
                text-gray-950
              "
            >
              Modellvergleich
            </h1>

            <p
              className="
                mt-3 max-w-3xl
                leading-7
                text-[#ffc144]
              "
            >
              Jahresvergleich der
              Regelmodelle über
              Netzbezug, Einspeisung
              und Eigenverbrauch.
            </p>
          </div>

          <button
            type="button"
            onClick={
              runAnnualComparison
            }
            disabled={
              loading
            }
            className="
              rounded-lg
              bg-gray-900
              px-4 py-2.5
              text-sm font-semibold
              text-white shadow-sm
              hover:bg-gray-700
              disabled:cursor-not-allowed
              disabled:bg-gray-400
            "
          >
            {loading
              ? "Jahresrechnung läuft …"
              : "Jahresrechnung starten"}
          </button>
        </header>

        {error && (
          <div
            className="
              mb-6 whitespace-pre-wrap
              rounded-xl
              border border-red-300
              bg-red-50
              p-4 text-sm
              text-red-900
            "
          >
            {error}
          </div>
        )}

        {!result &&
          !loading &&
          !error && (
            <div
              className="
                rounded-2xl
                border border-[#d8c5a5]
                bg-[#ffedca]
                p-6
                text-[#ffc144]
                shadow-sm
              "
            >
              Die Jahresrechnung wird
              erst nach Klick auf
              „Jahresrechnung starten“
              ausgeführt.
            </div>
          )}

        {result && (
          <div
            className="
              space-y-6
            "
          >
            <section
              className="
                rounded-2xl
                border border-[#d8c5a5]
                bg-[#ffedca]
                p-5 shadow-sm
              "
            >
              <div
                className="
                  mb-4 flex flex-wrap
                  items-stretch
                  justify-between
                  gap-4
                "
              >
                <div
                  className="
                    flex items-center
                  "
                >
                  <h2
                    className="
                      text-3xl font-bold
                      tracking-tight
                      text-gray-950
                    "
                  >
                    Jahreszusammenfassung
                    {" "}
                    {result.year}
                  </h2>
                </div>

                <MetricBox
                  label="Speicher"
                  value={
                    `${result.storageVolume} L`
                  }
                  size="large"
                />
              </div>

              <div
                className="
                  space-y-3
                "
              >
                {result.models.map(
                  (
                    model,
                  ) => (
                    <MetricBoxRowContainer
                      key={
                        model.id
                      }
                      title={
                        model.name
                      }
                    >
                      <MetricBox
                        label="Netzbezug"
                        value={
                          formatEnergy(
                            model
                              .annual
                              .gridImport,
                          )
                        }
                      />

                      <MetricBox
                        label="Einspeisung"
                        value={
                          formatEnergy(
                            model
                              .annual
                              .feedIn,
                          )
                        }
                      />

                      <MetricBox
                        label="Eigenverbrauch"
                        value={
                          formatEnergy(
                            model
                              .annual
                              .selfConsumption,
                          )
                        }
                      />

                      <MetricBox
                        label="PV-Erzeugung"
                        value={
                          formatEnergy(
                            model
                              .annual
                              .pvProduction,
                          )
                        }
                      />

                      <MetricBox
                        label="Eigenverbrauchsquote"
                        value={
                          formatPercent(
                            model
                              .annual
                              .selfConsumptionRate,
                          )
                        }
                      />
                    </MetricBoxRowContainer>
                  ),
                )}
              </div>
            </section>

            <BarChart
              title="Netzbezug pro Monat"
              data={
                buildMetricChartData(
                  "gridImport",
                )
              }
              series={
                CHART_SERIES
              }
              xAxisKey="month"
              unit=" kWh"
            />

            <BarChart
              title="Einspeisung pro Monat"
              data={
                buildMetricChartData(
                  "feedIn",
                )
              }
              series={
                CHART_SERIES
              }
              xAxisKey="month"
              unit=" kWh"
            />

            <BarChart
              title="Eigenverbrauch pro Monat"
              data={
                buildMetricChartData(
                  "selfConsumption",
                )
              }
              series={
                CHART_SERIES
              }
              xAxisKey="month"
              unit=" kWh"
            />
          </div>
        )}
      </div>
    </main>
  );
}
