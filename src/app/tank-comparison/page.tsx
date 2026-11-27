// src/app/tank-comparison/page.tsx

"use client";

import { KELLY_CHART_COLORS } from "@/styles/chart-palette";

import {
  useEffect,
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

import {
  TankComparisonModeSwitch,
} from "@/components/tank-comparison-mode-switch";

import type {
  TankComparisonBackendStatus,
  TankComparisonCausalPath,
  TankComparisonRequest,
  TankComparisonResponse,
  TankComparisonThermalReserve,
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
      KELLY_CHART_COLORS[1],
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

function ThermalReserveChart({
  reserve,
  volumes,
}: {
  reserve: TankComparisonThermalReserve;
  volumes: number[];
}) {
  const [annualHeatingInput, setAnnualHeatingInput] = useState("");
  const annualHeatingKWh = Number(annualHeatingInput);
  const heating = reserve.spaceHeating;
  const heatingKWh = Number.isFinite(annualHeatingKWh) && annualHeatingKWh > 0
    ? annualHeatingKWh * (heating?.peakWeekShare ?? 0)
    : 0;
  const totalDemandKWh = reserve.maxHotWaterDemandKWh + heatingKWh;
  const days = reserve.horizonHours / 24;
  const litres = Math.round(totalDemandKWh / reserve.usableCapacityKWhPerLitre);
  const demand = heatingKWh > 0
    ? `${totalDemandKWh.toFixed(1)} kWh thermisch, davon ${heatingKWh.toFixed(1)} für Raumheizung und ${reserve.maxHotWaterDemandKWh.toFixed(1)} für Warmwasser`
    : `${reserve.maxHotWaterDemandKWh.toFixed(1)} kWh thermisch für Warmwasser`;

  return (
    <div className="space-y-3">
      {heating && (
        <div className="rounded-2xl border border-[var(--sunopt-border)] bg-[var(--sunopt-bg)] p-5">
          <label className="block text-sm font-semibold" htmlFor="tank-annual-heating">
            Jahres-Heizwärmebedarf zusätzlich zum Warmwasser (kWh, optional)
          </label>
          <input
            id="tank-annual-heating"
            type="number"
            min="0"
            step="any"
            value={annualHeatingInput}
            onChange={event => setAnnualHeatingInput(event.target.value)}
            className="mt-2 w-full max-w-60 rounded-md border border-[var(--sunopt-border)] px-3 py-2"
          />
          <p className="mt-2 text-sm text-[var(--sunopt-muted)]">
            Das Heizprofil verteilt den Jahresbedarf nach Aussentemperatur.
            Annahmen für die Berechnung sind {heating.indoorTemperatureC} °C Raumtemperatur und
            eine Heizgrenze von {heating.heatingLimitTemperatureC} °C. Die Tanklauf-Ergebnisse
            basieren auf Warmwasser. Der Raumheizbedarf fliesst in die Bedarfsmarkierung ein.
          </p>
        </div>
      )}
      <BarChart
        title="Nutzbare Speicherkapazität und Wärmebedarf"
        description={`Die gestrichelte Linie zeigt den höchsten Bedarf in ${days} aufeinanderfolgenden Tagen: ${demand}. Bei ${reserve.minimumTemperatureC}–${reserve.maximumTemperatureC} °C entspricht das rund ${litres.toLocaleString("de-CH")} l Speichervolumen ohne Speicherverluste.`}
        data={volumes.map(volume => ({
          tank: `${volume} L`,
          capacity: volume * reserve.usableCapacityKWhPerLitre,
        }))}
        xAxisKey="tank"
        series={[{ dataKey: "capacity", name: "Nutzbare Speicherkapazität", color: KELLY_CHART_COLORS[1] }]}
        unit="kWh th."
        referenceValue={totalDemandKWh}
      />
    </div>
  );
}

export default function TankComparisonPage() {
  const [backend, setBackend] = useState<TankComparisonBackendStatus | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/tank-comparison", { signal: controller.signal })
      .then(response => response.ok ? response.json() : null)
      .then((status: TankComparisonBackendStatus | null) => {
        if (!controller.signal.aborted) setBackend(status);
      })
      .catch(() => {
        // The POST endpoint can still choose a backend if status is unavailable.
      });
    return () => controller.abort();
  }, []);
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
    causalPath,
    setCausalPath,
  ] =
    useState<
      TankComparisonCausalPath
    >(
      "B",
    );

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
              JSON.stringify(
                {
                  heatPumpThermalPower,
                  pvPeakPower,
                  gridPriceChfPerKWh:
                    gridPrice,
                  feedInPriceChfPerKWh:
                    feedInPrice,

                  causalPath,
                } satisfies
                  TankComparisonRequest,
              ),
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

      const payload =
        JSON.parse(
          text,
        ) as
          TankComparisonResponse;


      setResult(
        payload,
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
        bg-[var(--sunopt-page)]
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
            mb-6 grid
            grid-cols-[minmax(0,1fr)_auto]
            items-start gap-4
          "
        >
          <div>
            <h1
              className="
                text-3xl font-bold
                tracking-tight
                text-[var(--sunopt-ink)]
              "
            >
              Tankvergleich
            </h1>

            <p
              className="
                mt-3 max-w-4xl
                leading-7
                text-[var(--sunopt-muted)]
              "
            >
              Vergleich der
              Warmwasserspeicher von
              300 bis 3000 Litern bei
              identischer Wärmepumpe,
              PV-Anlage und
              Tarifannahme.
            </p>
          </div>

          {backend?.backend === "cpu" ? (
            <span className="rounded-full border border-[var(--sunopt-border)] px-3 py-2 text-sm font-bold">
              CPU · {backend.workerCount} Worker
            </span>
          ) : (
            <TankComparisonModeSwitch value={causalPath} disabled={loading} onChange={setCausalPath} />
          )}
        </header>

        <section
          className="
            mb-6 rounded-2xl
            border border-[var(--sunopt-border-soft)]
            bg-[var(--sunopt-panel)]
            p-5 shadow-sm
          "
        >
          <h2
            className="
              mb-4 text-xl
              font-bold
              text-[var(--sunopt-ink)]
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
                  text-[var(--sunopt-muted)]
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
                  border border-[var(--sunopt-border-soft)]
                  bg-[var(--sunopt-bg)]
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
                  text-[var(--sunopt-muted)]
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
                  border border-[var(--sunopt-border-soft)]
                  bg-[var(--sunopt-bg)]
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
                  text-[var(--sunopt-muted)]
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
                  border border-[var(--sunopt-border-soft)]
                  bg-[var(--sunopt-bg)]
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
                  text-[var(--sunopt-muted)]
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
                  border border-[var(--sunopt-border-soft)]
                  bg-[var(--sunopt-bg)]
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
                text-[var(--sunopt-muted)]
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
                bg-[var(--sunopt-ink)]
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
                border border-[var(--sunopt-border-soft)]
                bg-[var(--sunopt-panel)]
                p-5 shadow-sm
              "
            >
              <h2
                className="
                  mb-4 text-2xl
                  font-bold
                  text-[var(--sunopt-ink)]
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
                  text-[var(--sunopt-muted)]
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

            {result.thermalReserve && (
              <ThermalReserveChart
                reserve={result.thermalReserve}
                volumes={result.points.map(point => point.storageVolume)}
              />
            )}

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
              title="PV-Netzeinspeisung nach Tankgrösse"
              description="PV-Strom nach Abzug der modellierten Haushaltslast und des tatsächlichen Wärmepumpenverbrauchs."
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
              title="PV-Eigenverbrauch nach Tankgrösse"
              description="PV-Strom für die modellierte Haushaltslast und die Wärmepumpe."
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
    border border-[var(--sunopt-border-soft)]
    bg-[var(--sunopt-panel)]
    p-5 shadow-sm
  "
>
  <div
    className="
      mb-4 flex flex-wrap
      items-center justify-between
      gap-3
    "
  >
    <h2
      className="
        text-2xl font-bold
        text-[var(--sunopt-ink)]
      "
    >
      Simulationsperformance
    </h2>

    <div
      className="
        rounded-full border border-green-500
        bg-green-50 px-3 py-1
        text-xs font-bold
        text-green-900
      "
    >
      {result.backend === "cpu" ? `CPU · ${result.workerCount} Worker` : `CUDA · Pfad ${result.causalPath}`}
    </div>
  </div>

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
      text-[var(--sunopt-muted)]
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
