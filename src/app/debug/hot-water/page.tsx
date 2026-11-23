// src/app/debug/hot-water/page.tsx

"use client";

import { KELLY_CHART_COLORS } from "@/styles/chart-palette";

import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import type {
  ReactNode,
} from "react";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  HOT_WATER_TEST_CONFIG,
} from "@/config/config";


/*
 * ================================================================
 * MODEL TYPES
 * ================================================================
 */

type ModelId =
  | "existing"
  | "quantumGravity"
  | "plasmaWave";

type Model2Id =
  | ModelId
  | "na";


const MODEL_NAMES:
  Record<ModelId, string> = {

  existing:
    "Existing",

  quantumGravity:
    "Quantum Gravity",

  plasmaWave:
    "Plasma Wave",
};


/*
 * ================================================================
 * API TYPES
 * ================================================================
 */

interface ModelPoint {
  controlSignal: number;
  heatPumpEnergy: number;
  heatPumpElectricalPower: number;

  waterlineHeatPumpElectricalPower?: number;

  storageEnergy: number;
  pvEnergy: number;
  storageTemperature: number;
  gridImportEnergy: number;
  pvUsedByHeatPumpEnergy: number;
  storageLossEnergy: number;
  unmetThermalDemand: number;
  exportBeforeHpPower: number;
  exportAfterHpPower: number;
  waterLevelPower: number | null;

  userForecastWaterlinePower?: number | null;
  noSunshineWaterlinePower?: number | null;
  effectiveWaterlinePower?: number | null;

  limitingStrategy?:
    | "userForecast"
    | "noSunshine"
    | "equal"
    | null;
}


interface DebugPoint {
  timestamp: string;

  required48hEnergy: number;
  kWhNoSunshine: number;

  userForecastWaterlinePower: number | null;
  noSunshineWaterlinePower: number | null;
  effectiveWaterlinePower: number | null;

  limitingStrategy:
    | "userForecast"
    | "noSunshine"
    | "equal"
    | null;

  existing: ModelPoint;
  quantumGravity: ModelPoint;
  plasmaWave: ModelPoint;
}


interface DebugResponse {
  year: number;
  storageVolume: number;

  simulationDurationMs: number;

  safetyFactor: number;
  timeStepMinutes: number;
  dataPoints: number;
  planningHorizonHours: number;

  userForecastHorizonHours: number;
  noSunshineHorizonHours: number;
  noSunshineProtectedHorizonHours: number;

  data: DebugPoint[];
}


/*
 * ================================================================
 * CHART TYPES
 * ================================================================
 */

interface ChartPoint {

  timestamp:
    string;

  existingControlSignal:
    number;

  quantumGravityControlSignal:
    number;

  plasmaWaveControlSignal:
    number;

  required48hEnergy:
    number;

  model1HpEnergy:
    number;

  model1StorageEnergy:
    number;

  model1PvEnergy:
    number;

  model1StorageTemperature:
    number;

  model1ExportBeforeHpPower:
    number;

  model1ExportAfterHpPower:
    number;

  model1HeatPumpElectricalPower:
    number;

  model1WaterLevelPower:
    number | null;

  model2HpEnergy:
    number | null;

  model2StorageEnergy:
    number | null;

  model2PvEnergy:
    number | null;

  model2StorageTemperature:
    number | null;

  model2ExportBeforeHpPower:
    number | null;

  model2ExportAfterHpPower:
    number | null;

  model2HeatPumpElectricalPower:
    number | null;

  model2WaterLevelPower:
    number | null;

  plasmaWaveWaterLevelPower:
    number | null;

    /*
   * Upper edge of the Plasma-Wave HP load.
   *
   * Waterline + planned HP electrical power.
   */
  plasmaWavePlannedHpTopPower: number | null;
  
  userForecastWaterlinePower:
  number | null;

noSunshineWaterlinePower:
  number | null;

effectiveWaterlinePower:
  number | null;

combinedPlannedHpTopPower:
  number | null;
}


/*
 * ================================================================
 * SUMMARY TYPES
 * ================================================================
 */

interface ModelSummary {

  gridImport:
    number;

  pvUsedByHeatPump:
    number;

  heatPumpConsumption:
    number;

  storageLosses:
    number;

  unmetThermalDemand:
    number;

  minimumTemperature:
    number;

  maximumTemperature:
    number;
}


/*
 * ================================================================
 * COLORS
 * Kenneth L. Kelly maximum-contrast sequence
 * ================================================================
 */

const COLOR_EXISTING =
  KELLY_CHART_COLORS[0];

const COLOR_QUANTUM =
  KELLY_CHART_COLORS[1];

const COLOR_PLASMA =
  KELLY_CHART_COLORS[2];

const COLOR_REQUIRED =
  KELLY_CHART_COLORS[4];

const COLOR_HP =
  KELLY_CHART_COLORS[3];

const COLOR_STORAGE =
  KELLY_CHART_COLORS[7];

const COLOR_PV =
  KELLY_CHART_COLORS[8];

const COLOR_EXPORT_BEFORE =
  KELLY_CHART_COLORS[0];

const COLOR_EXPORT_AFTER =
  KELLY_CHART_COLORS[1];

const COLOR_SCHEDULED_HP =
  KELLY_CHART_COLORS[2];

const COLOR_WATER_LEVEL =
  KELLY_CHART_COLORS[4];

const COLOR_NO_SUNSHINE_WATERLINE =
  KELLY_CHART_COLORS[5];

const COLOR_NO_SUNSHINE_HP_TOP =
  KELLY_CHART_COLORS[11];

const CHART_HEIGHT =
  380;


/*
 * ================================================================
 * PAGE
 * ================================================================
 */

export default function HotWaterDebugPage() {

  /*
   * --------------------------------------------------------------
   * UI STATE
   * --------------------------------------------------------------
   */

  const [
    month,
    setMonth,
  ] =
    useState(
      9,
    );


  const [
    volume,
    setVolume,
  ] =
    useState(
      3000,
    );


  const [
    safetyMarginPercent,
    setSafetyMarginPercent,
  ] =
    useState(
      Math.round(
        (
          HOT_WATER_TEST_CONFIG
            .forecastSafetyFactor -
          1
        ) *
        100,
      ),
    );


  const [
    model1,
    setModel1,
  ] =
    useState<ModelId>(
      "existing",
    );


  const [
    model2,
    setModel2,
  ] =
    useState<Model2Id>(
      "na",
    );


  const [
    result,
    setResult,
  ] =
    useState<DebugResponse | null>(
      null,
    );


  const [
    loading,
    setLoading,
  ] =
    useState(
      false,
    );

    const [
      progress,
      setProgress,
    ] =
      useState(
        0,
      );
    
    
    const [
      expectedDurationMs,
      setExpectedDurationMs,
    ] =
      useState(
        30000,
      );
    
    
    const simulationStartRef =
      useRef<number | null>(
        null,
      );

  const [
    error,
    setError,
  ] =
    useState<string | null>(
      null,
    );

    useEffect(
      () => {
    
        if (!loading) {
          return;
        }
    
        const timer =
          window.setInterval(
            () => {
    
              const start =
                simulationStartRef.current;
    
              if (start === null) {
                return;
              }
    
              const elapsed =
                performance.now() - start;
    
              const estimated =
                Math.max(
                  1000,
                  expectedDurationMs,
                );
    
              const ratio =
                elapsed / estimated;
    
              let nextProgress: number;
    
              if (ratio <= 1) {
                nextProgress =
                  Math.min(
                    90,
                    ratio * 90,
                  );
              } else {
                const overtime =
                  ratio - 1;
    
                nextProgress =
                  Math.min(
                    98,
                    90 + overtime * 4,
                  );
              }
    
              setProgress(
                nextProgress,
              );
            },
            100,
          );
    
        return () => {
          window.clearInterval(
            timer,
          );
        };
      },
      [
        loading,
        expectedDurationMs,
      ],
    );

  /*
   * --------------------------------------------------------------
   * SYNCHRONIZED CHART SCROLL
   * --------------------------------------------------------------
   */

  const chartScrollRefs =
    useRef<
      Array<HTMLDivElement | null>
    >(
      [],
    );


  function registerChartScroller(
    index:
      number,

    element:
      HTMLDivElement | null,
  ): void {

    chartScrollRefs
      .current[index] =
        element;
  }


  function synchronizeScroll(
    sourceIndex:
      number,
  ): void {

    const source =
      chartScrollRefs
        .current[
          sourceIndex
        ];


    if (!source) {
      return;
    }


    const sourceMaximumScroll =
      source.scrollWidth -
      source.clientWidth;


    const scrollRatio =
      sourceMaximumScroll > 0
        ? source.scrollLeft /
          sourceMaximumScroll
        : 0;


    chartScrollRefs
      .current
      .forEach(
        (
          target,
          targetIndex,
        ) => {

          if (
            !target ||
            targetIndex ===
              sourceIndex
          ) {
            return;
          }


          const targetMaximumScroll =
            target.scrollWidth -
            target.clientWidth;


          target.scrollLeft =
            scrollRatio *
            targetMaximumScroll;
        },
      );
  }


  /*
   * --------------------------------------------------------------
   * RUN COMPLETE YEAR
   * --------------------------------------------------------------
   */

  async function runSimulation() {

    setLoading(
      true,
    );
  
    setProgress(
      0,
    );
  
    simulationStartRef.current =
      performance.now();
  
    setError(
      null,
    );

    try {

      const safetyFactor =
        1 +
        Math.max(
          0,
          safetyMarginPercent,
        ) /
        100;


      const response =
        await fetch(
          "/api/debug/hot-water",
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
                2023,
          
                    
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

                  volume,

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
                    safetyFactor,
                },
              }),
          },
        );


      const responseText =
        await response.text();


      let parsed:
        DebugResponse |
        {
          error?: string;
        };


      try {

        parsed =
          JSON.parse(
            responseText,
          ) as
            DebugResponse |
            {
              error?: string;
            };

      } catch {

        throw new Error(
          response.ok
            ? `API returned invalid JSON:\n${responseText}`
            : `API failed with HTTP ${response.status}:\n${responseText}`,
        );
      }


      if (!response.ok) {

        throw new Error(
          "error" in parsed &&
          parsed.error
            ? parsed.error
            : `Simulation failed with HTTP ${response.status}.`,
        );
      }


      setResult(
        parsed as
          DebugResponse,
      );

      const simulationResult =
  parsed as DebugResponse;

setResult(
  simulationResult,
);

setProgress(
  100,
);

if (
  Number.isFinite(
    simulationResult.simulationDurationMs,
  ) &&
  simulationResult.simulationDurationMs > 0
) {
  setExpectedDurationMs(
    simulationResult.simulationDurationMs,
  );
}

    } catch (
      simulationError
    ) {

      console.error(
        "Hot-water simulation failed:",
        simulationError,
      );


      setError(
        simulationError
          instanceof Error
            ? simulationError
                .message
            : "Simulation failed.",
      );

    } finally {

      setLoading(
        false,
      );
    }
  }


  /*
   * --------------------------------------------------------------
   * MONTH FILTER
   *
   * IMPORTANT:
   * No server request here.
   * --------------------------------------------------------------
   */

  const visibleData =
    useMemo(
      () => {

        if (!result) {
          return [];
        }


        return result.data.filter(
          (
            point,
          ) => {

            const date =
              new Date(
                point.timestamp,
              );


            return (
              date.getUTCFullYear() ===
                result.year &&
              date.getUTCMonth() +
                1 ===
                month
            );
          },
        );
      },

      [
        result,
        month,
      ],
    );


  /*
   * --------------------------------------------------------------
   * CHART DATA
   * --------------------------------------------------------------
   */

  const chartData =
  useMemo<ChartPoint[]>(
    () => {

      return visibleData.map(
        (point) => {

          const first =
            point[model1];

          const second =
            model2 === "na"
              ? null
              : point[model2];

          const effectiveWaterline =
            point.effectiveWaterlinePower;

          const scheduledHpPower =
            point.plasmaWave
              .waterlineHeatPumpElectricalPower ??
            0;

          return {
            timestamp:
              point.timestamp,

            plasmaWaveWaterLevelPower:
              effectiveWaterline,

            plasmaWavePlannedHpTopPower:
              effectiveWaterline === null
                ? null
                : effectiveWaterline +
                  scheduledHpPower,

            userForecastWaterlinePower:
              point.userForecastWaterlinePower,

            noSunshineWaterlinePower:
              point.noSunshineWaterlinePower,

            effectiveWaterlinePower:
              effectiveWaterline,

            combinedPlannedHpTopPower:
              effectiveWaterline === null
                ? null
                : effectiveWaterline +
                  scheduledHpPower,

            existingControlSignal:
              point.existing.controlSignal,

            quantumGravityControlSignal:
              point.quantumGravity.controlSignal,

            plasmaWaveControlSignal:
              point.plasmaWave.controlSignal,

            required48hEnergy:
              point.required48hEnergy,

            model1HpEnergy:
              first.heatPumpEnergy,

            model1StorageEnergy:
              first.storageEnergy,

            model1PvEnergy:
              first.pvEnergy,

            model1StorageTemperature:
              first.storageTemperature,

            model1ExportBeforeHpPower:
              first.exportBeforeHpPower,

            model1ExportAfterHpPower:
              first.exportAfterHpPower,

            model1HeatPumpElectricalPower:
              first.heatPumpElectricalPower,

            model1WaterLevelPower:
              first.waterLevelPower,

            model2HeatPumpElectricalPower:
              second?.heatPumpElectricalPower ??
              null,

            model2WaterLevelPower:
              second?.waterLevelPower ??
              null,

            model2HpEnergy:
              second?.heatPumpEnergy ??
              null,

            model2StorageEnergy:
              second?.storageEnergy ??
              null,

            model2PvEnergy:
              second?.pvEnergy ??
              null,

            model2StorageTemperature:
              second?.storageTemperature ??
              null,

            model2ExportBeforeHpPower:
              second?.exportBeforeHpPower ??
              null,

            model2ExportAfterHpPower:
              second?.exportAfterHpPower ??
              null,
          };
        },
      );
    },

    [
      visibleData,
      model1,
      model2,
    ],
  );

  /*
   * --------------------------------------------------------------
   * MONTHLY SUMMARIES
   * --------------------------------------------------------------
   */

  const summaries =
    useMemo(
      () => {

        return {

          existing:
            calculateModelSummary(
              visibleData,
              "existing",
            ),

          quantumGravity:
            calculateModelSummary(
              visibleData,
              "quantumGravity",
            ),

          plasmaWave:
            calculateModelSummary(
              visibleData,
              "plasmaWave",
            ),
        };
      },

      [
        visibleData,
      ],
    );


  /*
   * ==============================================================
   * RENDER
   * ==============================================================
   */

  return (

    <main className="sunopt-page sunopt-year-page min-h-screen px-4 py-8 sm:px-6">

      <div className="mx-auto max-w-7xl">

        <h1 className="mb-2 text-3xl font-bold">
          Jahressimulation
        </h1>


        <p className="mb-8 text-sm opacity-70">
          Die Berechnung umfasst das ganze Jahr. Die Monatsauswahl verändert nur die Anzeige, nicht das Simulationsergebnis.
        </p>


        {/*
         * ========================================================
         * CONTROLS
         * ========================================================
         */}

        <div className="sunopt-control-panel mb-4 flex flex-wrap items-end gap-4">

          <InputBox
            label="Monat"
          >

            <select
              className="sunopt-form-field h-10 w-40"
              value={
                month
              }
              onChange={(
                event,
              ) =>

                setMonth(
                  Number(
                    event.target
                      .value,
                  ),
                )
              }
            >

              {
                Array.from(
                  {
                    length:
                      12,
                  },
                  (
                    _,
                    index,
                  ) =>
                    index + 1,
                ).map(
                  (
                    monthNumber,
                  ) => (

                    <option
                      key={
                        monthNumber
                      }
                      value={
                        monthNumber
                      }
                    >
                      {monthNumber}
                    </option>
                  ),
                )
              }

            </select>

          </InputBox>


          <InputBox
            label="Speichervolumen [L]"
          >

            <input
              className="sunopt-form-field h-10 w-40"
              type="number"
              step="100"
              min="100"
              value={
                volume
              }
              onChange={(
                event,
              ) =>

                setVolume(
                  Number(
                    event.target
                      .value,
                  ),
                )
              }
            />

          </InputBox>


          <InputBox
            label="Sicherheitszuschlag [%]"
          >

            <input
              className="sunopt-form-field h-10 w-40"
              type="number"
              step="5"
              min="0"
              value={
                safetyMarginPercent
              }
              onChange={(
                event,
              ) =>

                setSafetyMarginPercent(
                  Number(
                    event.target
                      .value,
                  ),
                )
              }
            />

          </InputBox>

          <button
            className="sunopt-action-button h-10 px-6"
            onClick={
              runSimulation
            }
            disabled={
              loading
            }
          >

            {
              loading
                ? "Jahressimulation läuft …"
                : "Jahressimulation starten"
            }

          </button>


          <div className="sunopt-value-box flex min-h-10 min-w-[260px] flex-col justify-center gap-1">

  <span className="text-xs opacity-60">
    {
      loading
        ? `Simulation running – ${progress.toFixed(0)} %`
        : result
          ? `Complete – ${(result.simulationDurationMs / 1000)
              .toFixed(1)} s`
          : "Noch nicht berechnet"
    }
  </span>

  <progress
    className="h-3 w-full"
    max={100}
    value={progress}
  />

</div>

        </div>


        {/*
         * ========================================================
         * MODEL SELECTION
         * ========================================================
         */}

        <div className="sunopt-control-panel mb-8 flex flex-wrap gap-4">

          <InputBox
            label="Modell 1"
          >

            <select
              className="sunopt-form-field h-10 min-w-[220px]"
              value={
                model1
              }
              onChange={(
                event,
              ) =>

                setModel1(
                  event.target
                    .value as
                    ModelId,
                )
              }
            >

              <option value="existing">
                Existing
              </option>

              <option value="quantumGravity">
                Quantum Gravity
              </option>

              <option value="plasmaWave">
                Plasma Wave
              </option>

            </select>

          </InputBox>


          <InputBox
            label="Modell 2"
          >

            <select
              className="sunopt-form-field h-10 min-w-[220px]"
              value={
                model2
              }
              onChange={(
                event,
              ) =>

                setModel2(
                  event.target
                    .value as
                    Model2Id,
                )
              }
            >

              <option value="na">
                N/A
              </option>

              <option value="existing">
                Existing
              </option>

              <option value="quantumGravity">
                Quantum Gravity
              </option>

              <option value="plasmaWave">
                Plasma Wave
              </option>

            </select>

          </InputBox>

        </div>


        {
          error && (

            <div className="sunopt-card mb-8 p-4">
              {error}
            </div>
          )
        }


        {
          result && (

            <>

              {/*
               * ==================================================
               * SUMMARY BOXES
               * ==================================================
               */}

              <div className="mb-8 grid gap-6 xl:grid-cols-3">

                <ModelSummaryBox
                  title="Existing"
                  summary={
                    summaries
                      .existing
                  }
                />


                <ModelSummaryBox
                  title="Quantum Gravity"
                  summary={
                    summaries
                      .quantumGravity
                  }
                />


                <ModelSummaryBox
                  title="Plasma Wave"
                  summary={
                    summaries
                      .plasmaWave
                  }
                />

              </div>


              {/*
               * ==================================================
               * DIAGRAM 1
               * ==================================================
               */}

              <ChartBox
                title="Control signal and energy"
                dataPoints={
                  chartData
                }
                scrollIndex={0}
                registerScroller={
                  registerChartScroller
                }
                onScroll={
                  synchronizeScroll
                }
                yAxisLabel="Energy [kWh]"
                axisDataKeys={[
                  "existingControlSignal",
                  "quantumGravityControlSignal",
                  "plasmaWaveControlSignal",
                  "required48hEnergy",
                  "model1HpEnergy",
                  "model1StorageEnergy",
                  "model1PvEnergy",
                  "model2HpEnergy",
                  "model2StorageEnergy",
                  "model2PvEnergy",
                ]}
              >

                <ResponsiveContainer
                  width="100%"
                  height={
                    CHART_HEIGHT
                  }
                >

                  <LineChart
                    data={
                      chartData
                    }
                  >

                    <CartesianGrid />


                    <XAxis
                      dataKey="timestamp"
                      minTickGap={60}
                      tickFormatter={
                        formatTimestamp
                      }
                    />


                    <YAxis hide />


                    <Tooltip
                      labelFormatter={(
                        label,
                      ) =>
                        formatTimestamp(
                          String(
                            label,
                          ),
                        )
                      }
                    />


                    <Legend />


                    {/*
                     * CONTROL SIGNALS
                     */}

                    <Line
                      type="stepAfter"
                      dataKey="existingControlSignal"
                      name="Existing control"
                      dot={false}
                      stroke={
                        COLOR_EXISTING
                      }
                      strokeWidth={2}
                      isAnimationActive={false}
                    />


                    <Line
                      type="stepAfter"
                      dataKey="quantumGravityControlSignal"
                      name="Quantum Gravity control"
                      dot={false}
                      stroke={
                        COLOR_QUANTUM
                      }
                      strokeWidth={2}
                      isAnimationActive={false}
                    />


                    <Line
                      type="stepAfter"
                      dataKey="plasmaWaveControlSignal"
                      name="Plasma Wave control"
                      dot={false}
                      stroke={
                        COLOR_PLASMA
                      }
                      strokeWidth={2}
                      isAnimationActive={false}
                    />


                    <Line
                      type="stepAfter"
                      dataKey="required48hEnergy"
                      name="48 h required energy"
                      dot={false}
                      stroke={
                        COLOR_REQUIRED
                      }
                      strokeWidth={3}
                      isAnimationActive={false}
                    />


                    {/*
                     * MODEL 1
                     */}

                    <Line
                      type="stepAfter"
                      dataKey="model1HpEnergy"
                      name={`${MODEL_NAMES[model1]} – HP energy`}
                      dot={false}
                      stroke={
                        COLOR_HP
                      }
                      strokeWidth={2}
                      isAnimationActive={false}
                    />


                    <Line
                      type="stepAfter"
                      dataKey="model1StorageEnergy"
                      name={`${MODEL_NAMES[model1]} – storage energy`}
                      dot={false}
                      stroke={
                        COLOR_STORAGE
                      }
                      strokeWidth={2}
                      isAnimationActive={false}
                    />


                    <Line
                      type="stepAfter"
                      dataKey="model1PvEnergy"
                      name={`${MODEL_NAMES[model1]} – PV energy`}
                      dot={false}
                      stroke={
                        COLOR_PV
                      }
                      strokeWidth={2}
                      isAnimationActive={false}
                    />


                    {/*
                     * MODEL 2
                     */}

                    {
                      model2 !== "na" && (

                        <>

                          <Line
                            type="stepAfter"
                            dataKey="model2HpEnergy"
                            name={`${MODEL_NAMES[model2]} – HP energy`}
                            dot={false}
                            stroke={
                              COLOR_HP
                            }
                            strokeWidth={2}
                            strokeDasharray="7 4"
                            isAnimationActive={false}
                          />


                          <Line
                            type="stepAfter"
                            dataKey="model2StorageEnergy"
                            name={`${MODEL_NAMES[model2]} – storage energy`}
                            dot={false}
                            stroke={
                              COLOR_STORAGE
                            }
                            strokeWidth={2}
                            strokeDasharray="7 4"
                            isAnimationActive={false}
                          />


                          <Line
                            type="stepAfter"
                            dataKey="model2PvEnergy"
                            name={`${MODEL_NAMES[model2]} – PV energy`}
                            dot={false}
                            stroke={
                              COLOR_PV
                            }
                            strokeWidth={2}
                            strokeDasharray="7 4"
                            isAnimationActive={false}
                          />

                        </>
                      )
                    }

                  </LineChart>

                </ResponsiveContainer>

              </ChartBox>


              {/*
               * ==================================================
               * DIAGRAM 2
               * ==================================================
               */}

              <ChartBox
                title="Storage temperature"
                dataPoints={
                  chartData
                }
                scrollIndex={1}
                registerScroller={
                  registerChartScroller
                }
                onScroll={
                  synchronizeScroll
                }
                yAxisLabel="Temperature [°C]"
                axisDataKeys={[
                  "model1StorageTemperature",
                  "model2StorageTemperature",
                ]}
              >

                <ResponsiveContainer
                  width="100%"
                  height={
                    CHART_HEIGHT
                  }
                >

                  <LineChart
                    data={
                      chartData
                    }
                  >

                    <CartesianGrid />


                    <XAxis
                      dataKey="timestamp"
                      minTickGap={60}
                      tickFormatter={
                        formatTimestamp
                      }
                    />


                    <YAxis hide />


                    <Tooltip
                      labelFormatter={(
                        label,
                      ) =>
                        formatTimestamp(
                          String(
                            label,
                          ),
                        )
                      }
                    />


                    <Legend />


                    <Line
                      type="monotone"
                      dataKey="model1StorageTemperature"
                      name={
                        MODEL_NAMES[
                          model1
                        ]
                      }
                      dot={false}
                      stroke={
                        modelColor(
                          model1,
                        )
                      }
                      strokeWidth={3}
                      isAnimationActive={false}
                    />


                    {
                      model2 !== "na" && (

                        <Line
                          type="monotone"
                          dataKey="model2StorageTemperature"
                          name={
                            MODEL_NAMES[
                              model2
                            ]
                          }
                          dot={false}
                          stroke={
                            modelColor(
                              model2,
                            )
                          }
                          strokeWidth={3}
                          strokeDasharray="7 4"
                          isAnimationActive={false}
                        />
                      )
                    }

                  </LineChart>

                </ResponsiveContainer>

              </ChartBox>



{/*
 * ==================================================
 * DIAGRAM 3
 * ==================================================
 */}

<ChartBox
  title="PV export and HP scheduling"
  dataPoints={chartData}
  scrollIndex={2}
  registerScroller={registerChartScroller}
  onScroll={synchronizeScroll}
  yAxisLabel="Power [kW_el]"
  axisDataKeys={[
    "model1ExportBeforeHpPower",
    "model1ExportAfterHpPower",
    "model1HeatPumpElectricalPower",
    "model2ExportBeforeHpPower",
    "model2ExportAfterHpPower",
    "model2HeatPumpElectricalPower",
    "model2WaterLevelPower",
    "userForecastWaterlinePower",
"noSunshineWaterlinePower",
"effectiveWaterlinePower",
"combinedPlannedHpTopPower",
  ]}
>

  <ResponsiveContainer
    width="100%"
    height={CHART_HEIGHT}
  >

    <LineChart data={chartData}>

      <CartesianGrid />


      <XAxis
        dataKey="timestamp"
        minTickGap={60}
        tickFormatter={formatTimestamp}
      />


      <YAxis hide />


      <Tooltip
        labelFormatter={(label) =>
          formatTimestamp(String(label))
        }
      />


      <Legend />


      {/*
       * ==================================================
       * MODEL 1
       * ==================================================
       */}

      <Line
        type="monotone"
        dataKey="model1ExportBeforeHpPower"
        name={`${MODEL_NAMES[model1]} – PV export before HP`}
        dot={false}
        stroke={COLOR_EXPORT_BEFORE}
        strokeWidth={2}
        isAnimationActive={false}
      />


      <Line
        type="monotone"
        dataKey="model1ExportAfterHpPower"
        name={`${MODEL_NAMES[model1]} – PV export after HP`}
        dot={false}
        stroke={COLOR_EXPORT_AFTER}
        strokeWidth={2}
        isAnimationActive={false}
      />


      <Line
        type="stepAfter"
        dataKey="model1HeatPumpElectricalPower"
        name={`${MODEL_NAMES[model1]} – Heat pump electrical power`}
        dot={false}
        stroke={COLOR_SCHEDULED_HP}
        strokeWidth={3}
        isAnimationActive={false}
      />


      {model1 === "plasmaWave" && (
        <>
          <Line
            type="stepAfter"
            dataKey="plasmaWaveWaterLevelPower"
            name="kWh_user – 48 h Waterline"
            dot={false}
            stroke={COLOR_WATER_LEVEL}
            strokeWidth={3}
            isAnimationActive={false}
          />

          <Line
            type="stepAfter"
            dataKey="plasmaWavePlannedHpTopPower"
            name="kWh_user – HP top"
            dot={false}
            stroke="#00538A"
            strokeWidth={3}
            isAnimationActive={false}
          />
        </>
      )}


      {/*
       * ==================================================
       * MODEL 2
       * ==================================================
       */}

      {model2 !== "na" && (
        <>

          <Line
            type="monotone"
            dataKey="model2ExportBeforeHpPower"
            name={`${MODEL_NAMES[model2]} – PV export before HP`}
            dot={false}
            stroke={COLOR_EXPORT_BEFORE}
            strokeWidth={2}
            strokeDasharray="7 4"
            isAnimationActive={false}
          />


          <Line
            type="monotone"
            dataKey="model2ExportAfterHpPower"
            name={`${MODEL_NAMES[model2]} – PV export after HP`}
            dot={false}
            stroke={COLOR_EXPORT_AFTER}
            strokeWidth={2}
            strokeDasharray="7 4"
            isAnimationActive={false}
          />


          <Line
            type="stepAfter"
            dataKey="model2HeatPumpElectricalPower"
            name={`${MODEL_NAMES[model2]} – Heat pump electrical power`}
            dot={false}
            stroke={COLOR_SCHEDULED_HP}
            strokeWidth={3}
            strokeDasharray="7 4"
            isAnimationActive={false}
          />


          {model2 === "plasmaWave" && (
            <>
              <Line
                type="stepAfter"
                dataKey="plasmaWaveWaterLevelPower"
                name="kWh_user – 48 h Waterline"
                dot={false}
                stroke={COLOR_WATER_LEVEL}
                strokeWidth={3}
                strokeDasharray="7 4"
                isAnimationActive={false}
              />

              <Line
                type="stepAfter"
                dataKey="plasmaWavePlannedHpTopPower"
                name="kWh_user – HP top"
                dot={false}
                stroke="#00538A"
                strokeWidth={3}
                strokeDasharray="7 4"
                isAnimationActive={false}
              />
            </>
          )}

        </>
      )}


      {/*
       * ==================================================
       * kWh_noSunshine – 7 DAY DIAGNOSTIC
       *
       * This is deliberately OUTSIDE Model 1 and Model 2.
       *
       * It is one independent diagnostic schedule and
       * therefore must only be drawn once.
       * ==================================================
       */}

{(
  model1 === "plasmaWave" ||
  model2 === "plasmaWave"
) && (
  <>

    <Line
      type="stepAfter"
      dataKey="userForecastWaterlinePower"
      name="kWh_user – 48 h Waterline"
      dot={false}
      stroke={COLOR_WATER_LEVEL}
      strokeWidth={3}
      isAnimationActive={false}
    />

    <Line
      type="stepAfter"
      dataKey="noSunshineWaterlinePower"
      name="kWh_noSunshine – 7 d Waterline"
      dot={false}
      stroke={COLOR_NO_SUNSHINE_WATERLINE}
      strokeWidth={3}
      strokeDasharray="10 4"
      isAnimationActive={false}
    />

    <Line
      type="stepAfter"
      dataKey="effectiveWaterlinePower"
      name="Combined – effective Waterline"
      dot={false}
      stroke={COLOR_NO_SUNSHINE_HP_TOP}
      strokeWidth={4}
      isAnimationActive={false}
    />

    <Line
      type="stepAfter"
      dataKey="combinedPlannedHpTopPower"
      name="Combined – HP top"
      dot={false}
      stroke="#00538A"
      strokeWidth={3}
      strokeDasharray="4 3"
      isAnimationActive={false}
    />

  </>
)}


    </LineChart>

  </ResponsiveContainer>

</ChartBox>

            </>
          )
        }

      </div>

    </main>
  );
}


/*
 * ================================================================
 * INPUT BOX
 * ================================================================
 */

function InputBox(
  props: {

    label:
      string;

    children:
      ReactNode;
  },
) {

  return (

    <label className="flex flex-col gap-2">

      <span className="h-5 text-sm font-semibold text-[var(--sunopt-muted)]">
        {props.label}
      </span>

      {props.children}

    </label>
  );
}


/*
 * ================================================================
 * MODEL SUMMARY
 * ================================================================
 */

function ModelSummaryBox(
  props: {

    title:
      string;

    summary:
      ModelSummary;
  },
) {

  return (

    <section className="sunopt-card rounded-lg p-4">

      <h2 className="mb-4 text-lg font-semibold">
        {props.title}
      </h2>


      <div className="grid grid-cols-2 gap-3">

        <Summary
          label="Grid import"
          value={`${props.summary.gridImport.toFixed(2)} kWh_el`}
        />


        <Summary
          label="PV → HP"
          value={`${props.summary.pvUsedByHeatPump.toFixed(2)} kWh_el`}
        />


        <Summary
          label="HP consumption"
          value={`${props.summary.heatPumpConsumption.toFixed(2)} kWh_el`}
        />


        <Summary
          label="Storage losses"
          value={`${props.summary.storageLosses.toFixed(2)} kWh_th`}
        />


        <Summary
          label="Unmet DHW"
          value={`${props.summary.unmetThermalDemand.toFixed(2)} kWh_th`}
        />


        <Summary
          label="Min storage"
          value={`${props.summary.minimumTemperature.toFixed(1)} °C`}
        />


        <Summary
          label="Max storage"
          value={`${props.summary.maximumTemperature.toFixed(1)} °C`}
        />

      </div>

    </section>
  );
}


/*
 * ================================================================
 * SUMMARY BOX
 * ================================================================
 */

function Summary(
  props: {

    label:
      string;

    value:
      string;
  },
) {

  return (

    <div className="sunopt-value-box">

      <div className="text-xs opacity-60">
        {props.label}
      </div>


      <div className="mt-1 font-semibold tabular-nums">
        {props.value}
      </div>

    </div>
  );
}


/*
 * ================================================================
 * CALCULATE MONTH SUMMARY
 * ================================================================
 */

function calculateModelSummary(
  data:
    DebugPoint[],

  model:
    ModelId,
): ModelSummary {

  if (
    data.length === 0
  ) {

    return {

      gridImport:
        0,

      pvUsedByHeatPump:
        0,

      heatPumpConsumption:
        0,

      storageLosses:
        0,

      unmetThermalDemand:
        0,

      minimumTemperature:
        0,

      maximumTemperature:
        0,
    };
  }


  let gridImport =
    0;

  let pvUsedByHeatPump =
    0;

  let heatPumpConsumption =
    0;

  let storageLosses =
    0;

  let unmetThermalDemand =
    0;

  let minimumTemperature =
    Number.POSITIVE_INFINITY;

  let maximumTemperature =
    Number.NEGATIVE_INFINITY;


  for (
    const point of data
  ) {

    const current =
      point[
        model
      ];


    gridImport +=
      current
        .gridImportEnergy;


    pvUsedByHeatPump +=
      current
        .pvUsedByHeatPumpEnergy;


    heatPumpConsumption +=
      current
        .heatPumpEnergy;


    storageLosses +=
      current
        .storageLossEnergy;


    unmetThermalDemand +=
      current
        .unmetThermalDemand;


    minimumTemperature =
      Math.min(
        minimumTemperature,

        current
          .storageTemperature,
      );


    maximumTemperature =
      Math.max(
        maximumTemperature,

        current
          .storageTemperature,
      );
  }


  return {

    gridImport,

    pvUsedByHeatPump,

    heatPumpConsumption,

    storageLosses,

    unmetThermalDemand,

    minimumTemperature,

    maximumTemperature,
  };
}


/*
 * ================================================================
 * CHART BOX
 * ================================================================
 */

interface ChartBoxProps {

  title:
    string;

  dataPoints:
    ChartPoint[];

  scrollIndex:
    number;

  registerScroller:
    (
      index:
        number,

      element:
        HTMLDivElement |
        null,
    ) => void;

  onScroll:
    (
      sourceIndex:
        number,
    ) => void;

  axisDataKeys:
    string[];

  yAxisLabel:
    string;

  children:
    ReactNode;
}


function ChartBox(
  props:
    ChartBoxProps,
) {

  const pointsPerDay =
    96;


  const numberOfDays =
    Math.max(
      1,

      props
        .dataPoints
        .length /
        pointsPerDay,
    );


  /*
   * One viewport width = 24 h.
   */
  const chartWidthPercent =
    numberOfDays *
    100;


  return (

    <section className="sunopt-card mb-8 p-4">

      <h2 className="mb-3 text-lg font-semibold">
        {props.title}
      </h2>


      <div
        className="flex w-full"
        style={{
          height:
            CHART_HEIGHT,
        }}
      >

        {/*
         * --------------------------------------------------------
         * FIXED Y AXIS
         * --------------------------------------------------------
         */}

        <div
          className="w-[86px] shrink-0 bg-[var(--sunopt-card)]"
          style={{
            height:
              CHART_HEIGHT,
          }}
        >

          <ResponsiveContainer
            width="100%"
            height={
              CHART_HEIGHT
            }
          >

            <LineChart
              data={
                props.dataPoints
              }
              margin={{
                top:
                  10,

                right:
                  0,

                bottom:
                  20,

                left:
                  0,
              }}
            >

              <XAxis
                dataKey="timestamp"
                height={30}
                tick={false}
                axisLine={false}
              />


              <YAxis
                width={82}
                tick={{
                  fontSize:
                    11,
                }}
                label={{
                  value:
                    props
                      .yAxisLabel,

                  angle:
                    -90,

                  position:
                    "insideLeft",
                }}
              />


              {
                props
                  .axisDataKeys
                  .map(
                    (
                      dataKey,
                    ) => (

                      <Line
                        key={
                          dataKey
                        }
                        type="monotone"
                        dataKey={
                          dataKey
                        }
                        stroke="transparent"
                        dot={false}
                        connectNulls={false}
                        isAnimationActive={false}
                      />
                    ),
                  )
              }

            </LineChart>

          </ResponsiveContainer>

        </div>


        {/*
         * --------------------------------------------------------
         * SCROLL AREA
         * --------------------------------------------------------
         */}

        <div
          ref={(
            element,
          ) => {

            props
              .registerScroller(
                props.scrollIndex,
                element,
              );
          }}
          onScroll={() => {

            props
              .onScroll(
                props.scrollIndex,
              );
          }}
          className="min-w-0 flex-1 overflow-x-auto overflow-y-hidden"
          style={{
            height:
              CHART_HEIGHT,
          }}
        >

          <div
            style={{
              width:
                `${chartWidthPercent}%`,

              minWidth:
                "100%",

              height:
                CHART_HEIGHT,
            }}
          >

            {props.children}

          </div>

        </div>

      </div>

    </section>
  );
}


/*
 * ================================================================
 * MODEL COLOR
 * ================================================================
 */

function modelColor(
  model:
    ModelId,
): string {

  switch (
    model
  ) {

    case "existing":
      return COLOR_EXISTING;

    case "quantumGravity":
      return COLOR_QUANTUM;

    case "plasmaWave":
      return COLOR_PLASMA;
  }
}


/*
 * ================================================================
 * TIMESTAMP
 * ================================================================
 */

function formatTimestamp(
  value:
    string,
): string {

  const date =
    new Date(
      value,
    );


  return date.toLocaleString(
    "de-CH",
    {

      day:
        "2-digit",

      month:
        "2-digit",

      hour:
        "2-digit",

      minute:
        "2-digit",
    },
  );
}