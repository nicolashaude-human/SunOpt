// src/app/api/debug/hot-water/route.ts

import {
  NextRequest,
  NextResponse,
} from "next/server";

import {
  calculateAnnualHotWaterDemand,
  calculateStorageTemperature,
  DEFAULT_HOT_WATER_PROFILE,

  type HeatPump,
  type HotWaterControl,
  type HotWaterDemand,
  type PVSystem,
  type ThermalStorage,
  type WeatherData,
} from "@/domain";

import {
  resampleToQuarterHour,
} from "@/simulation/resample-to-quarter-hour";

import {
  simulateHotWaterSystem,
} from "@/simulation/hot-water-simulation";

import {
  calculateRollingHotWaterDemandForecast,
} from "@/simulation/hot-water-demand-forecast";

import {
  createQuantumGravityControlModel,
} from "@/simulation/control-model-quantum-gravity";

import {
  createPlasmaWaveControlModel,
  type PlasmaWaveControlDecision,
} from "@/simulation/control-model-plasma-wave";

import {
  CONFIG,
  HOT_WATER_TEST_CONFIG,
} from "@/config/config";

import {
  createHotWaterCacheKey,
  readHotWaterCache,
  writeHotWaterCache,
} from "@/simulation/hot-water-simulation-cache";


export const runtime = "nodejs";


interface WeatherApiResponse {
  weatherData: WeatherData;
}


interface RequestBody {
  postalCode: string;
  year: number;

  pvSystem: PVSystem;
  hotWaterDemand: HotWaterDemand;
  heatPump: HeatPump;
  storage: ThermalStorage;
  control: HotWaterControl;
}


const TIME_STEP_HOURS =
  HOT_WATER_TEST_CONFIG.timeStepHours;


/*
 * ================================================================
 * API
 * ================================================================
 */
export async function POST(
  request: NextRequest,
): Promise<NextResponse> {

  try {

    const body = (await request.json()) as RequestBody;


    const cacheKey =
      createHotWaterCacheKey({
        request: body,

        simulation: {
          timeStepHours:
            TIME_STEP_HOURS,

          plasmaWavePlanningHorizonHours:
            CONFIG.control
              .plasmaWavePlanningHorizonHours,

          noSunshineHorizonHours:
            CONFIG.control
              .noSunshineHorizonHours,

          noSunshineProtectedHorizonHours:
            CONFIG.control
              .noSunshineProtectedHorizonHours,

          noSunshineInitialWaterlinePower:
            CONFIG.control
              .noSunshineInitialWaterlinePower,

          pvTargetTemperature:
            CONFIG.control
              .pvTargetTemperature,
        },
      });


    const cachedResponse =
      await readHotWaterCache(
        cacheKey,
      );


    if (cachedResponse !== null) {
      return new NextResponse(
        new Uint8Array(cachedResponse),
        {
          status: 200,

          headers: {
            "Content-Type":
              "application/json; charset=utf-8",

            "Content-Encoding":
              "gzip",

            "Cache-Control":
              "no-store",

            "X-SunOpt-Cache":
              "HIT",

            "X-SunOpt-Cache-Key":
              cacheKey,
          },
        },
      );
    }


    const simulationStartTime =
      performance.now();
    /*
     * ============================================================
     * WEATHER – COMPLETE YEAR
     * ============================================================
     */
    const weatherResponse = await fetch(
      `${request.nextUrl.origin}/api/weather`,
      {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
        },

        body: JSON.stringify({
          postalCode: body.postalCode,
          pvSystem: body.pvSystem,
          startYear: body.year,
          endYear: body.year,
        }),

        cache: "no-store",
      },
    );


    if (!weatherResponse.ok) {
      const errorText = await weatherResponse.text();

      throw new Error(
        `Weather API failed: ${errorText}`,
      );
    }


    const weatherResult =
      (await weatherResponse.json()) as WeatherApiResponse;


    /*
     * ============================================================
     * HOT-WATER DEMAND
     * ============================================================
     */
    const hourlyHotWaterData =
      calculateAnnualHotWaterDemand(
        body.hotWaterDemand,
        DEFAULT_HOT_WATER_PROFILE,
        weatherResult.weatherData,
      );


    /*
     * ============================================================
     * COMMON 15-MINUTE DATA
     * ============================================================
     */
    const quarterHour =
      resampleToQuarterHour(
        weatherResult.weatherData,
        hourlyHotWaterData,
      );


    /*
     * ============================================================
     * QUANTUM-GRAVITY PHYSICAL HORIZON
     * ============================================================
     */
    const rollingDemandForecast =
      calculateRollingHotWaterDemandForecast({
        hotWaterData: quarterHour.hotWaterData,
        storage: body.storage,
        timeStepHours: TIME_STEP_HOURS,

        targetTemperature:
          body.control.pvTargetTemperature,

        minimumTemperature:
          body.storage.minTemperature,

        safetyFactor:
          body.control.forecastSafetyFactor,

        maximumForecastHours:
          CONFIG.control.noSunshineHorizonHours,
      });


    /*
     * ============================================================
     * 1. EXISTING CONTROLLER
     * ============================================================
     */
    const existingSimulation =
      simulateHotWaterSystem({
        weatherData: quarterHour.weatherData,
        hotWaterData: quarterHour.hotWaterData,
        heatPump: body.heatPump,
        storage: body.storage,
        control: body.control,
        timeStepHours: TIME_STEP_HOURS,
      });


    /*
     * ============================================================
     * 2. QUANTUM GRAVITY
     * ============================================================
     */
    const quantumGravityControlModel =
      createQuantumGravityControlModel({
        weatherData: quarterHour.weatherData,
        hotWaterData: quarterHour.hotWaterData,
        heatPump: body.heatPump,
        storage: body.storage,
        control: body.control,
        timeStepHours: TIME_STEP_HOURS,

        planningHorizonHours:
          rollingDemandForecast.planningHorizonHours,
      });


    const quantumGravitySimulation =
      simulateHotWaterSystem({
        weatherData: quarterHour.weatherData,
        hotWaterData: quarterHour.hotWaterData,
        heatPump: body.heatPump,
        storage: body.storage,
        control: body.control,
        timeStepHours: TIME_STEP_HOURS,
        controlModel: quantumGravityControlModel,
      });


    /*
     * ============================================================
     * 3. COMBINED WATERLINE CONTROLLER
     * ============================================================
     *
     * The controller calculates:
     *
     * - kWh_user 48-hour Waterline
     * - kWh_noSunshine seven-day Waterline
     * - effective Waterline = min(48 h, 7 d)
     *
     * The callback records the values calculated by the actual
     * controller. Nothing is recalculated for the debug output.
     */
    const plasmaWaveDecisions:
      Array<PlasmaWaveControlDecision | undefined> = [];


    const plasmaWaveControlModel =
      createPlasmaWaveControlModel({
        weatherData: quarterHour.weatherData,
        hotWaterData: quarterHour.hotWaterData,
        heatPump: body.heatPump,
        storage: body.storage,
        control: body.control,
        timeStepHours: TIME_STEP_HOURS,

        planningHorizonHours:
          CONFIG.control.plasmaWavePlanningHorizonHours,

        targetTemperature:
          CONFIG.control.pvTargetTemperature,

        onDecision: (decision) => {
          plasmaWaveDecisions[decision.currentIndex] =
            decision;
        },
      });


    const plasmaWaveSimulation =
      simulateHotWaterSystem({
        weatherData: quarterHour.weatherData,
        hotWaterData: quarterHour.hotWaterData,
        heatPump: body.heatPump,
        storage: body.storage,
        control: body.control,
        timeStepHours: TIME_STEP_HOURS,
        controlModel: plasmaWaveControlModel,
      });


    /*
     * ============================================================
     * FULL-YEAR DEBUG DATA
     * ============================================================
     *
     * There is deliberately no month filter here.
     *
     * Month selection belongs to the browser.
     */
    const data = existingSimulation.steps.map(
      (
        existingStep,
        index,
      ) => {

        const quantumStep =
          quantumGravitySimulation.steps[index];

        const plasmaStep =
          plasmaWaveSimulation.steps[index];

        const weatherPoint =
          quarterHour.weatherData[index];

        const plasmaDecision =
          plasmaWaveDecisions[index];


        if (
          !quantumStep ||
          !plasmaStep ||
          !weatherPoint
        ) {
          throw new Error(
            `Missing annual simulation point at index ${index}.`,
          );
        }


        const exportBeforeHpPower = Math.max(
          0,
          weatherPoint.pvPower -
            body.control.householdReservePower,
        );


        const existingHpPower =
          existingStep.heatPumpElectricalEnergy /
          TIME_STEP_HOURS;

        const quantumHpPower =
          quantumStep.heatPumpElectricalEnergy /
          TIME_STEP_HOURS;

        const plasmaHpPower =
          plasmaStep.heatPumpElectricalEnergy /
          TIME_STEP_HOURS;


        return {
          timestamp: existingStep.timestamp,

          /*
           * Forecast inventories.
           */
          required48hEnergy:
            plasmaDecision?.required48hEnergy ?? 0,

          kWhNoSunshine:
            plasmaDecision?.kWhNoSunshine ?? 0,


          /*
           * All three Waterlines.
           *
           * These are electrical power in kW_el.
           */
          userForecastWaterlinePower:
            plasmaDecision
              ?.userForecastWaterlinePower ??
            null,

          noSunshineWaterlinePower:
            plasmaDecision
              ?.noSunshineWaterlinePower ??
            null,

          effectiveWaterlinePower:
            plasmaDecision
              ?.effectiveWaterlinePower ??
            null,

          limitingStrategy:
            plasmaDecision
              ?.limitingStrategy ??
            null,


          /*
           * ========================================================
           * EXISTING CONTROLLER
           * ========================================================
           */
          existing: {
            controlSignal:
              existingStep.forecastTargetEnergy,

            heatPumpEnergy:
              existingStep.heatPumpElectricalEnergy,

            heatPumpElectricalPower:
              existingHpPower,

            storageEnergy:
              existingStep.storageEnergy,

            pvEnergy:
              existingStep.pvUsedByHeatPump,

            storageTemperature:
              calculateStorageTemperature(
                body.storage,
                {
                  storedEnergy:
                    existingStep.storageEnergy,
                },
              ),

            gridImportEnergy:
              existingStep.gridImport,

            pvUsedByHeatPumpEnergy:
              existingStep.pvUsedByHeatPump,

            storageLossEnergy:
              existingStep.storageLoss,

            unmetThermalDemand:
              existingStep.unmetThermalDemand,

            exportBeforeHpPower,

            exportAfterHpPower: Math.max(
              0,
              exportBeforeHpPower -
                existingHpPower,
            ),

            waterLevelPower:
              null,
          },


          /*
           * ========================================================
           * QUANTUM GRAVITY
           * ========================================================
           */
          quantumGravity: {
            controlSignal:
              quantumStep.forecastTargetEnergy,

            heatPumpEnergy:
              quantumStep.heatPumpElectricalEnergy,

            heatPumpElectricalPower:
              quantumHpPower,

            storageEnergy:
              quantumStep.storageEnergy,

            pvEnergy:
              quantumStep.pvUsedByHeatPump,

            storageTemperature:
              calculateStorageTemperature(
                body.storage,
                {
                  storedEnergy:
                    quantumStep.storageEnergy,
                },
              ),

            gridImportEnergy:
              quantumStep.gridImport,

            pvUsedByHeatPumpEnergy:
              quantumStep.pvUsedByHeatPump,

            storageLossEnergy:
              quantumStep.storageLoss,

            unmetThermalDemand:
              quantumStep.unmetThermalDemand,

            exportBeforeHpPower,

            exportAfterHpPower: Math.max(
              0,
              exportBeforeHpPower -
                quantumHpPower,
            ),

            waterLevelPower:
              null,
          },


          /*
           * ========================================================
           * COMBINED WATERLINE CONTROLLER
           * ========================================================
           */
          plasmaWave: {
            controlSignal:
              plasmaStep.forecastTargetEnergy,

            heatPumpEnergy:
              plasmaStep.heatPumpElectricalEnergy,

            heatPumpElectricalPower:
              plasmaHpPower,

            /*
             * HP electrical power requested by the combined
             * Waterline before mandatory reserve intervention.
             */
            waterlineHeatPumpElectricalPower:
              plasmaDecision
                ?.heatPumpElectricalPower ??
              0,

            storageEnergy:
              plasmaStep.storageEnergy,

            pvEnergy:
              plasmaStep.pvUsedByHeatPump,

            storageTemperature:
              calculateStorageTemperature(
                body.storage,
                {
                  storedEnergy:
                    plasmaStep.storageEnergy,
                },
              ),

            gridImportEnergy:
              plasmaStep.gridImport,

            pvUsedByHeatPumpEnergy:
              plasmaStep.pvUsedByHeatPump,

            storageLossEnergy:
              plasmaStep.storageLoss,

            unmetThermalDemand:
              plasmaStep.unmetThermalDemand,

            exportBeforeHpPower:
              plasmaDecision
                ?.exportBeforeHpPower ??
              exportBeforeHpPower,

            exportAfterHpPower:
              plasmaDecision
                ?.exportAfterHpPower ??
              Math.max(
                0,
                exportBeforeHpPower -
                  plasmaHpPower,
              ),

            /*
             * Backwards-compatible name.
             *
             * This now contains the effective combined Waterline.
             */
            waterLevelPower:
              plasmaDecision
                ?.effectiveWaterlinePower ??
              null,

            userForecastWaterlinePower:
              plasmaDecision
                ?.userForecastWaterlinePower ??
              null,

            noSunshineWaterlinePower:
              plasmaDecision
                ?.noSunshineWaterlinePower ??
              null,

            effectiveWaterlinePower:
              plasmaDecision
                ?.effectiveWaterlinePower ??
              null,

            limitingStrategy:
              plasmaDecision
                ?.limitingStrategy ??
              null,
          },
        };
      },
    );

    const simulationDurationMs =
      performance.now() - simulationStartTime;


    const responseBody = {
      year: body.year,
      storageVolume: body.storage.volume,

      simulationDurationMs,

      safetyFactor:
        body.control.forecastSafetyFactor,

      timeStepMinutes:
        TIME_STEP_HOURS * 60,

      dataPoints:
        data.length,

      planningHorizonHours:
        rollingDemandForecast.planningHorizonHours,

      userForecastHorizonHours:
        CONFIG.control.plasmaWavePlanningHorizonHours,

      noSunshineHorizonHours:
        CONFIG.control.noSunshineHorizonHours,

      noSunshineProtectedHorizonHours:
        CONFIG.control.noSunshineProtectedHorizonHours,

      data,
    };


    const compressedResponse =
      await writeHotWaterCache(
        cacheKey,
        responseBody,
      );


    return new NextResponse(
      new Uint8Array(compressedResponse),
      {
        status: 200,

        headers: {
          "Content-Type":
            "application/json; charset=utf-8",

          "Content-Encoding":
            "gzip",

          "Cache-Control":
            "no-store",

          "X-SunOpt-Cache":
            "MISS",

          "X-SunOpt-Cache-Key":
            cacheKey,
        },
      },
    );


  } catch (error) {

    console.error(
      "Hot-water debug simulation error:",
      error,
    );


    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Debug simulation failed.",
      },
      {
        status: 400,
      },
    );
  }
}