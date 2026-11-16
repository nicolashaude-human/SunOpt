// src/app/api/research/b12-physical-reference/route.ts

import {
  NextRequest,
  NextResponse,
} from "next/server";

import {
  calculateAnnualHotWaterDemand,
  calculateStorageCapacity,
  DEFAULT_HOT_WATER_PROFILE,

  type HeatPump,
  type HotWaterControl,
  type HotWaterDemand,
  type PVSystem,
  type ThermalStorage,
  type WeatherData,
} from "@/domain";

import {
  calculateThermalStorageStep,
} from "@/domain/thermal-storage-calculation";

import {
  CONFIG,
  HOT_WATER_TEST_CONFIG,
} from "@/config/config";

import {
  resampleToQuarterHour,
} from "@/simulation/resample-to-quarter-hour";

import {
  createPlasmaWaveControlModel,
  type PlasmaWaveControlDecision,
} from "@/simulation/control-model-plasma-wave";

import {
  simulateHotWaterSystem,
} from "@/simulation/hot-water-simulation";

import {
  getPlasmaWaveHistoricalSeries,
} from "@/simulation/plasma-wave-historical-series-cache";

import {
  mkdir,
  open,
  stat,
  writeFile,
} from "node:fs/promises";

import path from "node:path";


export const runtime =
  "nodejs";

export const dynamic =
  "force-dynamic";


interface WeatherApiResponse {
  weatherData: WeatherData;
}


interface RequestBody {
  heatPumpThermalPower?:
    number;

  pvPeakPower?:
    number;
}


const YEAR =
  2023;

const POSTAL_CODE =
  "8001";

const TIME_STEP_HOURS =
  HOT_WATER_TEST_CONFIG
    .timeStepHours;

const STORAGE_VOLUMES =
  Array.from(
    {
      length:
        ((3000 - 300) / 100) +
        1,
    },
    (_, index) =>
      300 +
      index *
        100,
  );


const FIELD_NAMES = [
  "storageStart",
  "storageAfterDemand",
  "directThermalDemand",
  "minimumReserveEnergy",
  "mandatoryStorageCharge",
  "storageAfterMandatoryCharge",
  "forecastTargetEnergy",
  "forecastRequiredEnergy",
  "required48hEnergy",
  "kWhNoSunshine",
  "userForecastWaterlinePower",
  "noSunshineWaterlinePower",
  "effectiveWaterlinePower",
  "exportBeforeHpPower",
  "exportAfterHpPower",
  "heatPumpElectricalPower",
  "heatPumpElectricalEnergy",
  "heatPumpThermalEnergy",
  "storageLoss",
  "storageEnd",
  "gridImport",
  "pvUsedByHeatPump",
  "pvSurplus",
  "unmetThermalDemand",
] as const;


const MAGIC =
  "SUNOPT_B12_PHY1";

const VERSION =
  1;


const COUPLED_MAGIC =
  "SUNOPT_B12_CPL1";

const COUPLED_VERSION =
  1;

const COUPLED_DYNAMIC_FIELDS = [
  "currentStorageEnergy",
  "minimumReserveEnergy",
  "requestedElectricalEnergy",
  "targetStorageEnergy",
  "maximumHeatPumpElectricalPower",
  "cop",
  "requiredThermalInventory",
  "requestedChargeThermalEnergy",
] as const;


function makeStorage(
  volume:
    number,
): ThermalStorage {

  return {
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
  };
}


function makeControl():
  HotWaterControl {

  return {
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
  };
}


function encodeHeader(
  stepCount:
    number,
): Buffer {

  /*
   * Layout:
   *
   * bytes  0..15  magic, zero padded
   * u32   16      version
   * u32   20      stepCount
   * u32   24      tankCount
   * u32   28      fieldCount
   * f64   32      timeStepHours
   * f64[] 40      storage volumes
   *
   * Records follow immediately after the volumes.
   *
   * Record order:
   *
   *   tank-major
   *   [tank][step][field]
   */
  const bytes =
    40 +
    STORAGE_VOLUMES.length *
      8;

  const buffer =
    Buffer.alloc(
      bytes,
      0,
    );

  buffer.write(
    MAGIC,
    0,
    "ascii",
  );

  buffer.writeUInt32LE(
    VERSION,
    16,
  );

  buffer.writeUInt32LE(
    stepCount,
    20,
  );

  buffer.writeUInt32LE(
    STORAGE_VOLUMES.length,
    24,
  );

  buffer.writeUInt32LE(
    FIELD_NAMES.length,
    28,
  );

  buffer.writeDoubleLE(
    TIME_STEP_HOURS,
    32,
  );

  for (
    let index = 0;
    index <
      STORAGE_VOLUMES.length;
    index++
  ) {
    buffer.writeDoubleLE(
      STORAGE_VOLUMES[index] ??
        0,
      40 +
        index *
          8,
    );
  }

  return buffer;
}


function encodeCoupledHeader(
  stepCount:
    number,

  horizonSteps:
    number,
): Buffer {

  /*
   * B12 coupled snapshot layout:
   *
   * bytes  0..15  magic, zero padded
   * u32   16      version
   * u32   20      stepCount
   * u32   24      tankCount
   * u32   28      dynamicFieldCount
   * f64   32      timeStepHours
   * u32   40      horizonSteps
   * u32   44      reserved
   * f64[] 48      storage volumes
   *
   * Then:
   *   exportPowerByIndex[stepCount]      f64
   *   loadEnergyByIndex[stepCount]       f64
   *   dynamics[tank][step][field]        f64
   */
  const bytes =
    48 +
    STORAGE_VOLUMES.length *
      8;

  const buffer =
    Buffer.alloc(
      bytes,
      0,
    );

  buffer.write(
    COUPLED_MAGIC,
    0,
    "ascii",
  );

  buffer.writeUInt32LE(
    COUPLED_VERSION,
    16,
  );

  buffer.writeUInt32LE(
    stepCount,
    20,
  );

  buffer.writeUInt32LE(
    STORAGE_VOLUMES.length,
    24,
  );

  buffer.writeUInt32LE(
    COUPLED_DYNAMIC_FIELDS.length,
    28,
  );

  buffer.writeDoubleLE(
    TIME_STEP_HOURS,
    32,
  );

  buffer.writeUInt32LE(
    horizonSteps,
    40,
  );

  for (
    let index = 0;
    index <
      STORAGE_VOLUMES.length;
    index++
  ) {
    buffer.writeDoubleLE(
      STORAGE_VOLUMES[index] ??
        0,
      48 +
        index *
          8,
    );
  }

  return buffer;
}


function assertFloat64ArraysEqual(
  label:
    string,

  reference:
    Float64Array,

  candidate:
    Float64Array,
): void {

  if (
    reference.length !==
    candidate.length
  ) {
    throw new Error(
      `${label} length mismatch: ${reference.length} vs ${candidate.length}.`,
    );
  }

  for (
    let index = 0;
    index <
      reference.length;
    index++
  ) {
    if (
      reference[index] !==
      candidate[index]
    ) {
      throw new Error(
        `${label} differs at index ${index}: ${reference[index]} vs ${candidate[index]}.`,
      );
    }
  }
}


export async function POST(
  request:
    NextRequest,
): Promise<NextResponse> {

  const started =
    performance.now();

  try {

    const body =
      (await request.json()) as
        RequestBody;

    const heatPumpThermalPower =
      body
        .heatPumpThermalPower ??
      5;

    const pvPeakPower =
      body
        .pvPeakPower ??
      10;


    const pvSystem:
      PVSystem = {

      peakPower:
        pvPeakPower,

      azimuth:
        0,

      inclination:
        30,

      systemLoss:
        0.14,
    };


    const hotWaterDemand:
      HotWaterDemand = {

      persons:
        2,

      dailyConsumptionPerPerson:
        50,

      coldWaterTemperature:
        10,

      deliveryTemperature:
        60,
    };


    const heatPump:
      HeatPump = {

      thermalPower:
        heatPumpThermalPower,

      maxFlowTemperature:
        75,

      copModel: {
        type:
          "constant",

        cop:
          3,
      },
    };


    const control =
      makeControl();


    const weatherResponse =
      await fetch(
        `${request.nextUrl.origin}/api/weather`,
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
                POSTAL_CODE,

              pvSystem,

              startYear:
                YEAR,

              endYear:
                YEAR,
            }),

          cache:
            "no-store",
        },
      );


    if (
      !weatherResponse.ok
    ) {
      const errorText =
        await weatherResponse.text();

      throw new Error(
        `Weather API failed: ${errorText}`,
      );
    }


    const weatherResult =
      (await weatherResponse.json()) as
        WeatherApiResponse;


    const hourlyHotWaterData =
      calculateAnnualHotWaterDemand(
        hotWaterDemand,
        DEFAULT_HOT_WATER_PROFILE,
        weatherResult.weatherData,
      );


    const quarterHour =
      resampleToQuarterHour(
        weatherResult.weatherData,
        hourlyHotWaterData,
      );


    const stepCount =
      quarterHour
        .weatherData
        .length;


    if (
      stepCount ===
        0 ||
      quarterHour
        .hotWaterData
        .length !==
        stepCount
    ) {
      throw new Error(
        "Quarter-hour annual data is empty or inconsistent.",
      );
    }


    const researchDirectory =
      path.join(
        process.cwd(),
        "rust",
        "sunopt-node",
      );


    await mkdir(
      researchDirectory,
      {
        recursive:
          true,
      },
    );


    const binaryPath =
      path.join(
        researchDirectory,
        "blackwell-physical-reference.bin",
      );


    const metadataPath =
      path.join(
        researchDirectory,
        "blackwell-physical-reference.json",
      );


    const coupledPath =
      path.join(
        researchDirectory,
        "blackwell-b12-coupled-reference.bin",
      );


    const coupledMetadataPath =
      path.join(
        researchDirectory,
        "blackwell-b12-coupled-reference.json",
      );


    const planningHorizonSteps =
      Math.min(
        Math.ceil(
          HOT_WATER_TEST_CONFIG
            .userForecastHours /
          TIME_STEP_HOURS,
        ),
        stepCount,
      );


    /*
     * Use the production historical-series builder itself.
     * This guarantees that B12 records the same static export/load arrays
     * consumed by the 48 h causal accelerator.
     */
    const firstStorage =
      makeStorage(
        STORAGE_VOLUMES[0] ??
          300,
      );


    const firstHistoricalSeries =
      getPlasmaWaveHistoricalSeries({
        weatherData:
          quarterHour.weatherData,

        hotWaterData:
          quarterHour.hotWaterData,

        heatPump,

        storage:
          firstStorage,

        timeStepHours:
          TIME_STEP_HOURS,

        householdReservePower:
          control.householdReservePower,

        safetyFactor:
          control.forecastSafetyFactor,

        lossReferenceTemperature:
          firstStorage.minTemperature,
      });


    const coupledExportPower =
      Float64Array.from(
        firstHistoricalSeries
          .exportPowerByIndex,
      );


    const coupledLoadEnergy =
      Float64Array.from(
        firstHistoricalSeries
          .loadEnergyByIndex,
      );


    if (
      coupledExportPower.length !==
        stepCount ||
      coupledLoadEnergy.length !==
        stepCount
    ) {
      throw new Error(
        `B12 historical series length mismatch: export=${coupledExportPower.length}, load=${coupledLoadEnergy.length}, steps=${stepCount}.`,
      );
    }


    const handle =
      await open(
        binaryPath,
        "w",
      );


    const coupledHandle =
      await open(
        coupledPath,
        "w",
      );


    try {

      await handle.write(
        encodeHeader(
          stepCount,
        ),
      );


      await coupledHandle.write(
        encodeCoupledHeader(
          stepCount,
          planningHorizonSteps,
        ),
      );


      await coupledHandle.write(
        Buffer.from(
          coupledExportPower.buffer,
          coupledExportPower.byteOffset,
          coupledExportPower.byteLength,
        ),
      );


      await coupledHandle.write(
        Buffer.from(
          coupledLoadEnergy.buffer,
          coupledLoadEnergy.byteOffset,
          coupledLoadEnergy.byteLength,
        ),
      );


      for (
        const storageVolume of
        STORAGE_VOLUMES
      ) {

        const storage =
          makeStorage(
            storageVolume,
          );


        /*
         * Fail closed if the current production historical-series builder
         * ever becomes tank-dependent. The coupled format stores these
         * static annual arrays once, so silent divergence is forbidden.
         */
        const tankHistoricalSeries =
          getPlasmaWaveHistoricalSeries({
            weatherData:
              quarterHour.weatherData,

            hotWaterData:
              quarterHour.hotWaterData,

            heatPump,

            storage,

            timeStepHours:
              TIME_STEP_HOURS,

            householdReservePower:
              control.householdReservePower,

            safetyFactor:
              control.forecastSafetyFactor,

            lossReferenceTemperature:
              storage.minTemperature,
          });


        assertFloat64ArraysEqual(
          `Tank ${storageVolume} L exportPowerByIndex`,
          coupledExportPower,
          Float64Array.from(
            tankHistoricalSeries
              .exportPowerByIndex,
          ),
        );


        assertFloat64ArraysEqual(
          `Tank ${storageVolume} L loadEnergyByIndex`,
          coupledLoadEnergy,
          Float64Array.from(
            tankHistoricalSeries
              .loadEnergyByIndex,
          ),
        );


        const storageCapacity =
          calculateStorageCapacity(
            storage,
          );


        const decisions:
          Array<
            PlasmaWaveControlDecision |
            undefined
          > =
          new Array(
            stepCount,
          );


        const controlModel =
          createPlasmaWaveControlModel({
            weatherData:
              quarterHour.weatherData,

            hotWaterData:
              quarterHour.hotWaterData,

            heatPump,

            storage,

            control,

            timeStepHours:
              TIME_STEP_HOURS,

            planningHorizonHours:
              HOT_WATER_TEST_CONFIG
                .userForecastHours,

            targetTemperature:
              75,

            onDecision:
              (
                decision,
              ) => {

                decisions[
                  decision.currentIndex
                ] =
                  decision;
              },
          });


        const simulation =
          simulateHotWaterSystem({
            weatherData:
              quarterHour.weatherData,

            hotWaterData:
              quarterHour.hotWaterData,

            heatPump,

            storage,

            control,

            timeStepHours:
              TIME_STEP_HOURS,

            controlModel,
          });


        if (
          simulation.steps.length !==
          stepCount
        ) {
          throw new Error(
            `Tank ${storageVolume} L returned ${simulation.steps.length} steps instead of ${stepCount}.`,
          );
        }


        const values =
          new Float64Array(
            stepCount *
            FIELD_NAMES.length,
          );


        const coupledValues =
          new Float64Array(
            stepCount *
            COUPLED_DYNAMIC_FIELDS.length,
          );


        /*
         * Benchmark storage starts at 60 °C, equal to minTemperature.
         * Usable energy relative to minTemperature is therefore exactly 0.
         */
        let storageStart =
          0;


        for (
          let index = 0;
          index <
            stepCount;
          index++
        ) {

          const step =
            simulation
              .steps[index];

          const decision =
            decisions[index];

          const weather =
            quarterHour
              .weatherData[index];

          const hotWater =
            quarterHour
              .hotWaterData[index];


          if (
            !step ||
            !decision ||
            !weather ||
            !hotWater
          ) {
            throw new Error(
              `Missing B12 reference data for tank ${storageVolume} L at index ${index}.`,
            );
          }


          /*
           * Re-evaluate the exact canonical no-charge preview used by the
           * production simulator. This deliberately does not infer the value
           * from the final state.
           */
          const preview =
            calculateThermalStorageStep({
              storage,

              currentEnergy:
                storageStart,

              thermalInputEnergy:
                0,

              thermalOutputEnergy:
                hotWater
                  .thermalEnergy,

              timeStepHours:
                TIME_STEP_HOURS,
            });


          const storageAfterDemand =
            preview
              .state
              .storedEnergy;


          const directThermalDemand =
            preview
              .unmetDemand;


          const missingToReserve =
            Math.max(
              0,

              step
                .minimumReserveEnergy -
                storageAfterDemand,
            );


          const availableStorageCapacity =
            Math.max(
              0,

              storageCapacity -
                storageAfterDemand,
            );


          const mandatoryStorageCharge =
            Math.min(
              missingToReserve,
              availableStorageCapacity,
            );


          const storageAfterMandatoryCharge =
            storageAfterDemand +
            mandatoryStorageCharge;


          const heatPumpElectricalPower =
            step
              .heatPumpElectricalEnergy /
            TIME_STEP_HOURS;


          const base =
            index *
            FIELD_NAMES.length;


          /*
           * Exact current controller handoff into the acceleration scope.
           * IMPORTANT: the controller receives the PRE-PREVIEW storageState.
           */
          const controllerMissingToReserve =
            Math.max(
              0,
              step.minimumReserveEnergy -
                storageStart,
            );


          const controllerCurrentStorageEnergy =
            Math.min(
              storageCapacity,
              storageStart +
                controllerMissingToReserve,
            );


          const cop =
            3;


          const maximumHeatPumpElectricalPower =
            heatPumpThermalPower /
            cop;


          const requestedChargeThermalEnergy =
            Math.min(
              Math.max(
                0,
                decision.required48hEnergy -
                  controllerCurrentStorageEnergy,
              ),
              Math.max(
                0,
                storageCapacity -
                  controllerCurrentStorageEnergy,
              ),
            );


          /*
           * Cross-check the scheduler diagnostic written by the actual
           * controller. A mismatch means our reconstruction no longer
           * matches production semantics and B12 must stop.
           */
          if (
            Math.abs(
              requestedChargeThermalEnergy -
                step.forecastRequiredEnergy,
            ) >
            1.0e-9
          ) {
            throw new Error(
              `B12 requested-charge parity failed for tank ${storageVolume} L at step ${index}: derived=${requestedChargeThermalEnergy}, controller=${step.forecastRequiredEnergy}.`,
            );
          }


          const coupledBase =
            index *
            COUPLED_DYNAMIC_FIELDS.length;


          const coupledRow = [
            controllerCurrentStorageEnergy,
            step.minimumReserveEnergy,
            requestedChargeThermalEnergy /
              cop,
            storageCapacity,
            maximumHeatPumpElectricalPower,
            cop,
            decision.required48hEnergy,
            requestedChargeThermalEnergy,
          ];


          for (
            let field = 0;
            field <
              coupledRow.length;
            field++
          ) {
            coupledValues[
              coupledBase +
              field
            ] =
              coupledRow[field] ??
              Number.NaN;
          }


          const row = [
            storageStart,
            storageAfterDemand,
            directThermalDemand,
            step.minimumReserveEnergy,
            mandatoryStorageCharge,
            storageAfterMandatoryCharge,
            step.forecastTargetEnergy,
            step.forecastRequiredEnergy,
            decision.required48hEnergy,
            decision.kWhNoSunshine,
            decision.userForecastWaterlinePower,
            decision.noSunshineWaterlinePower,
            decision.effectiveWaterlinePower,
            decision.exportBeforeHpPower,
            decision.exportAfterHpPower,
            heatPumpElectricalPower,
            step.heatPumpElectricalEnergy,
            step.heatPumpThermalEnergy,
            step.storageLoss,
            step.storageEnergy,
            step.gridImport,
            step.pvUsedByHeatPump,
            step.pvSurplus,
            step.unmetThermalDemand,
          ];


          for (
            let field = 0;
            field <
              row.length;
            field++
          ) {
            values[
              base +
              field
            ] =
              row[field] ??
              Number.NaN;
          }


          storageStart =
            step
              .storageEnergy;
        }


        await handle.write(
          Buffer.from(
            values.buffer,
            values.byteOffset,
            values.byteLength,
          ),
        );


        await coupledHandle.write(
          Buffer.from(
            coupledValues.buffer,
            coupledValues.byteOffset,
            coupledValues.byteLength,
          ),
        );


        console.log(
          "[SunOpt B12] reference tank complete",
          {
            storageVolume,
            steps:
              stepCount,
          },
        );
      }

    } finally {

      await handle.close();
      await coupledHandle.close();
    }


    const binaryStat =
      await stat(
        binaryPath,
      );


    const coupledStat =
      await stat(
        coupledPath,
      );


    const coupledMetadata = {
      format:
        "sunopt-b12-coupled-reference",

      version:
        COUPLED_VERSION,

      binaryFile:
        path.basename(
          coupledPath,
        ),

      stepCount,

      tankCount:
        STORAGE_VOLUMES.length,

      storageVolumes:
        STORAGE_VOLUMES,

      timeStepHours:
        TIME_STEP_HOURS,

      horizonSteps:
        planningHorizonSteps,

      dynamicFieldCount:
        COUPLED_DYNAMIC_FIELDS.length,

      dynamicFields:
        COUPLED_DYNAMIC_FIELDS,

      staticArrays: [
        "exportPowerByIndex",
        "loadEnergyByIndex",
      ],

      binaryBytes:
        coupledStat.size,

      semanticNote:
        "Generated atomically in the same current B12 annual run as blackwell-physical-reference.bin. Static arrays come from getPlasmaWaveHistoricalSeries(). Dynamic currentStorageEnergy uses the production controller PRE-PREVIEW mandatory-reserve handoff. requestedChargeThermalEnergy is cross-checked against simulation step.forecastRequiredEnergy at every job.",
    };


    await writeFile(
      coupledMetadataPath,
      JSON.stringify(
        coupledMetadata,
        null,
        2,
      ) +
        "\n",
      "utf8",
    );


    const metadata = {
      format:
        "sunopt-b12-physical-reference",

      version:
        VERSION,

      binaryFile:
        path.basename(
          binaryPath,
        ),

      recordOrder:
        "tank-major",

      year:
        YEAR,

      postalCode:
        POSTAL_CODE,

      timeStepHours:
        TIME_STEP_HOURS,

      stepCount,

      tankCount:
        STORAGE_VOLUMES.length,

      storageVolumes:
        STORAGE_VOLUMES,

      fieldCount:
        FIELD_NAMES.length,

      fields:
        FIELD_NAMES,

      binaryBytes:
        binaryStat.size,

      heatPumpThermalPower,
      pvPeakPower,

      benchmarkStorage: {
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
          control.minimumReserveHours,

        forecastHorizonHours:
          control.forecastHorizonHours,

        pvTargetTemperature:
          control.pvTargetTemperature,

        householdReservePower:
          control.householdReservePower,

        forecastSafetyFactor:
          control.forecastSafetyFactor,

        plasmaWavePlanningHorizonHours:
          HOT_WATER_TEST_CONFIG
            .userForecastHours,

        noSunshineHorizonHours:
          CONFIG.control
            .noSunshineHorizonHours,

        noSunshineProtectedHorizonHours:
          CONFIG.control
            .noSunshineProtectedHorizonHours,
      },

      semanticNote:
        "Reference preserves current production semantics. Controller decisions are the actual onDecision values. storageAfterDemand is recalculated with calculateThermalStorageStep from the previous physical storageEnd. No production algorithm is modified.",
    };


    await writeFile(
      metadataPath,
      JSON.stringify(
        metadata,
        null,
        2,
      ) +
        "\n",
      "utf8",
    );


    const totalDurationMs =
      performance.now() -
      started;


    console.log(
      "[SunOpt B12] physical reference complete",
      {
        binaryPath,
        metadataPath,
        stepCount,
        tankCount:
          STORAGE_VOLUMES.length,
        fieldCount:
          FIELD_NAMES.length,
        binaryBytes:
          binaryStat.size,
        coupledPath,
        coupledMetadataPath,
        coupledBytes:
          coupledStat.size,
        totalDurationMs,
      },
    );


    return NextResponse.json({
      ok:
        true,

      binaryPath,
      metadataPath,

      stepCount,

      tankCount:
        STORAGE_VOLUMES.length,

      fieldCount:
        FIELD_NAMES.length,

      binaryBytes:
        binaryStat.size,

      coupledPath,
      coupledMetadataPath,

      coupledBytes:
        coupledStat.size,

      totalDurationMs,
    });

  } catch (
    error
  ) {

    console.error(
      "[SunOpt B12] physical reference failed",
      error,
    );


    return NextResponse.json(
      {
        ok:
          false,

        error:
          error instanceof Error
            ? error.message
            : String(
                error,
              ),
      },
      {
        status:
          500,
      },
    );
  }
}

