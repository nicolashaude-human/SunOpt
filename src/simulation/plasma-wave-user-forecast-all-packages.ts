// src/simulation/plasma-wave-user-forecast-all-packages.ts



import type {

  HeatPump,

  HotWaterData,

  ThermalStorage,

  WeatherData,

} from "@/domain";



import {

  calculateStorageThermalConductance,

} from "@/domain"; // wir in index deklariert



import {
  getPlasmaWaveHistoricalSeries,
} from "./plasma-wave-historical-series-cache";





export interface PlasmaWaveAllPackagesForecastPoint {



  kWhUser:

    number;



  /*

   * Raw forecast DHW energy that must be

   * covered until the next PV START.

   *

   * Unit: kWh_th

   */

  hotWaterForecastEnergy:

    number;



  /*

   * Same DHW forecast, but each future

   * consumption segment is corrected for

   * storage time since PV END.

   *

   * Unit: kWh_th

   */

  lossAdjustedHotWaterForecastEnergy: number;



  /*

   * Difference between loss-adjusted and

   * raw DHW forecast.

   *

   * Unit: kWh_th

   */

  hotWaterForecastLossEnergy:

    number;



  segmentLoadEnergy:

    number;



  segmentPvAfterHouseholdEnergy:

    number;



  segmentPvThermalEnergy:

    number;



  segmentNetEnergy:

    number;



  horizonLoadEnergy:

    number;



  horizonPvThermalEnergy:

    number;



  maximumCumulativeDeficit:

    number;



  pvStart:

    boolean;



  pvEnd:

    boolean;

}





export interface PlasmaWaveAllPackagesForecastResult {



  planningHorizonHours:

    number;



  forecastSteps:

    number;



  points:

    PlasmaWaveAllPackagesForecastPoint[];

}





export interface PlasmaWaveAllPackagesForecastInput {



  weatherData:

    WeatherData;



  hotWaterData:

    HotWaterData;



  heatPump:

    HeatPump;



  storage:

    ThermalStorage;



  timeStepHours:

    number;



  planningHorizonHours:

    number;



  householdReservePower:

    number;



  lossReferenceTemperature:

    number;



  safetyFactor:

    number;

}





function wrapIndex(

  index: number,

  length: number,

): number {



  return (

    (

      index %

      length

    ) +

    length

  ) % length;

}





function calculateStorageRetentionFactor(

  storage: ThermalStorage,

  storageHours: number,

): number {



  if (

    storageHours <= 0

  ) {



    return 1;

  }





  const thermalConductance =

    calculateStorageThermalConductance(

      storage,

    );





  if (

    thermalConductance <= 0

  ) {



    return 1;

  }





  const thermalCapacityJPerK =

    storage.volume *

    4186;





  const timeConstantSeconds =

    thermalCapacityJPerK /

    thermalConductance;





  if (

    timeConstantSeconds <= 0

  ) {



    return 1;

  }





  const storageSeconds =

    storageHours *

    3600;





  return Math.exp(

    -storageSeconds /

    timeConstantSeconds,

  );

}





export function calculatePlasmaWaveAllPackagesForecast(

  input: PlasmaWaveAllPackagesForecastInput,

): PlasmaWaveAllPackagesForecastResult {



  const {

    weatherData,

    hotWaterData,

    heatPump,

    storage,

    timeStepHours,

    planningHorizonHours,

    householdReservePower,

    lossReferenceTemperature,

    safetyFactor,

  } = input;





  if (

    weatherData.length !==

    hotWaterData.length

  ) {



    throw new Error(

      "Weather data and hot-water data must have the same length.",

    );

  }





  if (

    weatherData.length === 0

  ) {



    return {

      planningHorizonHours,

      forecastSteps: 0,

      points: [],

    };

  }





  if (

    timeStepHours <= 0

  ) {



    throw new Error(

      "timeStepHours must be greater than zero.",

    );

  }





  if (

    planningHorizonHours <= 0

  ) {



    throw new Error(

      "planningHorizonHours must be greater than zero.",

    );

  }





  if (

    safetyFactor <= 0

  ) {



    throw new Error(

      "safetyFactor must be greater than zero.",

    );

  }





  const forecastSteps =

    Math.min(

      Math.ceil(

        planningHorizonHours /

        timeStepHours,

      ),

      weatherData.length,

    );

    const historicalSeries =
    getPlasmaWaveHistoricalSeries({
      weatherData,
      hotWaterData,
      heatPump,
      storage,
      timeStepHours,
      householdReservePower,
      safetyFactor,
      lossReferenceTemperature,
    });
  
  
  const segmentLoadEnergy =
    historicalSeries.loadEnergyByIndex;
  
  
  const segmentPvAfterHouseholdEnergy =
    historicalSeries
      .pvAfterHouseholdEnergyByIndex;
  
  
  const segmentPvThermalEnergy =
    historicalSeries
      .usablePvThermalEnergyByIndex;

  const hotWaterForecastEnergyByIndex =

    new Array<number>(

      weatherData.length,

    ).fill(0);





  const lossAdjustedHotWaterForecastEnergyByIndex =

    new Array<number>(

      weatherData.length,

    ).fill(0);





  const hotWaterForecastLossEnergyByIndex =

    new Array<number>(

      weatherData.length,

    ).fill(0);

  /*
   * ================================================================
   * FORWARD / EARLIEST-POSSIBLE DHW PACKAGE FORECAST
   * ================================================================
   *
   * This branch answers one deliberately simple question:
   *
   *   "How much thermal energy would be required NOW if every
   *    protected DHW package in the next planning horizon were
   *    stored as early as possible?"
   *
   * Normally the planning horizon is 48 h.
   *
   * IMPORTANT:
   *
   * PV START and PV END do NOT delimit this forecast. On rainy or
   * cloudy days there may be several PV START / PV END transitions.
   * We therefore ignore these transitions here and collect ALL future
   * DHW packages inside the complete horizon.
   *
   * A separate later branch will calculate the "latest possible"
   * schedule using the passive tank-cooling interval / roughly seven
   * days. The two branches must not be mixed here.
   *
   * LOSS MODEL
   * ----------
   *
   * Each future 15-minute DHW package has its own storage duration:
   *
   *   storageHours = offset * timeStepHours
   *
   * Because this is the earliest-possible branch, time zero is NOW.
   *
   * The further in the future the package is consumed, the longer it
   * is assumed to remain in the tank and the larger its loss uplift.
   *
   *   retentionFactor = exp(-t / tau)
   *
   *   requiredStoredEnergy =
   *       dhwEnergy / retentionFactor
   *
   * This remains a DIAGNOSTIC forecast. The controller later decides
   * when PV and heat-pump power actually permit charging.
   * ================================================================
   */
  for (
    let currentIndex = 0;
    currentIndex < weatherData.length;
    currentIndex++
  ) {

    let rawForecastEnergy =
      0;

    let lossAdjustedForecastEnergy =
      0;


    /*
     * Scan every 15-minute package in the COMPLETE forward horizon.
     *
     * No break at PV END.
     * No break at PV START.
     * No selection of one dark interval.
     */
    for (
      let offset = 0;
      offset < forecastSteps;
      offset++
    ) {

      const index =
        wrapIndex(
          currentIndex +
            offset,
          weatherData.length,
        );

      /*
       * Thermal DHW demand belonging to this individual 15-minute
       * package.
       */
      const dhwEnergy =
        historicalSeries
          .loadEnergyByIndex[index] ?? 0;


      if (
        dhwEnergy <= 0
      ) {

        continue;
      }


      /*
       * RAW forecast:
       * every protected DHW package is counted exactly once.
       */
      rawForecastEnergy +=
        dhwEnergy;


      /*
       * Earliest-possible storage age.
       *
       * offset = 0  -> package needed now      -> 0 h storage
       * offset = 4  -> package needed in 1 h  -> 1 h storage
       * offset = 96 -> package needed in 24 h -> 24 h storage
       */
      const storageHours =
        offset *
        timeStepHours;


      const retentionFactor =
        calculateStorageRetentionFactor(
          storage,
          storageHours,
        );


      /*
       * Energy which would have to be present NOW so that this future
       * DHW package is still available after its holding losses.
       */
      const requiredStoredEnergy =
        retentionFactor > 0
          ? dhwEnergy /
            retentionFactor
          : dhwEnergy;


      lossAdjustedForecastEnergy +=
        requiredStoredEnergy;
    }


    hotWaterForecastEnergyByIndex[
      currentIndex
    ] =
      rawForecastEnergy;


    lossAdjustedHotWaterForecastEnergyByIndex[
      currentIndex
    ] =
      lossAdjustedForecastEnergy;


    hotWaterForecastLossEnergyByIndex[
      currentIndex
    ] =
      Math.max(
        0,
        lossAdjustedForecastEnergy -
          rawForecastEnergy,
      );
  }







  const points:

    PlasmaWaveAllPackagesForecastPoint[] =

      new Array(

        weatherData.length,

      );





  for (

    let currentIndex = 0;

    currentIndex <

      weatherData.length;

    currentIndex++

  ) {



    let cumulativeDeficit =

      0;



    let maximumCumulativeDeficit =

      0;



    let horizonLoadEnergy =

      0;



    let horizonPvThermalEnergy =

      0;





    for (

      let offset = 0;

      offset < forecastSteps;

      offset++

    ) {



      const index =

        wrapIndex(

          currentIndex +

            offset,

          weatherData.length,

        );



        // marker

        const futureDemand =
        historicalSeries
          .loadEnergyByIndex[index] ?? 0;
      
      
      const pvThermal =
        historicalSeries
          .usablePvThermalEnergyByIndex[index] ?? 0;

          // marker


      horizonLoadEnergy +=

        futureDemand;





      horizonPvThermalEnergy +=

        pvThermal;





      cumulativeDeficit +=

        futureDemand -

        pvThermal;





      maximumCumulativeDeficit =

        Math.max(

          maximumCumulativeDeficit,

          cumulativeDeficit,

        );

    }





    const currentSegmentLoad =

      segmentLoadEnergy[

        currentIndex

      ];





    const currentSegmentPv =

      segmentPvThermalEnergy[

        currentIndex

      ];





    points[currentIndex] = {



      kWhUser:

        Math.max(

          0,

          maximumCumulativeDeficit,

        ),



      hotWaterForecastEnergy:

        hotWaterForecastEnergyByIndex[

          currentIndex

        ],



      lossAdjustedHotWaterForecastEnergy:

        lossAdjustedHotWaterForecastEnergyByIndex[

          currentIndex

        ],



      hotWaterForecastLossEnergy:

        hotWaterForecastLossEnergyByIndex[

          currentIndex

        ],



      segmentLoadEnergy:

        currentSegmentLoad,



      segmentPvAfterHouseholdEnergy:

        segmentPvAfterHouseholdEnergy[

          currentIndex

        ],



      segmentPvThermalEnergy:

        currentSegmentPv,



      segmentNetEnergy:

        currentSegmentLoad -

        currentSegmentPv,



      horizonLoadEnergy,



      horizonPvThermalEnergy,



      maximumCumulativeDeficit:

        Math.max(

          0,

          maximumCumulativeDeficit,

        ),



        pvStart:
        historicalSeries
          .pvStartByIndex[currentIndex] === 1,
      
      pvEnd:
        historicalSeries
          .pvEndByIndex[currentIndex] === 1,

    };

  }





  return {

    planningHorizonHours,



    forecastSteps,



    points,

  };

}
