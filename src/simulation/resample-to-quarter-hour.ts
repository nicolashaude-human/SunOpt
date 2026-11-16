// src/simulation/resample-to-quarter-hour.ts

import type {
  HotWaterData,
  WeatherData,
} from "@/domain";


export interface QuarterHourData {
  weatherData: WeatherData;
  hotWaterData: HotWaterData;
}


export function resampleToQuarterHour(
  weatherData: WeatherData,
  hotWaterData: HotWaterData,
): QuarterHourData {

  if (
    weatherData.length !==
    hotWaterData.length
  ) {
    throw new Error(
      "Weather data and hot-water data must have the same length.",
    );
  }


  const quarterWeather: WeatherData = [];
  const quarterHotWater: HotWaterData = [];


  for (
    let index = 0;
    index < weatherData.length;
    index++
  ) {

    const weather =
      weatherData[index];

    const hotWater =
      hotWaterData[index];


    for (
      let quarter = 0;
      quarter < 4;
      quarter++
    ) {

      const timestamp =
        addMinutes(
          weather.timestamp,
          quarter * 15,
        );


      /*
       * PVGIS pvPower is treated as the average
       * electrical power during the hour.
       *
       * Therefore the power remains unchanged.
       */
      quarterWeather.push({
        ...weather,
        timestamp,
      });


      /*
       * The hourly DHW volume and energy are
       * divided equally into four 15-minute
       * intervals.
       *
       * Total daily/annual energy remains unchanged.
       */
      quarterHotWater.push({
        ...hotWater,

        timestamp,

        waterVolume:
          hotWater.waterVolume / 4,

        thermalEnergy:
          hotWater.thermalEnergy / 4,
      });
    }
  }


  return {
    weatherData:
      quarterWeather,

    hotWaterData:
      quarterHotWater,
  };
}


function addMinutes(
  timestamp: string,
  minutesToAdd: number,
): string {

  const date =
    parseTimestamp(timestamp);


  date.setUTCMinutes(
    date.getUTCMinutes() +
      minutesToAdd,
  );


  return date.toISOString();
}


function parseTimestamp(
  timestamp: string,
): Date {

  /*
   * First try normal ISO timestamps.
   */
  const normalDate =
    new Date(timestamp);


  if (
    !Number.isNaN(
      normalDate.getTime(),
    )
  ) {
    return normalDate;
  }


  /*
   * PVGIS commonly uses:
   *
   * YYYYMMDD:HHMM
   *
   * Example:
   * 20230101:0010
   */
  const match =
    timestamp.match(
      /^(\d{4})(\d{2})(\d{2}):(\d{2})(\d{2})$/,
    );


  if (!match) {
    throw new Error(
      `Unsupported timestamp format: ${timestamp}`,
    );
  }


  const [
    ,
    year,
    month,
    day,
    hour,
    minute,
  ] = match;


  return new Date(
    Date.UTC(
      Number(year),
      Number(month) - 1,
      Number(day),
      Number(hour),
      Number(minute),
    ),
  );
}