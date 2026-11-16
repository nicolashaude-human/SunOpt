// src/domain/hot-water-calculation.ts

import type {
  DailyHotWaterData,
  HotWaterData,
  HotWaterDemand,
  HotWaterProfile,
} from "./hot-water";

import type {
  WeatherData,
} from "./weather";


const WATER_SPECIFIC_HEAT =
  4.186;
// kJ/(kg K)

const WATER_DENSITY =
  1;
// kg/L

const KJ_PER_KWH =
  3600;


export function calculateThermalEnergy(
  waterVolume: number,
  coldWaterTemperature: number,
  hotWaterTemperature: number,
): number {

  const deltaTemperature =
    hotWaterTemperature -
    coldWaterTemperature;

  if (deltaTemperature < 0) {
    throw new Error(
      "Hot-water temperature cannot be below cold-water temperature.",
    );
  }

  return (
    waterVolume *
    WATER_DENSITY *
    WATER_SPECIFIC_HEAT *
    deltaTemperature /
    KJ_PER_KWH
  );
}


export function calculateDailyHotWaterDemand(
  demand: HotWaterDemand,
  profile: HotWaterProfile,
): DailyHotWaterData {

  if (
    profile.hourlyFractions.length !== 24
  ) {
    throw new Error(
      "Hot-water profile must contain exactly 24 hourly fractions.",
    );
  }


  const fractionSum =
    profile.hourlyFractions.reduce(
      (sum, fraction) =>
        sum + fraction,
      0,
    );


  if (
    Math.abs(fractionSum - 1) >
    0.000001
  ) {
    throw new Error(
      "Hot-water profile fractions must sum to 1.",
    );
  }


  const dailyWaterVolume =
    demand.persons *
    demand.dailyConsumptionPerPerson;


  return profile.hourlyFractions.map(
    (fraction, hour) => {

      const waterVolume =
        dailyWaterVolume *
        fraction;


      const thermalEnergy =
        calculateThermalEnergy(
          waterVolume,
          demand.coldWaterTemperature,
          demand.deliveryTemperature,
        );


      return {
        hour,
        waterVolume,
        thermalEnergy,
      };
    },
  );
}


export function calculateAnnualHotWaterDemand(
  demand: HotWaterDemand,
  profile: HotWaterProfile,
  weatherData: WeatherData,
): HotWaterData {

  if (
    !Array.isArray(weatherData) ||
    weatherData.length === 0
  ) {
    throw new Error(
      "Weather data must contain at least one data point.",
    );
  }


  const dailyProfile =
    calculateDailyHotWaterDemand(
      demand,
      profile,
    );


  return weatherData.map(
    (weatherPoint, index) => {

      const hour =
        index % 24;

      const dailyPoint =
        dailyProfile[hour];


      return {
        timestamp:
          weatherPoint.timestamp,

        hour,

        waterVolume:
          dailyPoint.waterVolume,

        thermalEnergy:
          dailyPoint.thermalEnergy,
      };
    },
  );
}