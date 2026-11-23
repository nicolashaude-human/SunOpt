// src/domain/space-heating-calculation.ts

import type {
  Building,
} from "./building";

import type {
  WeatherData,
} from "./weather";

import type {
  SpaceHeatingData,
} from "./space-heating";


export function calculateAnnualHeatingDemand(
  building: Pick<Building, "heatedArea" | "specificHeatingDemand">,
): number {

  if (building.heatedArea <= 0) {
    throw new Error(
      "Heated area must be greater than zero.",
    );
  }

  if (
    building.specificHeatingDemand < 0
  ) {
    throw new Error(
      "Specific heating demand cannot be negative.",
    );
  }

  return (
    building.heatedArea *
    building.specificHeatingDemand
  );
}


export function calculateSpaceHeatingProfile(
  building: Pick<
    Building,
    "heatedArea" | "specificHeatingDemand" | "indoorTemperature" | "heatingLimitTemperature"
  >,
  weatherData: WeatherData,
): SpaceHeatingData {

  if (weatherData.length === 0) {
    throw new Error(
      "Weather data cannot be empty.",
    );
  }


  const annualHeatingDemand =
    calculateAnnualHeatingDemand(
      building,
    );


  // ---------------------------------
  // Relative Heizlast jeder Stunde
  // ---------------------------------

  const weights =
    weatherData.map(
      (weather) => {

        if (
          weather.outdoorTemperature >=
          building.heatingLimitTemperature
        ) {
          return 0;
        }

        return Math.max(
          0,
          building.indoorTemperature -
            weather.outdoorTemperature,
        );
      },
    );


  const totalWeight =
    weights.reduce(
      (sum, value) =>
        sum + value,
      0,
    );


  if (totalWeight <= 0) {
    throw new Error(
      "No heating demand was calculated for the supplied weather data.",
    );
  }


  // ---------------------------------
  // Auf Jahresbedarf normieren
  // ---------------------------------

  return weatherData.map(
    (weather, index) => {

      const fraction =
        weights[index] /
        totalWeight;

      const thermalEnergy =
        annualHeatingDemand *
        fraction;

      return {
        timestamp:
          weather.timestamp,

        outdoorTemperature:
          weather.outdoorTemperature,

        thermalEnergy,
      };
    },
  );
}
