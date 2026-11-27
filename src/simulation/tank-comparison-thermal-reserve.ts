import { calculateSpaceHeatingProfile, calculateStorageCapacity, type HotWaterData, type ThermalStorage, type WeatherData } from "@/domain";
import { CONFIG } from "@/config/config";
import type { TankComparisonThermalReserve } from "@/types/tank-comparison";

export function createTankComparisonStorage(
  storageVolume:
    number,
): ThermalStorage {

  return {
    volume:
      storageVolume,

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


function maxRollingThermalDemand(
  demand: readonly { thermalEnergy: number }[],
  horizonSteps: number,
): number {
  let windowEnergy = 0;
  let maxWindowEnergy = 0;

  for (const [index, step] of demand.entries()) {
    windowEnergy += step.thermalEnergy;
    if (index >= horizonSteps) {
      windowEnergy -= demand[index - horizonSteps].thermalEnergy;
    }
    maxWindowEnergy = Math.max(maxWindowEnergy, windowEnergy);
  }
  return maxWindowEnergy;
}

export function calculateTankThermalReserve(
  hotWater: HotWaterData,
  weather: WeatherData,
  timeStepHours: number,
  storage: ThermalStorage,
): TankComparisonThermalReserve {
  const horizonHours = CONFIG.control.noSunshineHorizonHours;
  const horizonSteps = Math.round(horizonHours / timeStepHours);
  const indoorTemperature = 20;
  const heatingLimitTemperature = 15;
  const hasTemperatureData = weather.every(point => Number.isFinite(point.outdoorTemperature));
  const heatingProfile = hasTemperatureData &&
      weather.some(point => point.outdoorTemperature < heatingLimitTemperature)
    ? calculateSpaceHeatingProfile({
        heatedArea: 1,
        specificHeatingDemand: 1,
        indoorTemperature,
        heatingLimitTemperature,
      }, weather)
    : [];

  return {
    horizonHours,
    maxHotWaterDemandKWh: maxRollingThermalDemand(hotWater, horizonSteps),
    usableCapacityKWhPerLitre: calculateStorageCapacity(storage) / storage.volume,
    minimumTemperatureC: storage.minTemperature,
    maximumTemperatureC: storage.maxTemperature,
    // The heating profile is normalised to 1 kWh/year. The tank endpoints
    // repeat the same DHW profile daily, so its weekly sum is constant.
    spaceHeating: hasTemperatureData ? {
      peakWeekShare: maxRollingThermalDemand(heatingProfile, horizonSteps),
      indoorTemperatureC: indoorTemperature,
      heatingLimitTemperatureC: heatingLimitTemperature,
    } : undefined,
  };
}
