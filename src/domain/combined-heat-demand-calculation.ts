// src/domain/combined-heat-demand-calculation.ts

import type {
  HotWaterData,
} from "./hot-water";

import type {
  SpaceHeatingData,
} from "./space-heating";

import type {
  CombinedHeatDemand,
} from "./combined-heat-demand";


export function combineHeatDemand(
  hotWaterData: HotWaterData,
  heatingData: SpaceHeatingData,
): CombinedHeatDemand {

  if (
    hotWaterData.length !==
    heatingData.length
  ) {
    throw new Error(
      "Hot-water and space-heating data must contain the same number of time steps.",
    );
  }


  return hotWaterData.map(
    (hotWater, index) => {

      const heating =
        heatingData[index];

      const hotWaterDemand =
        hotWater.thermalEnergy;

      const spaceHeatingDemand =
        heating.thermalEnergy;

      return {
        timestamp:
          heating.timestamp,

        hotWaterDemand,

        spaceHeatingDemand,

        totalThermalDemand:
          hotWaterDemand +
          spaceHeatingDemand,
      };
    },
  );
}