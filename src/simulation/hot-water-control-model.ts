// src/simulation/hot-water-control-model.ts

import type {
  HeatPump,
  HotWaterControl,
  HotWaterData,
  ThermalStorage,
  ThermalStorageState,
  WeatherData,
} from "@/domain";

import type {
  HotWaterForecastTarget,
} from "@/domain";


export interface HotWaterControlModelInput {

  currentIndex:
    number;

  weatherData:
    WeatherData;

  hotWaterData:
    HotWaterData;

  heatPump:
    HeatPump;

  storage:
    ThermalStorage;

  control:
    HotWaterControl;

  timeStepHours:
    number;


  /*
   * Current storage state AFTER:
   *
   * - storage heat loss
   * - current DHW discharge
   *
   * This is essential for controllers that determine
   * charging from the actual tank content.
   */
  storageState:
    ThermalStorageState;
}


export type HotWaterControlModel =
  (
    input:
      HotWaterControlModelInput,
  ) =>
    HotWaterForecastTarget;