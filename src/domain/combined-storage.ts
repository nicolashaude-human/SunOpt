// src/domain/combined-storage.ts

import type {
  ThermalStorage,
} from "./thermal-storage";


export interface CombinedThermalStorage {
  hotWaterZone: ThermalStorage;

  heatingZone: ThermalStorage;
}