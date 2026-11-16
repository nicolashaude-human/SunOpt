// src/domain/index.ts

export type { Location } from "./location";
export type { PVSystem } from "./pv-system";

export type {
  WeatherData,
  WeatherDataPoint,
} from "./weather";

export type {
  PVAnalysis,
} from "./pv-analysis";

export {
  analyzePVProduction,
} from "./pv-analysis";

export type {
  HotWaterDemand,
  HotWaterProfile,

  DailyHotWaterData,
  DailyHotWaterDataPoint,

  HotWaterData,
  HotWaterDataPoint,
} from "./hot-water";

export {
    calculateThermalEnergy,
    calculateDailyHotWaterDemand,
    calculateAnnualHotWaterDemand,
} from "./hot-water-calculation";
  
export {
    DEFAULT_HOT_WATER_PROFILE,
} from "./default-hot-water-profile";


export type {
  HeatPump,
  COPModel,
  COPPoint,
} from "./heat-pump";

export type {
  HeatPumpStepResult,
} from "./heat-pump-calculation";

export {
  getHeatPumpCOP,
  calculateHeatPumpStep,
} from "./heat-pump-calculation";

export type {
  CylindricalStorageGeometry,
} from "./thermal-storage-calculation";

export {
  calculateCylindricalStorageGeometry,
  calculateStorageThermalConductance,
  calculateStorageHeatLossAtTemperature,
  calculatePassiveCoolingTimeHours,
} from "./thermal-storage-calculation";

export {
  DEFAULT_THERMAL_STORAGE_INSULATION,
} from "./thermal-storage";

export type {
  ThermalStorageInsulation,
} from "./thermal-storage";

export type {
    ThermalStorage,
    ThermalStorageState,
    ThermalStorageStepResult,
  } from "./thermal-storage";
  
  export {
    calculateStorageCapacity,
    calculateStorageEnergyAtTemperature,
    createInitialStorageState,
    simulateStorageStep,
    calculateStorageTemperature,
  } from "./thermal-storage-calculation";
  

  export type {
    Building,
    HeatingSystem,
  } from "./building";
  
  export type {
    SpaceHeatingData,
    SpaceHeatingDataPoint,
  } from "./space-heating";
  
  export {
    calculateAnnualHeatingDemand,
    calculateSpaceHeatingProfile,
  } from "./space-heating-calculation";

export type {
  CombinedThermalStorage,
} from "./combined-storage";

export type {
  CombinedHeatDemand,
  CombinedHeatDemandPoint,
} from "./combined-heat-demand";

export {
  combineHeatDemand,
} from "./combined-heat-demand-calculation";

// src/domain/index.ts

export type {
  HotWaterControl,
} from "./hot-water-control";

export type {
  HotWaterForecastInput,
  HotWaterForecastTarget,
} from "./hot-water-forecast-control";

export {
  calculateHotWaterForecastTarget,
} from "./hot-water-forecast-control";