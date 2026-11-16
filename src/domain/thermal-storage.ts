// src/domain/thermal-storage.ts

export interface ThermalStorageInsulation {
  /**
   * Thermal conductivity.
   *
   * Unit:
   * W / (m K)
   */
  thermalConductivity: number;

  /**
   * Insulation thickness.
   *
   * Unit:
   * m
   */
  thickness: number;

  /**
   * Height of cylindrical water volume.
   *
   * Unit:
   * m
   */
  cylinderHeight: number;

  /**
   * Ambient room temperature.
   *
   * Unit:
   * °C
   */
  ambientTemperature: number;
}


export const DEFAULT_THERMAL_STORAGE_INSULATION:
  ThermalStorageInsulation = {

    // PUR
    thermalConductivity: 0.030,

    // Temporary model assumption:
    // 100 mm insulation
    thickness: 0.10,

    // User-defined default
    cylinderHeight: 1.70,

    // Typical technical room assumption
    ambientTemperature: 20,
  };


export interface ThermalStorage {
  volume: number;

  minTemperature: number;
  maxTemperature: number;
  initialTemperature: number;

  /**
   * Physical insulation model.
   *
   * If omitted, the default PUR model above
   * is used.
   */
  insulation?:
    ThermalStorageInsulation;

  /**
   * Legacy model.
   *
   * If this value is supplied AND insulation
   * is omitted, the old constant loss model
   * can still be used temporarily.
   *
   * Unit:
   * kWh_th / day
   */
  dailyHeatLoss?: number;
}


export interface ThermalStorageState {
  storedEnergy: number;
}


export interface ThermalStorageStepResult {
  state:
    ThermalStorageState;

  chargedEnergy:
    number;

  dischargedEnergy:
    number;

  rejectedEnergy:
    number;

  unmetDemand:
    number;

  heatLoss:
    number;
}