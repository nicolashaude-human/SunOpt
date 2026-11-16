// src/domain/pv-system.ts

export interface PVSystem {
  peakPower: number; // kWp

  // PVGIS convention:
  // -180° = North
  //  -90° = East
  //    0° = South
  //   90° = West
  //  180° = North
  azimuth: number;

  // 0° = horizontal
  // 90° = vertical
  inclination: number;

  // System losses as fraction:
  // 0.14 = 14 %
  systemLoss: number;
}