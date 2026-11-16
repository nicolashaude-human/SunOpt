// src/domain/building.ts

export interface Building {
  heatedArea: number; // m²

  // Jahres-Heizwärmebedarf des Gebäudes
  specificHeatingDemand: number; // kWh/(m² a)

  indoorTemperature: number; // °C

  // Oberhalb dieser Temperatur wird nicht geheizt.
  heatingLimitTemperature: number; // °C

  heatingSystem: HeatingSystem;
}


export interface HeatingSystem {
  type:
    | "floor-heating"
    | "radiator"
    | "other";

  flowTemperature: number;   // °C
  returnTemperature: number; // °C
}