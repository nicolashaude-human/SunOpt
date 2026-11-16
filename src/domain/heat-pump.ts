// src/domain/heat-pump.ts

export interface HeatPump {
  thermalPower: number;       // kW_th
  maxFlowTemperature: number; // °C

  copModel: COPModel;
}

export type COPModel =
  | {
      type: "constant";
      cop: number;
    }
  | {
      type: "curve";
      points: COPPoint[];
    };

export interface COPPoint {
  sourceTemperature: number; // °C
  flowTemperature: number;   // °C
  cop: number;
}