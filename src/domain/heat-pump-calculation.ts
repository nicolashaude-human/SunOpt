// src/domain/heat-pump-calculation.ts

import type {
  HeatPump,
} from "./heat-pump";

export interface HeatPumpStepResult {
  requestedThermalEnergy: number; // kWh_th

  deliveredThermalEnergy: number; // kWh_th
  unmetThermalEnergy: number;     // kWh_th

  electricalEnergy: number;       // kWh_el

  cop: number;
}

export function getHeatPumpCOP(
  heatPump: HeatPump,
): number {
  if (heatPump.copModel.type === "constant") {
    const cop = heatPump.copModel.cop;

    if (
      !Number.isFinite(cop) ||
      cop <= 0
    ) {
      throw new Error(
        "Heat pump COP must be greater than zero.",
      );
    }

    return cop;
  }

  throw new Error(
    "COP curve calculation is not implemented yet.",
  );
}


export function calculateHeatPumpStep(
  heatPump: HeatPump,
  requestedThermalEnergy: number,
  timeStepHours: number = 1,
): HeatPumpStepResult {

  if (requestedThermalEnergy < 0) {
    throw new Error(
      "Requested thermal energy cannot be negative.",
    );
  }

  if (timeStepHours <= 0) {
    throw new Error(
      "Time step must be greater than zero.",
    );
  }

  const cop =
    getHeatPumpCOP(heatPump);

  const maximumThermalEnergy =
    heatPump.thermalPower *
    timeStepHours;

  const deliveredThermalEnergy =
    Math.min(
      requestedThermalEnergy,
      maximumThermalEnergy,
    );

  const unmetThermalEnergy =
    requestedThermalEnergy -
    deliveredThermalEnergy;

  const electricalEnergy =
    deliveredThermalEnergy / cop;

  return {
    requestedThermalEnergy,

    deliveredThermalEnergy,
    unmetThermalEnergy,

    electricalEnergy,

    cop,
  };
}