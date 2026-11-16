// src/domain/thermal-storage-calculation.ts

import {
  type ThermalStorage,
  type ThermalStorageState,
  type ThermalStorageStepResult,
} from "./thermal-storage";

import {
  CONFIG,
} from "@/config/config";



export interface CylindricalStorageGeometry {
  volumeM3: number;

  height: number;

  innerRadius: number;

  outerRadius: number;

  sideArea: number;

  endArea: number;

  totalInnerArea: number;
}

// src/domain/thermal-storage-calculation.ts

export interface RequiredStorageEnergyBeforeStepResult {
  requiredEnergyBeforeStep: number;
  coolingLossEnergy: number;
  unavoidableThermalDeficit: number;
}


/**
 * Calculates the minimum storage energy required before
 * a storage step.
 *
 * This is the analytical inverse of the passive-loss and
 * DHW part of calculateThermalStorageStep().
 *
 * No numerical search is required.
 *
 * Energy values: kWh_th
 */
export function calculateRequiredStorageEnergyBeforeStep(
  storage: ThermalStorage,
  requiredEnergyAfterStep: number,
  thermalOutputEnergy: number,
  timeStepHours: number,
  maximumEnergy?: number,
): RequiredStorageEnergyBeforeStepResult {

  validateStorage(storage);

  if (timeStepHours <= 0) {
    throw new Error("Time step must be greater than zero.");
  }

  if (requiredEnergyAfterStep < 0) {
    throw new Error(
      "Required energy after step cannot be negative.",
    );
  }

  if (thermalOutputEnergy < 0) {
    throw new Error(
      "Thermal output energy cannot be negative.",
    );
  }

  const capacity = calculateStorageCapacity(storage);

  const energyLimit = Math.min(
    capacity,
    Math.max(0, maximumEnergy ?? capacity),
  );

  const requiredAfterLoss =
    requiredEnergyAfterStep + thermalOutputEnergy;

  if (requiredAfterLoss <= 0) {
    return {
      requiredEnergyBeforeStep: 0,
      coolingLossEnergy: 0,
      unavoidableThermalDeficit: 0,
    };
  }
  /*
   * ================================================================
   * LEGACY CONSTANT-LOSS MODEL
   * ================================================================
   */
  if (
    storage.insulation === undefined &&
    storage.dailyHeatLoss !== undefined
  ) {
    const requestedHeatLoss =
      storage.dailyHeatLoss * timeStepHours / 24;

    const requiredEnergy =
      requiredAfterLoss + requestedHeatLoss;

    if (requiredEnergy <= energyLimit) {
      return {
        requiredEnergyBeforeStep: requiredEnergy,
        coolingLossEnergy: requestedHeatLoss,
        unavoidableThermalDeficit: 0,
      };
    }

    const heatLoss = Math.min(
      energyLimit,
      requestedHeatLoss,
    );

    const energyAfterLoss = Math.max(
      0,
      energyLimit - heatLoss,
    );

    return {
      requiredEnergyBeforeStep: energyLimit,
      coolingLossEnergy: heatLoss,
      unavoidableThermalDeficit: Math.max(
        0,
        requiredAfterLoss - energyAfterLoss,
      ),
    };
  }


  /*
   * ================================================================
   * ANALYTICAL INSULATION MODEL
   * ================================================================
   *
   * Forward:
   *
   * T = Tmin + E / C
   *
   * Qloss = q * (T - Tamb)
   *
   * q = G * dt / 1000
   *
   * Required:
   *
   * E - Qloss = requiredAfterLoss
   *
   * Therefore:
   *
   * E =
   *   (
   *     requiredAfterLoss +
   *     q * (Tmin - Tamb)
   *   )
   *   /
   *   (1 - q / C)
   */

  const insulation =
    storage.insulation ?? CONFIG.storage.insulation;

  const energyPerKelvin =
    storage.volume *
    CONFIG.physics.waterDensityKgPerLiter *
    CONFIG.physics.waterSpecificHeatKJPerKgK /
    CONFIG.physics.kJPerKWh;

  const conductance =
    calculateStorageThermalConductance(storage);

  const lossEnergyPerKelvin =
    conductance * timeStepHours / 1000;

  const zeroLossTemperature =
    storage.minTemperature +
    requiredAfterLoss / energyPerKelvin;

  let requiredEnergy: number;

  if (
    zeroLossTemperature <= insulation.ambientTemperature
  ) {
    requiredEnergy = requiredAfterLoss;
  } else {
    const lossSlope =
      lossEnergyPerKelvin / energyPerKelvin;

    const denominator = 1 - lossSlope;

    if (denominator <= 0) {
      throw new Error(
        "Storage loss model cannot be inverted analytically.",
      );
    }

    const baseTemperatureDifference =
      storage.minTemperature -
      insulation.ambientTemperature;

    requiredEnergy =
      (
        requiredAfterLoss +
        lossEnergyPerKelvin * baseTemperatureDifference
      ) /
      denominator;
  }


  /*
   * ================================================================
   * REQUIREMENT FITS INTO STORAGE
   * ================================================================
   */
  if (requiredEnergy <= energyLimit) {
    const temperature =
      storage.minTemperature +
      requiredEnergy / energyPerKelvin;

    const requestedHeatLoss =
      calculateStorageHeatLossAtTemperature(
        storage,
        temperature,
        timeStepHours,
      );

    const heatLoss = Math.min(
      requiredEnergy,
      requestedHeatLoss,
    );

    return {
      requiredEnergyBeforeStep: requiredEnergy,
      coolingLossEnergy: heatLoss,
      unavoidableThermalDeficit: 0,
    };
  }


  /*
   * ================================================================
   * STORAGE LIMIT REACHED
   * ================================================================
   */
  const maximumTemperature =
    storage.minTemperature +
    energyLimit / energyPerKelvin;

  const requestedHeatLoss =
    calculateStorageHeatLossAtTemperature(
      storage,
      maximumTemperature,
      timeStepHours,
    );

  const heatLoss = Math.min(
    energyLimit,
    requestedHeatLoss,
  );

  const energyAfterLoss = Math.max(
    0,
    energyLimit - heatLoss,
  );

  const unavoidableThermalDeficit = Math.max(
    0,
    requiredAfterLoss - energyAfterLoss,
  );

  return {
    requiredEnergyBeforeStep: energyLimit,
    coolingLossEnergy: heatLoss,
    unavoidableThermalDeficit,
  };
}


/*
 * Compatibility wrapper.
 *
 * Can be removed once all old callers have been migrated.
 */
export function simulateStorageStep(
  storage: ThermalStorage,
  previousState: ThermalStorageState,
  chargeEnergy: number,
  demandEnergy: number,
  timeStepHours: number =
    CONFIG.simulation.timeStepHours,
): ThermalStorageStepResult {

  return calculateThermalStorageStep({
    storage,
    currentEnergy: previousState.storedEnergy,
    thermalInputEnergy: chargeEnergy,
    thermalOutputEnergy: demandEnergy,
    timeStepHours,
  });
}

export function calculateCylindricalStorageGeometry(
  storage: ThermalStorage,
): CylindricalStorageGeometry {

  const insulation =
    storage.insulation ??
    CONFIG.storage.insulation;


  const volumeM3 =
    storage.volume /
    1000;


  const height =
    insulation.cylinderHeight;


  const innerRadius =
    Math.sqrt(
      volumeM3 /
      (
        Math.PI *
        height
      ),
    );


  const outerRadius =
    innerRadius +
    insulation.thickness;


  const sideArea =
    2 *
    Math.PI *
    innerRadius *
    height;


  const endArea =
    2 *             // Top und bottom side
    Math.PI *
    innerRadius *
    innerRadius;


    return {
      volumeM3,
    
      height,
    
      innerRadius,
    
      outerRadius,
    
      sideArea,
    
      endArea,
    
      totalInnerArea:
        sideArea +
        endArea,
    };
}


/**
 * Overall thermal conductance of the insulated
 * cylindrical storage.
 *
 * Unit:
 * W/K
 *
 * Cylindrical radial conduction is used for
 * the side wall.
 *
 * Top and bottom are approximated as planar
 * insulation layers.
 */
export function calculateStorageThermalConductance(
  storage: ThermalStorage,
): number {

  const insulation =
    storage.insulation ??
    CONFIG.storage.insulation;


  const geometry =
    calculateCylindricalStorageGeometry(
      storage,
    );


  const lambda =
    insulation.thermalConductivity;


  const thickness =
    insulation.thickness;


  const sideConductance =
    (
      2 *
      Math.PI *
      lambda *
      geometry.height
    ) /
    Math.log(
      geometry.outerRadius /
      geometry.innerRadius,
    );


  const endConductance =
    lambda *
    geometry.endArea /
    thickness;


  return (
    sideConductance +
    endConductance
  );
}

// src/domain/thermal-storage-calculation.ts

export function calculateStorageHeatLossAtTemperature(
  storage: ThermalStorage,
  storageTemperature: number,
  timeStepHours: number,
): number {

  /*
   * Legacy fallback.
   */
  if (
    storage.insulation === undefined &&
    storage.dailyHeatLoss !== undefined
  ) {

    return (
      storage.dailyHeatLoss *
      timeStepHours /
      24
    );
  }


  const insulation =
    storage.insulation ??
    CONFIG.storage.insulation;


  const conductance =
    calculateStorageThermalConductance(
      storage,
    );


  const deltaTemperature =
    Math.max(
      0,
      storageTemperature -
        insulation.ambientTemperature,
    );


  const heatLossPowerW =
    conductance *
    deltaTemperature;


  return (
    heatLossPowerW *
    timeStepHours /
    1000
  );
}

// src/domain/thermal-storage-calculation.ts

export function calculatePassiveCoolingTimeHours(
  storage: ThermalStorage,
  startTemperature: number,
  endTemperature: number,
): number {

  if (
    startTemperature <=
    endTemperature
  ) {
    return 0;
  }


  /*
   * Legacy fallback.
   */
  if (
    storage.insulation === undefined &&
    storage.dailyHeatLoss !== undefined
  ) {

    const startEnergy =
    calculateStorageEnergyAtTemperature(
      storage,
      startTemperature,
    );


  const endEnergy =
    calculateStorageEnergyAtTemperature(
      storage,
      endTemperature,
    );


    const usableEnergy =
      startEnergy -
      endEnergy;


    return (
      usableEnergy /
      storage.dailyHeatLoss *
      24
    );
  }


  const insulation =
    storage.insulation ??
    CONFIG.storage.insulation;


  const ambientTemperature =
    insulation.ambientTemperature;


  if (
    endTemperature <=
    ambientTemperature
  ) {
    throw new Error(
      "End temperature must be above ambient temperature.",
    );
  }


  const conductance =
    calculateStorageThermalConductance(
      storage,
    );


  /*
   * Thermal capacitance of water.
   *
   * J/K
   */
  const thermalCapacity =
    (
      storage.volume /
      1000
    ) *
    1000 *
    4186;


  /*
   * Newton cooling:
   *
   * T(t)-T ambient =
   * (T0-T ambient) * exp(-G/C * t)
   */
  const timeSeconds =
    (
      thermalCapacity /
      conductance
    ) *
    Math.log(
      (
        startTemperature -
        ambientTemperature
      ) /
      (
        endTemperature -
        ambientTemperature
      ),
    );


  return (
    timeSeconds /
    3600
  );
}

/**
 * Calculates the usable thermal storage capacity
 * between minTemperature and maxTemperature.
 *
 * Result: kWh_th
 */
export function calculateStorageCapacity(
  storage: ThermalStorage,
): number {

  validateStorage(storage);

  const deltaTemperature =
    storage.maxTemperature -
    storage.minTemperature;


  return (
    storage.volume *
    CONFIG.physics.waterDensityKgPerLiter *
    CONFIG.physics.waterSpecificHeatKJPerKgK *
    deltaTemperature /
    CONFIG.physics.kJPerKWh
  );
}


/**
 * Creates the initial storage state from
 * initialTemperature.
 *
 * storedEnergy is measured relative to
 * minTemperature.
 *
 * Result: kWh_th
 */
export function createInitialStorageState(
  storage: ThermalStorage,
): ThermalStorageState {

  validateStorage(storage);


  const deltaTemperature =
    storage.initialTemperature -
    storage.minTemperature;


  const storedEnergy =
    storage.volume *
    CONFIG.physics.waterDensityKgPerLiter *
    CONFIG.physics.waterSpecificHeatKJPerKgK *
    deltaTemperature /
    CONFIG.physics.kJPerKWh;


  return {
    storedEnergy,
  };
}


/**
 * Calculates the current average storage temperature
 * from the stored thermal energy.
 *
 * This is currently a simplified fully mixed storage
 * model.
 *
 * Result: °C
 */
export function calculateStorageTemperature(
  storage: ThermalStorage,
  state: ThermalStorageState,
): number {

  validateStorage(storage);


  const energyPerKelvin =
    storage.volume *
    CONFIG.physics.waterDensityKgPerLiter *
    CONFIG.physics.waterSpecificHeatKJPerKgK /
    CONFIG.physics.kJPerKWh;


  if (energyPerKelvin <= 0) {
    throw new Error(
      "Storage energy per Kelvin must be greater than zero.",
    );
  }


  const temperature =
    storage.minTemperature +
    state.storedEnergy /
    energyPerKelvin;


  return Math.min(
    storage.maxTemperature,
    Math.max(
      storage.minTemperature,
      temperature,
    ),
  );
}


/**
 * Simulates one storage time step.
 *
 * Order:
 *
 * 1. storage heat loss
 * 2. charging
 * 3. thermal demand / discharge
 *
 * All energy values are kWh_th.
 */
// src/domain/thermal-storage-calculation.ts

export interface ThermalStorageStepInput {

  storage:
    ThermalStorage;

  /*
   * Current usable storage energy above minTemperature.
   *
   * kWh_th
   */
  currentEnergy:
    number;

  /*
   * Thermal energy entering the storage during this step.
   *
   * kWh_th
   */
  thermalInputEnergy:
    number;

  /*
   * Thermal energy requested from the storage during this step.
   *
   * kWh_th
   */
  thermalOutputEnergy:
    number;

  timeStepHours:
    number;
}


export function calculateThermalStorageStep(
  input: ThermalStorageStepInput,
): ThermalStorageStepResult {

  const {
    storage,
    currentEnergy,
    thermalInputEnergy,
    thermalOutputEnergy,
    timeStepHours,
  } = input;


  validateStorage(
    storage,
  );


  if (
    timeStepHours <= 0
  ) {

    throw new Error(
      "Time step must be greater than zero.",
    );
  }


  if (
    thermalInputEnergy < 0
  ) {

    throw new Error(
      "Thermal input energy cannot be negative.",
    );
  }


  if (
    thermalOutputEnergy < 0
  ) {

    throw new Error(
      "Thermal output energy cannot be negative.",
    );
  }


  const capacity =
    calculateStorageCapacity(
      storage,
    );


  let storedEnergy =
    Math.min(
      capacity,
      Math.max(
        0,
        currentEnergy,
      ),
    );


  /*
   * ================================================================
   * 1. PASSIVE STORAGE LOSS
   * ================================================================
   *
   * Exactly ONCE per physical simulation step.
   */
  const currentTemperature =
    calculateStorageTemperature(
      storage,
      {
        storedEnergy,
      },
    );


  const requestedHeatLoss =
    calculateStorageHeatLossAtTemperature(
      storage,
      currentTemperature,
      timeStepHours,
    );


  const heatLoss =
    Math.min(
      storedEnergy,
      requestedHeatLoss,
    );


  storedEnergy -=
    heatLoss;


  /*
   * ================================================================
   * 2. THERMAL OUTPUT / DHW
   * ================================================================
   *
   * Demand is taken before new HP charging.
   *
   * This preserves the current simulation semantics:
   * unmet current DHW can be supplied directly by the HP.
   */
  const dischargedEnergy =
    Math.min(
      thermalOutputEnergy,
      storedEnergy,
    );


  storedEnergy -=
    dischargedEnergy;


  const unmetDemand =
    Math.max(
      0,
      thermalOutputEnergy -
        dischargedEnergy,
    );


  /*
   * ================================================================
   * 3. THERMAL INPUT / HP CHARGE
   * ================================================================
   */
  const availableCapacity =
    Math.max(
      0,
      capacity -
        storedEnergy,
    );


  const chargedEnergy =
    Math.min(
      thermalInputEnergy,
      availableCapacity,
    );


  const rejectedEnergy =
    Math.max(
      0,
      thermalInputEnergy -
        chargedEnergy,
    );


  storedEnergy +=
    chargedEnergy;


  /*
   * ================================================================
   * RESULT
   * ================================================================
   */
  const state:
    ThermalStorageState = {

    storedEnergy:
      Math.min(
        capacity,
        Math.max(
          0,
          storedEnergy,
        ),
      ),

  };


  return {
    state,
    chargedEnergy,
    dischargedEnergy,
    rejectedEnergy,
    unmetDemand,
    heatLoss,
  };
}

/*
 * ================================================================
 * STORAGE VALIDATION
 * ================================================================
 */
function validateStorage(
  storage: ThermalStorage,
): void {

  if (
    storage.volume <= 0
  ) {

    throw new Error(
      "Storage volume must be greater than zero.",
    );
  }


  if (
    storage.maxTemperature <=
    storage.minTemperature
  ) {

    throw new Error(
      "Storage maximum temperature must be greater than minimum temperature.",
    );
  }


  if (
    storage.initialTemperature <
      storage.minTemperature ||
    storage.initialTemperature >
      storage.maxTemperature
  ) {

    throw new Error(
      "Storage initial temperature must be between minimum and maximum temperature.",
    );
  }


  if (
    storage.dailyHeatLoss !== undefined &&
    storage.dailyHeatLoss < 0
  ) {

    throw new Error(
      "Storage daily heat loss cannot be negative.",
    );
  }
}


/*
 * ================================================================
 * STORAGE ENERGY AT TEMPERATURE
 * ================================================================
 *
 * Usable storage energy relative to minTemperature.
 *
 * Unit: kWh_th
 */
export function calculateStorageEnergyAtTemperature(
  storage: ThermalStorage,
  temperature: number,
): number {

  validateStorage(
    storage,
  );


  const clampedTemperature =
    Math.min(
      storage.maxTemperature,
      Math.max(
        storage.minTemperature,
        temperature,
      ),
    );


  const deltaTemperature =
    clampedTemperature -
    storage.minTemperature;


  return (
    storage.volume *
    CONFIG.physics.waterDensityKgPerLiter *
    CONFIG.physics.waterSpecificHeatKJPerKgK *
    deltaTemperature /
    CONFIG.physics.kJPerKWh
  );
}