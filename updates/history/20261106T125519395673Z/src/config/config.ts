// src/config/config.ts

export const TANK_COMPARISON_CONFIG = {
  workerCount: 6,
} as const;



export const CONFIG = {

  physics: {

    /*
     * Water properties.
     */
    waterSpecificHeatKJPerKgK:
      4.186,

    waterDensityKgPerLiter:
      1,

    kJPerKWh:
      3600,

    secondsPerHour:
      3600,

  },


  simulation: {

    /*
     * 15-minute simulation step.
     */
    timeStepHours:
      0.25,

  },


  pv: {

    peakPower:
      10,

    azimuth:
      0,

    inclination:
      30,

    systemLoss:
      0.14,

  },


  hotWater: {

    persons:
      2,

    dailyConsumptionPerPerson:
      50,

    coldWaterTemperature:
      10,

    deliveryTemperature:
      60,

  },


  heatPump: {

    thermalPower:
      5,

    maxFlowTemperature:
      75,

    cop:
      3,

  },


  storage: {

    defaultVolume:
      3000,

    minTemperature:
      60,

    maxTemperature:
      75,

    initialTemperature:
      60,

    /*
     * Default PUR insulation.
     */
    insulation: {

      thermalConductivity:
        0.030,

      thickness:
        0.10,

      cylinderHeight:
        1.70,

      ambientTemperature:
        20,

    },

  },


  control: {
    minimumReserveHours: 2,
    forecastHorizonHours: 72,
  
    plasmaWavePlanningHorizonHours: 48,
  
    noSunshineHorizonHours: 7 * 24,
    noSunshineProtectedHorizonHours: 48,
    noSunshineInitialWaterlinePower: 10,
  
    pvTargetTemperature: 75,
    householdReservePower: 2,
    forecastSafetyFactor: 1.20,
  },


  debug: {

    postalCode:
      "8001",

    year:
      2023,

  },

} as const;


/*
 * Temporary compatibility object.
 *
 * Existing files can continue using the old property names
 * while we migrate them to CONFIG.
 *
 * There are NO duplicated numeric constants here.
 */
export const HOT_WATER_TEST_CONFIG = {

  timeStepHours:
    CONFIG.simulation.timeStepHours,

  userForecastHours:
    CONFIG.control
      .plasmaWavePlanningHorizonHours,

  householdReservePower:
    CONFIG.control
      .householdReservePower,

  forecastSafetyFactor:
    CONFIG.control
      .forecastSafetyFactor,

} as const;