// src/domain/hot-water-control.ts

export interface HotWaterControl {

  /**
   * Emergency reserve.
   *
   * Grid charging is allowed only if the
   * storage falls below this reserve.
   */
  minimumReserveHours: number;


  /**
   * Forecast horizon.
   *
   * Example:
   * 48 h
   */
  forecastHorizonHours: number;


  /**
   * Maximum desired storage temperature.
   *
   * Example:
   * 75 °C
   */
  pvTargetTemperature: number;


  /**
   * PV power reserved for household consumption.
   *
   * Only PV power above this value may be
   * used by the heat pump.
   *
   * Unit: kW_el
   *
   * Example:
   * 4 kW
   */
  householdReservePower: number;
}

export interface HotWaterControl {

  minimumReserveHours: number;

  forecastHorizonHours: number;

  pvTargetTemperature: number;

  householdReservePower: number;


  /**
   * Safety factor applied to predicted thermal
   * energy demand.
   *
   * Example:
   * 1.2 = 20 % forecast reserve.
   */
  forecastSafetyFactor: number;
}