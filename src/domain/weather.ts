export interface WeatherDataPoint {
  timestamp: string;

  outdoorTemperature: number; // °C
  irradiance: number;         // W/m²
  pvPower: number;            // kW
}

export type WeatherData = WeatherDataPoint[];