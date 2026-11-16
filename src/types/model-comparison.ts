// src/types/model-comparison.ts

export type ComparisonModelId =
  | "existing"
  | "quantumGravity"
  | "plasmaWave";

export interface MonthlyEnergyValues {
  month: number;
  gridImport: number;
  feedIn: number;
  selfConsumption: number;
}

export interface AnnualEnergySummary {
  gridImport: number;
  feedIn: number;
  selfConsumption: number;
  pvProduction: number;
  selfConsumptionRate: number;
}

export interface ModelComparisonResult {
  id: ComparisonModelId;
  name: string;
  monthly: MonthlyEnergyValues[];
  annual: AnnualEnergySummary;
}

export interface ModelComparisonResponse {
  fake: false;
  year: number;
  storageVolume: number;
  simulationDurationMs:
    number | null;
  generatedAt: string;
  models:
    ModelComparisonResult[];
}
