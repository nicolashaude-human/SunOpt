// src/simulation/combined-waterline-scheduler.ts

export interface CombinedWaterlineSchedulerInput {
  userForecastWaterlinePower: number;
  noSunshineWaterlinePower: number;
}

export interface CombinedWaterlineSchedulerResult {
  userForecastWaterlinePower: number;
  noSunshineWaterlinePower: number;
  effectiveWaterlinePower: number;
  limitingStrategy: "userForecast" | "noSunshine" | "equal";
}


/**
 * Combines the independent Waterline controllers.
 *
 * The lower Waterline always wins because it represents
 * the controller which requires earlier PV charging.
 *
 * All Waterline values are electrical power in kW_el.
 */
export function calculateCombinedWaterline(
  input: CombinedWaterlineSchedulerInput,
): CombinedWaterlineSchedulerResult {

  const userWaterline = input.userForecastWaterlinePower;
  const noSunshineWaterline = input.noSunshineWaterlinePower;

  if (!Number.isFinite(userWaterline) || userWaterline < 0) {
    throw new Error(
      "User-forecast Waterline must be finite and non-negative.",
    );
  }

  if (!Number.isFinite(noSunshineWaterline) || noSunshineWaterline < 0) {
    throw new Error(
      "No-sunshine Waterline must be finite and non-negative.",
    );
  }

  const effectiveWaterlinePower = Math.min(
    userWaterline,
    noSunshineWaterline,
  );

  let limitingStrategy:
    CombinedWaterlineSchedulerResult["limitingStrategy"];

  if (userWaterline < noSunshineWaterline) {
    limitingStrategy = "userForecast";
  } else if (noSunshineWaterline < userWaterline) {
    limitingStrategy = "noSunshine";
  } else {
    limitingStrategy = "equal";
  }

  return {
    userForecastWaterlinePower: userWaterline,
    noSunshineWaterlinePower: noSunshineWaterline,
    effectiveWaterlinePower,
    limitingStrategy,
  };
}