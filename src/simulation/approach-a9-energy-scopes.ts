/*
 * Approach A9: application-supplied TypeScript SOURCE of the energy search.
 * This is NOT part of the generic compiler. It is appended to the project's
 * rust-acceleration-scopes.ts before generic AST lowering.
 *
 * Critical criterion: the compiler is ignorant of this algorithm and can
 * compile different supported functions just as it compiled G7/A8 fixtures.
 * The input ABI adapter must still map the controller's static/dynamic data.
 */
export interface A9EnergySearchInput {
  exportPower: Float64Array;
  currentIndex: number;
  horizonSteps: number;
  requestedElectricalEnergy: number;
  maximumHeatPumpElectricalPower: number;
  timeStepHours: number;
}

export function runA9EnergyCutLevel(input: A9EnergySearchInput): number {
  const {
    exportPower, currentIndex, horizonSteps, requestedElectricalEnergy,
    maximumHeatPumpElectricalPower, timeStepHours,
  } = input;

  if (exportPower.length === 0 || horizonSteps < 1 || horizonSteps > exportPower.length) {
    throw new Error('A9 invalid horizon/annual array');
  }

  let maximumExportPower = 0;
  for (let offset = 0; offset < horizonSteps; offset++) {
    const i = wrapIndex(currentIndex + offset, exportPower.length);
    maximumExportPower = Math.max(maximumExportPower, exportPower[i] ?? 0);
  }

  function electricalEnergyAtCut(cut: number): number {
    let energy = 0;
    for (let offset = 0; offset < horizonSteps; offset++) {
      const i = wrapIndex(currentIndex + offset, exportPower.length);
      const exportAtStep = exportPower[i] ?? 0;
      const hpElectricalPower = Math.min(
        maximumHeatPumpElectricalPower,
        Math.max(0, exportAtStep - cut),
      );
      energy += hpElectricalPower * timeStepHours;
    }
    return energy;
  }

  const maximumSchedulable = electricalEnergyAtCut(0);
  const target = Math.min(Math.max(0, requestedElectricalEnergy), maximumSchedulable);
  if (target <= 0 || maximumExportPower <= 0) return maximumExportPower;
  if (target >= maximumSchedulable) return 0;

  let lower = 0;
  let upper = maximumExportPower;
  for (let iteration = 0; iteration < 32; iteration++) {
    const middle = (lower + upper) / 2;
    const energy = electricalEnergyAtCut(middle);
    if (energy > target) lower = middle;
    else upper = middle;
  }
  return (lower + upper) / 2;
}
