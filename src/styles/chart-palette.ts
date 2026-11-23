/** Chart series only: ordered Kenneth L. Kelly maximum-contrast colours.
 *  These are data-visualization colours, NOT application/status colours.
 *  Existing series keep their original assignments. */
export const KELLY_CHART_COLORS = [
  "#FFB300", // 1 vivid yellow
  "#803E75", // 2 strong purple
  "#FF6800", // 3 vivid orange
  "#A6BDD7", // 4 light blue
  "#C10020", // 5 vivid red
  "#CEA262", // 6 buff
  "#817066", // 7 medium gray
  "#007D34", // 8 vivid green
  "#F6768E", // 9 strong pink
  "#00538A", // 10 strong blue
  "#FF7A5C", // 11 yellowish pink
  "#53377A", // 12 violet
  "#FF8E00", // 13 orange yellow
  "#B32851", // 14 purplish red
  "#F4C800", // 15 greenish yellow
  "#7F180D", // 16 reddish brown
  "#93AA00", // 17 yellowish green
] as const;

export const MODEL_CHART_COLORS = {
  existing: KELLY_CHART_COLORS[0],
  quantumGravity: KELLY_CHART_COLORS[1],
  plasmaWave: KELLY_CHART_COLORS[2],
} as const;
