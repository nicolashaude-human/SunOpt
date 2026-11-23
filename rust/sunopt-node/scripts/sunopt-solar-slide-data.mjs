// PVGIS calculation restored from the 2026-10-30 presentation v6.
const nightHours = 12;
const seasonalHours = 8;
const coreStart = 10;
const coreEnd = 14;
const referencePercentile = 0.95;
const thresholdFraction = 0.35;
const hourMs = 3_600_000;

function percentile(values, fraction) {
  const sorted = values.slice().sort((a, b) => a - b);
  const position = fraction * (sorted.length - 1);
  const lower = Math.floor(position);
  const remainder = position - lower;
  return remainder === 0 ? sorted[lower] :
    sorted[lower] * (1 - remainder) + sorted[lower + 1] * remainder;
}

export function analyseSolarSlide(cache, rain) {
  const { source, points } = cache;
  const { year, presentationPeakPower: peakPower } = source;
  const start = Date.UTC(year, 0, 1);
  const hours = (Date.UTC(year + 1, 0, 1) - start) / hourMs;
  const first = Date.parse(points[0]?.timestamp);
  if (points.length !== hours || !(first >= start && first < start + hourMs) ||
      !Number.isFinite(peakPower) || peakPower <= 0) {
    throw new Error('Folie 1 benötigt ein vollständiges PVGIS-Jahr mit Stundenwerten.');
  }

  if (rain.year !== year || rain.latitude !== source.latitude || rain.longitude !== source.longitude ||
      rain.unit !== 'mm' || !['GMT', 'UTC'].includes(rain.timezone) ||
      Object.keys(rain.rainByDate).length !== hours / 24) {
    throw new Error('Die täglichen Regendaten müssen zum Ort und Jahr der PVGIS-Daten passen.');
  }

  const summer = { totalDays: 0, dryDays: 0, hourlySums: Array(24).fill(0) };
  const winter = { totalDays: 0, dryDays: 0, hourlySums: Array(24).fill(0) };
  let dryDay = false;
  const corePowers = [];
  let annualEnergy = 0;
  let summerEnergy = 0;
  let winterEnergy = 0;
  let maximumPower = 0;

  for (const [index, point] of points.entries()) {
    const timestamp = Date.parse(point.timestamp);
    if (timestamp !== first + index * hourMs || !Number.isFinite(point.pvPowerPerKwp)) {
      throw new Error(`Ungültiger PVGIS-Stundenwert an Position ${index}.`);
    }
    const date = new Date(timestamp);
    const hour = date.getUTCHours();
    const month = date.getUTCMonth() + 1;
    const isSummer = month >= 4 && month <= 9;
    const season = isSummer ? summer : winter;
    const power = Math.max(0, point.pvPowerPerKwp * peakPower);
    maximumPower = Math.max(maximumPower, power);
    if (hour === 0) {
      const day = date.toISOString().slice(0, 10);
      const rainMm = rain.rainByDate[day];
      if (!Number.isFinite(rainMm) || rainMm < 0) {
        throw new Error(`Fehlender oder ungültiger Regenwert für ${day}.`);
      }
      dryDay = rainMm === 0;
      season.totalDays++;
      if (dryDay) season.dryDays++;
    }
    if (dryDay) season.hourlySums[hour] += power;
    if (hour >= coreStart && hour < coreEnd) corePowers.push(power);
    annualEnergy += power; // An hourly sample represents one hour of energy.
    if (isSummer) summerEnergy += power;
    else winterEnergy += power;
  }

  const referencePower = percentile(corePowers, referencePercentile);
  if (!(referencePower > 0 && annualEnergy > 0)) {
    throw new Error('Die PVGIS-Daten enthalten keine positive Referenzleistung für Folie 1.');
  }
  const thresholdPower = referencePower * thresholdFraction;
  const weakCoreFraction = corePowers.filter(power => power < thresholdPower).length / corePowers.length;
  const corePercent = (coreEnd - coreStart) / 24 * 100;
  const weakPercent = corePercent * weakCoreFraction;
  const annualMeanPower = annualEnergy / hours;
  const percentOfAnnualMean = power => power / annualMeanPower * 100;
  if (summer.dryDays === 0 || winter.dryDays === 0) {
    throw new Error('Folie 1 benötigt regenfreie Tage in beiden Halbjahren.');
  }
  const profile = ({ dryDays, totalDays, hourlySums }) => ({
    dryDays,
    totalDays,
    values: hourlySums.map(sum => percentOfAnnualMean(sum / dryDays)),
  });

  return {
    source: { postalCode: source.postalCode, year, peakPower, azimuth: source.azimuth,
      inclination: source.inclination, systemLoss: source.systemLoss },
    pointCount: points.length,
    nightHours, seasonalHours, coreStart, coreEnd,
    nightPercent: nightHours / 24 * 100,
    seasonalPercent: seasonalHours / 24 * 100,
    weakPercent,
    productivePercent: corePercent - weakPercent,
    thresholdPower, annualEnergy, annualMeanPower, maximumPower, summerEnergy, winterEnergy,
    summerPercent: summerEnergy / annualEnergy * 100,
    winterPercent: winterEnergy / annualEnergy * 100,
    rainSource: rain.source,
    dryDayProfiles: { summer: profile(summer), winter: profile(winter) },
  };
}
