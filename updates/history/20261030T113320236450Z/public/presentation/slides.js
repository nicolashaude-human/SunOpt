// public/presentation/slides.js

/*
 * ============================================================
 * SUNOPT PRESENTATION PARAMETERS
 * ============================================================
 */
const PRESENTATION_CONFIG = {
  source: {
    postalCode: "8001",
    year: 2023,
    pvSystem: {
      peakPower: 10,
      azimuth: 0,
      inclination: 30,
      systemLoss: 0.14,
    },
  },

  solarAvailability: {
    // Conceptual 24 h wheel:
    // 12 h night + 8 h seasonal daylight shift + 4 h noon core.
    nightHours: 12,
    seasonalShiftHoursTotal: 8,
    coreStartHour: 10,
    coreEndHour: 14,

    // Data-derived classification inside the 10:00-14:00 core.
    clearReferencePercentile: 0.95,
    weakIrradianceThresholdFraction: 0.35,

    // Energy split used to show where annual kWh are produced.
    summerHalfYearMonths: [4, 5, 6, 7, 8, 9],
    winterHalfYearMonths: [10, 11, 12, 1, 2, 3],
  },

  dailyPower: {
    // Measured average daily profile is scaled so 12:00 equals 500 %.
    noonPowerPercent: 500,
    yAxisMaximumPercent: 550,
  },

  slides: [
    "Verfügbarkeit der Sonne",
    "Hot-Water Debug",
    "Modellvergleich",
    "Tankvergleich",
    "Softwareaufbau & Performance",
  ],
};


const root = document.getElementById("presentation-app");
const tabsElement = document.getElementById("slide-tabs");
const frameElement = document.getElementById("slide-frame");
const counterElement = document.getElementById("slide-counter");
const previousButton = document.getElementById("previous-slide");
const nextButton = document.getElementById("next-slide");

if (
  !root ||
  !tabsElement ||
  !frameElement ||
  !counterElement ||
  !previousButton ||
  !nextButton
) {
  throw new Error("SunOpt presentation HTML is incomplete.");
}

let currentSlideIndex = 0;
let solarAnalysis = null;
let solarAnalysisError = null;
let solarAnalysisPromise = null;


function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}


function round(value, digits = 1) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}


function percent(value, digits = 1) {
  return `${round(value, digits).toLocaleString("de-CH")} %`;
}


function energy(value) {
  return `${Math.round(value).toLocaleString("de-CH")} kWh`;
}


function toFiniteNumber(value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === "string") {
    const parsed = Number(value.trim().replace(",", "."));

    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  return null;
}


function percentile(values, q) {
  if (!Array.isArray(values) || values.length === 0) {
    return 0;
  }

  const sorted = [...values].sort((a, b) => a - b);
  const position = clamp(q, 0, 1) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const fraction = position - lower;

  if (lower === upper) {
    return sorted[lower];
  }

  return sorted[lower] * (1 - fraction) + sorted[upper] * fraction;
}


function median(values) {
  return percentile(values, 0.5);
}


function svgElement(name, attributes = {}) {
  const element = document.createElementNS(
    "http://www.w3.org/2000/svg",
    name,
  );

  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }

  return element;
}


function polarPoint(cx, cy, radius, angleDeg) {
  const angle = angleDeg * Math.PI / 180;

  return {
    x: cx + radius * Math.sin(angle),
    y: cy - radius * Math.cos(angle),
  };
}


function describeArc(cx, cy, radius, startAngle, endAngle) {
  const start = polarPoint(cx, cy, radius, startAngle);
  const end = polarPoint(cx, cy, radius, endAngle);
  const largeArcFlag = endAngle - startAngle > 180 ? 1 : 0;

  return [
    `M ${cx} ${cy}`,
    `L ${start.x} ${start.y}`,
    `A ${radius} ${radius} 0 ${largeArcFlag} 1 ${end.x} ${end.y}`,
    "Z",
  ].join(" ");
}


/*
 * ============================================================
 * ROBUST WEATHER PARSER
 * ============================================================
 */
function extractWeatherArray(payload) {
  if (Array.isArray(payload)) {
    return payload;
  }

  if (!payload || typeof payload !== "object") {
    throw new Error(
      "Weather API returned neither an object nor an array.",
    );
  }

  const directCandidates = [
    payload.weatherData,
    payload.data,
    payload.points,
    payload.values,
  ];

  for (const candidate of directCandidates) {
    if (Array.isArray(candidate)) {
      return candidate;
    }
  }

  if (
    payload.weatherData &&
    typeof payload.weatherData === "object"
  ) {
    const nestedCandidates = [
      payload.weatherData.data,
      payload.weatherData.points,
      payload.weatherData.values,
    ];

    for (const candidate of nestedCandidates) {
      if (Array.isArray(candidate)) {
        return candidate;
      }
    }
  }

  throw new Error(
    "Weather API response contains no recognizable weather array.",
  );
}


function parseTimestampParts(value) {
  if (value instanceof Date && Number.isFinite(value.getTime())) {
    return {
      epochMs: value.getTime(),
      month: value.getUTCMonth() + 1,
      hour:
        value.getUTCHours() +
        value.getUTCMinutes() / 60,
    };
  }

  if (typeof value === "number" && Number.isFinite(value)) {
    const epochMs =
      value < 1e12
        ? value * 1000
        : value;

    const date = new Date(epochMs);

    if (Number.isFinite(date.getTime())) {
      return {
        epochMs: date.getTime(),
        month: date.getUTCMonth() + 1,
        hour:
          date.getUTCHours() +
          date.getUTCMinutes() / 60,
      };
    }
  }

  if (typeof value !== "string") {
    return null;
  }

  const text = value.trim();

  // PVGIS native timestamp: YYYYMMDD:HHMM
  let match =
    text.match(
      /^(\d{4})(\d{2})(\d{2}):(\d{2})(\d{2})$/,
    );

  if (match) {
    const year = Number(match[1]);
    const month = Number(match[2]);
    const day = Number(match[3]);
    const hours = Number(match[4]);
    const minutes = Number(match[5]);

    return {
      epochMs:
        Date.UTC(
          year,
          month - 1,
          day,
          hours,
          minutes,
        ),
      month,
      hour:
        hours +
        minutes / 60,
    };
  }

  // ISO-like timestamp. Read month/hour from the text itself to avoid TZ drift.
  match =
    text.match(
      /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?$/,
    );

  if (match) {
    const month = Number(match[2]);
    const hours = Number(match[4]);
    const minutes = Number(match[5]);
    const date = new Date(text);

    return {
      epochMs:
        Number.isFinite(
          date.getTime(),
        )
          ? date.getTime()
          : null,
      month,
      hour:
        hours +
        minutes / 60,
    };
  }

  const date = new Date(text);

  if (Number.isFinite(date.getTime())) {
    return {
      epochMs: date.getTime(),
      month: date.getUTCMonth() + 1,
      hour:
        date.getUTCHours() +
        date.getUTCMinutes() / 60,
    };
  }

  return null;
}


function normalizeWeatherPoint(rawPoint, index) {
  if (!rawPoint || typeof rawPoint !== "object") {
    return null;
  }

  const timestamp =
    rawPoint.timestamp ??
    rawPoint.time ??
    rawPoint.datetime ??
    rawPoint.dateTime ??
    rawPoint.date ??
    null;

  const pvPower =
    toFiniteNumber(
      rawPoint.pvPower ??
      rawPoint.pv_power ??
      rawPoint.P ??
      rawPoint.power ??
      rawPoint.pv ??
      null,
    );

  if (pvPower == null) {
    return null;
  }

  const time =
    parseTimestampParts(
      timestamp,
    );

  if (!time) {
    return null;
  }

  return {
    index,
    timestamp,
    epochMs: time.epochMs,
    month: time.month,
    hour: time.hour,
    pvPower:
      Math.max(
        0,
        pvPower,
      ),
  };
}


function parseWeatherPayload(payload) {
  const raw =
    extractWeatherArray(
      payload,
    );

  const points =
    raw
      .map(
        normalizeWeatherPoint,
      )
      .filter(
        (point) =>
          point !== null,
      );

  if (points.length < 100) {
    throw new Error(
      `Weather parser found only ${points.length} usable points out of ${raw.length}.`,
    );
  }

  return points;
}


function estimateTimeStepHours(points) {
  const differences = [];

  for (
    let index = 1;
    index < points.length;
    index += 1
  ) {
    const previous =
      points[index - 1]
        .epochMs;

    const current =
      points[index]
        .epochMs;

    if (
      previous == null ||
      current == null
    ) {
      continue;
    }

    const hours =
      (current - previous) /
      3_600_000;

    if (
      hours > 0 &&
      hours <= 6
    ) {
      differences.push(
        hours,
      );
    }
  }

  if (
    differences.length >
    0
  ) {
    return median(
      differences,
    );
  }

  // 2023 fallback: handles both 8760 hourly and 35040 quarter-hour points.
  return 8760 /
    points.length;
}


/*
 * ============================================================
 * DATA-DERIVED SOLAR CLASSIFICATION
 * ============================================================
 */
function buildMeasuredDailyProfile(points) {
  const bins =
    Array.from(
      { length: 24 },
      () => ({
        sum: 0,
        count: 0,
      }),
    );

  for (
    const point of points
  ) {
    const hour =
      clamp(
        Math.floor(
          point.hour,
        ),
        0,
        23,
      );

    bins[hour].sum +=
      point.pvPower;

    bins[hour].count +=
      1;
  }

  const averages =
    bins.map(
      (
        bin,
        hour,
      ) => ({
        hour,
        power:
          bin.count > 0
            ? bin.sum /
              bin.count
            : 0,
      }),
    );

  const noonPower =
    averages[12]
      ?.power ??
    0;

  const fallbackPeak =
    Math.max(
      ...averages.map(
        (point) =>
          point.power,
      ),
      0,
    );

  const reference =
    noonPower > 0
      ? noonPower
      : fallbackPeak;

  const target =
    PRESENTATION_CONFIG
      .dailyPower
      .noonPowerPercent;

  const scale =
    reference > 0
      ? target /
        reference
      : 0;

  return averages.map(
    (point) => ({
      hour:
        point.hour,

      value:
        point.power *
        scale,

      rawPower:
        point.power,
    }),
  );
}


function analyseSolarAvailability(points) {
  const config =
    PRESENTATION_CONFIG
      .solarAvailability;

  const timeStepHours =
    estimateTimeStepHours(
      points,
    );

  const nightPercent =
    config.nightHours /
    24 *
    100;

  const seasonalTotalPercent =
    config
      .seasonalShiftHoursTotal /
    24 *
    100;

  const seasonalEachPercent =
    seasonalTotalPercent /
    2;

  const coreHours =
    config.coreEndHour -
    config.coreStartHour;

  const corePercent =
    coreHours /
    24 *
    100;

  const corePoints =
    points.filter(
      (point) =>
        point.hour >=
          config
            .coreStartHour &&
        point.hour <
          config
            .coreEndHour,
    );

  if (
    corePoints.length ===
    0
  ) {
    throw new Error(
      "No weather points fall inside the configured 10:00-14:00 core.",
    );
  }

  const corePowers =
    corePoints.map(
      (point) =>
        point.pvPower,
    );

  const clearReferencePower =
    percentile(
      corePowers,
      config
        .clearReferencePercentile,
    );

  if (
    !(
      clearReferencePower >
      0
    )
  ) {
    throw new Error(
      "PVGIS core-window reference power is zero.",
    );
  }

  const weakThresholdPower =
    clearReferencePower *
    config
      .weakIrradianceThresholdFraction;

  const weakCorePoints =
    corePoints.filter(
      (point) =>
        point.pvPower <
        weakThresholdPower,
    );

  const weakCoreFraction =
    weakCorePoints.length /
    corePoints.length;

  const weatherTotalPercent =
    corePercent *
    weakCoreFraction;

  const weatherEachPercent =
    weatherTotalPercent /
    2;

  const highOutputPercent =
    Math.max(
      0,
      corePercent -
        weatherTotalPercent,
    );

  const summerMonths =
    new Set(
      config
        .summerHalfYearMonths,
    );

  const winterMonths =
    new Set(
      config
        .winterHalfYearMonths,
    );

  let annualEnergy = 0;
  let summerEnergy = 0;
  let winterEnergy = 0;

  for (
    const point of points
  ) {
    const pointEnergy =
      point.pvPower *
      timeStepHours;

    annualEnergy +=
      pointEnergy;

    if (
      summerMonths.has(
        point.month,
      )
    ) {
      summerEnergy +=
        pointEnergy;
    }

    if (
      winterMonths.has(
        point.month,
      )
    ) {
      winterEnergy +=
        pointEnergy;
    }
  }

  const summerEnergyPercent =
    annualEnergy > 0
      ? summerEnergy /
        annualEnergy *
        100
      : 0;

  const winterEnergyPercent =
    annualEnergy > 0
      ? winterEnergy /
        annualEnergy *
        100
      : 0;

  return {
    pointCount:
      points.length,

    timeStepHours,

    nightPercent,

    seasonalTotalPercent,

    seasonalEachPercent,

    corePercent,

    weatherTotalPercent,

    weatherEachPercent,

    highOutputPercent,

    clearReferencePower,

    weakThresholdPower,

    weakCoreFraction,

    annualEnergy,

    summerEnergy,

    winterEnergy,

    summerEnergyPercent,

    winterEnergyPercent,

    dailyProfile:
      buildMeasuredDailyProfile(
        points,
      ),
  };
}


async function fetchSolarAnalysis() {
  if (
    solarAnalysisPromise
  ) {
    return solarAnalysisPromise;
  }

  solarAnalysisPromise =
    (async () => {
      const source =
        PRESENTATION_CONFIG
          .source;

      const response =
        await fetch(
          "/api/weather",
          {
            method:
              "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body:
              JSON.stringify({
                postalCode:
                  source
                    .postalCode,

                pvSystem:
                  source
                    .pvSystem,

                startYear:
                  source
                    .year,

                endYear:
                  source
                    .year,
              }),

            cache:
              "no-store",
          },
        );

      const responseText =
        await response.text();

      let payload;

      try {
        payload =
          JSON.parse(
            responseText,
          );
      } catch {
        throw new Error(
          `Weather API returned invalid JSON (HTTP ${response.status}).`,
        );
      }

      if (
        !response.ok
      ) {
        const message =
          payload &&
          typeof payload ===
            "object" &&
          typeof payload.error ===
            "string"
            ? payload.error
            : `Weather API failed with HTTP ${response.status}.`;

        throw new Error(
          message,
        );
      }

      const points =
        parseWeatherPayload(
          payload,
        );

      return analyseSolarAvailability(
        points,
      );
    })();

  try {
    solarAnalysis =
      await solarAnalysisPromise;

    solarAnalysisError =
      null;

    return solarAnalysis;

  } catch (
    error
  ) {
    solarAnalysisError =
      error instanceof Error
        ? error.message
        : String(error);

    throw error;
  }
}


/*
 * ============================================================
 * VISUALIZATION
 * ============================================================
 */
function fallbackAnalysis() {
  const config =
    PRESENTATION_CONFIG
      .solarAvailability;

  const nightPercent =
    config.nightHours /
    24 *
    100;

  const seasonalTotalPercent =
    config
      .seasonalShiftHoursTotal /
    24 *
    100;

  const corePercent =
    (
      config.coreEndHour -
      config.coreStartHour
    ) /
    24 *
    100;

  return {
    nightPercent,

    seasonalTotalPercent,

    seasonalEachPercent:
      seasonalTotalPercent /
      2,

    corePercent,

    weatherTotalPercent:
      0,

    weatherEachPercent:
      0,

    highOutputPercent:
      corePercent,

    clearReferencePower:
      0,

    weakThresholdPower:
      0,

    annualEnergy:
      0,

    summerEnergy:
      0,

    winterEnergy:
      0,

    summerEnergyPercent:
      0,

    winterEnergyPercent:
      0,

    dailyProfile:
      Array.from(
        { length: 24 },
        (
          _,
          hour,
        ) => ({
          hour,
          value: 0,
        }),
      ),
  };
}


function renderSolarWheel(
  container,
  analysis,
) {
  const svg =
    svgElement(
      "svg",
      {
        class:
          "solar-wheel",

        viewBox:
          "0 0 820 620",

        role:
          "img",

        "aria-label":
          "SunOpt Zeitverfügbarkeit der Solarenergie",
      },
    );

  const cx = 410;
  const cy = 305;
  const radius = 205;

  const seasonalEachDeg =
    analysis
      .seasonalEachPercent *
    3.6;

  const weatherEachDeg =
    analysis
      .weatherEachPercent *
    3.6;

  const highDeg =
    analysis
      .highOutputPercent *
    3.6;

  let cursor = 270;

  const topSlices = [
    {
      key:
        "seasonalMorning",

      label:
        "Saisonal · Morgen",

      subLabel:
        `im Winter nicht verfügbar · ${percent(analysis.seasonalEachPercent)}`,

      size:
        seasonalEachDeg,

      fill:
        "var(--sunopt-zero)",
    },
    {
      key:
        "weatherMorning",

      label:
        "Regen / Bewölkung",

      subLabel:
        `schwache Einstrahlung · ${percent(analysis.weatherEachPercent)}`,

      size:
        weatherEachDeg,

      fill:
        "var(--sunopt-rain)",
    },
    {
      key:
        "high",

      label:
        "Produktives Sonnenfenster",

      subLabel:
        percent(
          analysis
            .highOutputPercent,
        ),

      size:
        highDeg,

      fill:
        "var(--sunopt-high)",
    },
    {
      key:
        "weatherEvening",

      label:
        "Regen / Bewölkung",

      subLabel:
        `schwache Einstrahlung · ${percent(analysis.weatherEachPercent)}`,

      size:
        weatherEachDeg,

      fill:
        "var(--sunopt-rain)",
    },
    {
      key:
        "seasonalEvening",

      label:
        "Saisonal · Abend",

      subLabel:
        `im Winter nicht verfügbar · ${percent(analysis.seasonalEachPercent)}`,

      size:
        seasonalEachDeg,

      fill:
        "var(--sunopt-zero)",
    },
  ];

  const slices = [];

  for (
    const slice of topSlices
  ) {
    const start =
      cursor;

    const end =
      cursor +
      slice.size;

    cursor =
      end;

    if (
      slice.size >
      0.0001
    ) {
      slices.push({
        ...slice,
        start,
        end,
      });
    }
  }

  slices.push({
    key:
      "night",

    label:
      "Nacht",

    subLabel:
      `0 Sonne · ${percent(analysis.nightPercent)}`,

    start:
      90,

    end:
      270,

    fill:
      "var(--sunopt-night)",
  });

  for (
    const slice of slices
  ) {
    svg.appendChild(
      svgElement(
        "path",
        {
          d:
            describeArc(
              cx,
              cy,
              radius,
              slice.start,
              slice.end,
            ),

          fill:
            slice.fill,

          stroke:
            "#fffaf2",

          "stroke-width":
            3,
        },
      ),
    );
  }

  svg.appendChild(
    svgElement(
      "circle",
      {
        cx,
        cy,
        r:
          radius,

        fill:
          "none",

        stroke:
          "#8f806b",

        "stroke-width":
          1.2,
      },
    ),
  );

  const labelSettings = {
    seasonalMorning: {
      x: 105,
      y: 150,
      anchor: "start",
    },

    weatherMorning: {
      x: 110,
      y: 72,
      anchor: "start",
    },

    high: {
      x: 410,
      y: 38,
      anchor: "middle",
    },

    weatherEvening: {
      x: 710,
      y: 72,
      anchor: "end",
    },

    seasonalEvening: {
      x: 715,
      y: 150,
      anchor: "end",
    },

    night: {
      x: 410,
      y: 590,
      anchor: "middle",
    },
  };

  for (
    const slice of slices
  ) {
    const label =
      labelSettings[
        slice.key
      ];

    if (!label) {
      continue;
    }

    const mid =
      (
        slice.start +
        slice.end
      ) /
      2;

    const edge =
      polarPoint(
        cx,
        cy,
        radius + 4,
        mid,
      );

    const bend =
      polarPoint(
        cx,
        cy,
        radius + 34,
        mid,
      );

    const lineEndX =
      label.anchor ===
      "start"
        ? label.x - 8
        : label.anchor ===
          "end"
          ? label.x + 8
          : label.x;

    const lineEndY =
      label.y - 7;

    svg.appendChild(
      svgElement(
        "path",
        {
          d:
            `M ${edge.x} ${edge.y} L ${bend.x} ${bend.y} L ${lineEndX} ${lineEndY}`,

          class:
            "leader-line",
        },
      ),
    );

    const text =
      svgElement(
        "text",
        {
          x:
            label.x,

          y:
            label.y,

          "text-anchor":
            label.anchor,

          class:
            "wheel-label",
        },
      );

    text.textContent =
      slice.label;

    svg.appendChild(
      text,
    );

    const sub =
      svgElement(
        "text",
        {
          x:
            label.x,

          y:
            label.y +
            18,

          "text-anchor":
            label.anchor,

          class:
            "wheel-sub-label",
        },
      );

    sub.textContent =
      slice.subLabel;

    svg.appendChild(
      sub,
    );
  }

  const clockLabels = [
    {
      angle: 0,
      text: "12:00",
      dx: 0,
      dy: -4,
    },
    {
      angle: 90,
      text: "18:00",
      dx: 14,
      dy: 4,
    },
    {
      angle: 180,
      text: "00:00",
      dx: 0,
      dy: 18,
    },
    {
      angle: 270,
      text: "06:00",
      dx: -14,
      dy: 4,
    },
  ];

  for (
    const item of clockLabels
  ) {
    const point =
      polarPoint(
        cx,
        cy,
        radius + 12,
        item.angle,
      );

    const text =
      svgElement(
        "text",
        {
          x:
            point.x +
            item.dx,

          y:
            point.y +
            item.dy,

          "text-anchor":
            item.angle ===
            90
              ? "start"
              : item.angle ===
                270
                ? "end"
                : "middle",

          class:
            "clock-label",
        },
      );

    text.textContent =
      item.text;

    svg.appendChild(
      text,
    );
  }

  container.replaceChildren(
    svg,
  );
}


function renderPowerChart(
  container,
  analysis,
) {
  const config =
    PRESENTATION_CONFIG
      .dailyPower;

  const availability =
    PRESENTATION_CONFIG
      .solarAvailability;

  const svg =
    svgElement(
      "svg",
      {
        class:
          "power-chart",

        viewBox:
          "0 0 1000 380",

        role:
          "img",

        "aria-label":
          "Aus PVGIS berechnetes mittleres Tagesprofil",
      },
    );

  const margin = {
    left: 72,
    right: 30,
    top: 30,
    bottom: 62,
  };

  const width =
    1000 -
    margin.left -
    margin.right;

  const height =
    380 -
    margin.top -
    margin.bottom;

  const maxY =
    config
      .yAxisMaximumPercent;

  const x =
    (hour) =>
      margin.left +
      hour /
      24 *
      width;

  const y =
    (value) =>
      margin.top +
      (
        1 -
        clamp(
          value,
          0,
          maxY,
        ) /
        maxY
      ) *
      height;

  function addBand(
    startHour,
    endHour,
    className,
  ) {
    const left =
      x(
        startHour,
      );

    const right =
      x(
        endHour,
      );

    svg.appendChild(
      svgElement(
        "rect",
        {
          x:
            left,

          y:
            margin.top,

          width:
            Math.max(
              0,
              right - left,
            ),

          height,

          class:
            className,
        },
      ),
    );
  }

  addBand(
    0,
    6,
    "band-night",
  );

  addBand(
    18,
    24,
    "band-night",
  );

  addBand(
    6,
    availability
      .coreStartHour,
    "band-seasonal",
  );

  addBand(
    availability
      .coreEndHour,
    18,
    "band-seasonal",
  );

  const weatherEachHours =
    analysis
      .weatherEachPercent /
    100 *
    24;

  addBand(
    availability
      .coreStartHour,

    availability
      .coreStartHour +
      weatherEachHours,

    "band-weather",
  );

  addBand(
    availability
      .coreEndHour -
      weatherEachHours,

    availability
      .coreEndHour,

    "band-weather",
  );

  for (
    const value of
    [
      0,
      100,
      200,
      300,
      400,
      500,
    ]
  ) {
    const yy =
      y(value);

    svg.appendChild(
      svgElement(
        "line",
        {
          x1:
            margin.left,

          y1:
            yy,

          x2:
            margin.left +
            width,

          y2:
            yy,

          class:
            "grid-line",
        },
      ),
    );

    const label =
      svgElement(
        "text",
        {
          x:
            margin.left -
            12,

          y:
            yy +
            4,

          "text-anchor":
            "end",

          class:
            "chart-tick",
        },
      );

    label.textContent =
      `${value}%`;

    svg.appendChild(
      label,
    );
  }

  for (
    const hour of
    [
      0,
      6,
      10,
      12,
      14,
      18,
      24,
    ]
  ) {
    const xx =
      x(hour);

    svg.appendChild(
      svgElement(
        "line",
        {
          x1:
            xx,

          y1:
            margin.top,

          x2:
            xx,

          y2:
            margin.top +
            height,

          class:
            hour ===
            12
              ? "noon-line"
              : "grid-line",
        },
      ),
    );

    const label =
      svgElement(
        "text",
        {
          x:
            xx,

          y:
            margin.top +
            height +
            28,

          "text-anchor":
            "middle",

          class:
            "chart-tick",
        },
      );

    label.textContent =
      `${String(hour % 24).padStart(2, "0")}:00`;

    svg.appendChild(
      label,
    );
  }

  svg.appendChild(
    svgElement(
      "line",
      {
        x1:
          margin.left,

        y1:
          margin.top +
          height,

        x2:
          margin.left +
          width,

        y2:
          margin.top +
          height,

        class:
          "axis-line",
      },
    ),
  );

  svg.appendChild(
    svgElement(
      "line",
      {
        x1:
          margin.left,

        y1:
          margin.top,

        x2:
          margin.left,

        y2:
          margin.top +
          height,

        class:
          "axis-line",
      },
    ),
  );

  const points =
    analysis
      .dailyProfile ??
    [];

  if (
    points.length >
    0
  ) {
    const linePath =
      points
        .map(
          (
            point,
            index,
          ) => {
            const command =
              index ===
              0
                ? "M"
                : "L";

            return `${command} ${x(point.hour)} ${y(point.value)}`;
          },
        )
        .join(
          " ",
        );

    svg.appendChild(
      svgElement(
        "path",
        {
          d:
            linePath,

          class:
            "power-line",
        },
      ),
    );
  }

  const noonPoint =
    points.find(
      (point) =>
        point.hour ===
        12,
    );

  if (noonPoint) {
    svg.appendChild(
      svgElement(
        "circle",
        {
          cx:
            x(12),

          cy:
            y(
              noonPoint
                .value,
            ),

          r:
            6,

          fill:
            "var(--sunopt-accent)",

          stroke:
            "#fff",

          "stroke-width":
            2,
        },
      ),
    );

    const peakLabel =
      svgElement(
        "text",
        {
          x:
            x(12) +
            14,

          y:
            y(
              noonPoint
                .value,
            ) +
            5,

          class:
            "chart-title-value",
        },
      );

    peakLabel.textContent =
      `12:00 = ${config.noonPowerPercent} %`;

    svg.appendChild(
      peakLabel,
    );
  }

  const yLabel =
    svgElement(
      "text",
      {
        x:
          18,

        y:
          margin.top +
          height /
          2,

        transform:
          `rotate(-90 18 ${margin.top + height / 2})`,

        "text-anchor":
          "middle",

        class:
          "chart-label",
      },
    );

  yLabel.textContent =
    "relative PV-Leistung";

  svg.appendChild(
    yLabel,
  );

  const xLabel =
    svgElement(
      "text",
      {
        x:
          margin.left +
          width /
          2,

        y:
          370,

        "text-anchor":
          "middle",

        class:
          "chart-label",
      },
    );

  xLabel.textContent =
    "Tageszeit · gleiche Geometrie wie im Kuchen";

  svg.appendChild(
    xLabel,
  );

  container.replaceChildren(
    svg,
  );
}


/*
 * ============================================================
 * SLIDES
 * ============================================================
 */
function createSolarAvailabilitySlide() {
  const analysis =
    solarAnalysis ??
    fallbackAnalysis();

  const source =
    PRESENTATION_CONFIG
      .source;

  const availability =
    PRESENTATION_CONFIG
      .solarAvailability;

  const slide =
    document.createElement(
      "article",
    );

  slide.className =
    "slide";

  const sourceLine =
    `${source.postalCode} · ${source.year} · ${source.pvSystem.peakPower} kWp · Azimut ${source.pvSystem.azimuth}° · ${source.pvSystem.inclination}° Neigung · ${Math.round(source.pvSystem.systemLoss * 100)} % Verluste`;

  slide.innerHTML = `
    <p class="slide-kicker">Folie 1 · Ausgangslage</p>
    <h1 class="slide-title">Wann ist Solarenergie wirklich verfügbar?</h1>

    <p class="slide-lead">
      Die Geometrie ist vorgegeben: 12 h Nacht und 8 h saisonal verschobenes Tageslicht.
      Die verbleibenden Wetter- und Energieanteile werden direkt aus den SunOpt/PVGIS-Daten berechnet.
    </p>

    <div class="analysis-status ${solarAnalysisError ? "analysis-status-error" : ""}" role="status">
      ${
        solarAnalysis
          ? `Datenbasis: ${sourceLine} · ${solarAnalysis.pointCount.toLocaleString("de-CH")} Wetterpunkte`
          : solarAnalysisError
            ? `PVGIS-Auswertung nicht verfügbar: ${solarAnalysisError}`
            : "PVGIS-Daten werden ausgewertet …"
      }
    </div>

    <div class="solar-layout">
      <section class="diagram-card">
        <h2>Zeitliche Verfügbarkeit im 24-h-Kreis</h2>
        <div id="solar-wheel" class="solar-wheel-wrap"></div>
      </section>

      <aside class="note-card">
        <h2>Berechnete Einteilung</h2>

        <dl class="metric-list">
          <div class="metric-row">
            <dt>Nacht · 12 h</dt>
            <dd>${percent(analysis.nightPercent)}</dd>
          </div>

          <div class="metric-row">
            <dt>Saisonal verschoben · 8 h</dt>
            <dd>${percent(analysis.seasonalTotalPercent)}</dd>
          </div>

          <div class="metric-row">
            <dt>davon Morgen</dt>
            <dd>${percent(analysis.seasonalEachPercent)}</dd>
          </div>

          <div class="metric-row">
            <dt>davon Abend</dt>
            <dd>${percent(analysis.seasonalEachPercent)}</dd>
          </div>

          <div class="metric-row">
            <dt>Regen/Bewölkung/schwache Einstrahlung</dt>
            <dd>${solarAnalysis ? percent(analysis.weatherTotalPercent) : "…"}</dd>
          </div>

          <div class="metric-row">
            <dt>Produktives Sonnenfenster</dt>
            <dd>${solarAnalysis ? percent(analysis.highOutputPercent) : "…"}</dd>
          </div>
        </dl>

        <div class="energy-split">
          <h3>Woher kommen die kWh?</h3>

          <div class="energy-bars">
            <div class="energy-bar-row">
              <span>Apr–Sep</span>
              <div class="energy-bar-track">
                <div class="energy-bar-fill energy-bar-summer" style="width:${analysis.summerEnergyPercent}%"></div>
              </div>
              <strong>${solarAnalysis ? percent(analysis.summerEnergyPercent) : "…"}</strong>
            </div>

            <div class="energy-bar-row">
              <span>Okt–Mär</span>
              <div class="energy-bar-track">
                <div class="energy-bar-fill energy-bar-winter" style="width:${analysis.winterEnergyPercent}%"></div>
              </div>
              <strong>${solarAnalysis ? percent(analysis.winterEnergyPercent) : "…"}</strong>
            </div>
          </div>

          <p class="energy-caption">
            ${
              solarAnalysis
                ? `${energy(analysis.summerEnergy)} im Sommerhalbjahr gegenüber ${energy(analysis.winterEnergy)} im Winterhalbjahr.`
                : "Die Energieanteile erscheinen nach erfolgreichem Laden der PVGIS-Daten."
            }
          </p>
        </div>

        <details class="classification-details">
          <summary>Klassifikation</summary>

          <p>
            Wetter/schwache Einstrahlung wird nur im zentralen
            ${availability.coreStartHour}:00–${availability.coreEndHour}:00-Fenster bestimmt.
            Referenz ist das ${(availability.clearReferencePercentile * 100).toFixed(0)}. Perzentil
            der PV-Leistung; als schwach gilt weniger als
            ${(availability.weakIrradianceThresholdFraction * 100).toFixed(0)} % dieser Referenz.
          </p>

          ${
            solarAnalysis
              ? `<p>Referenzleistung: ${round(analysis.clearReferencePower, 2)} kW · Schwellwert: ${round(analysis.weakThresholdPower, 2)} kW.</p>`
              : ""
          }
        </details>
      </aside>
    </div>

    <section class="diagram-card power-card">
      <h2>Gemessenes mittleres Tagesprofil · auf 12:00 = ${PRESENTATION_CONFIG.dailyPower.noonPowerPercent} % normiert</h2>
      <div id="power-chart"></div>
    </section>
  `;

  requestAnimationFrame(
    () => {
      const wheel =
        slide.querySelector(
          "#solar-wheel",
        );

      const power =
        slide.querySelector(
          "#power-chart",
        );

      if (wheel) {
        renderSolarWheel(
          wheel,
          analysis,
        );
      }

      if (power) {
        renderPowerChart(
          power,
          analysis,
        );
      }
    },
  );

  return slide;
}


function createPlaceholderSlide(index) {
  const title =
    PRESENTATION_CONFIG
      .slides[index];

  const slide =
    document.createElement(
      "article",
    );

  slide.className =
    "slide";

  slide.innerHTML = `
    <p class="slide-kicker">Folie ${index + 1}</p>
    <h1 class="slide-title">${title}</h1>
    <p class="slide-lead">
      Diese Folie ist als Navigationsziel angelegt und wird mit den realen SunOpt-Daten aufgebaut.
    </p>
  `;

  return slide;
}


function createSlide(index) {
  return index === 0
    ? createSolarAvailabilitySlide()
    : createPlaceholderSlide(
        index,
      );
}


/*
 * ============================================================
 * NAVIGATION
 * ============================================================
 */
function renderTabs() {
  tabsElement.replaceChildren();

  PRESENTATION_CONFIG
    .slides
    .forEach(
      (
        title,
        index,
      ) => {
        const button =
          document.createElement(
            "button",
          );

        button.type =
          "button";

        button.className =
          "slide-tab";

        button.textContent =
          `${index + 1}. ${title}`;

        button.setAttribute(
          "aria-current",
          index ===
          currentSlideIndex
            ? "page"
            : "false",
        );

        button.addEventListener(
          "click",
          () =>
            showSlide(
              index,
            ),
        );

        tabsElement.appendChild(
          button,
        );
      },
    );
}


function showSlide(index) {
  const lastIndex =
    PRESENTATION_CONFIG
      .slides
      .length -
    1;

  currentSlideIndex =
    clamp(
      index,
      0,
      lastIndex,
    );

  renderTabs();

  frameElement.replaceChildren(
    createSlide(
      currentSlideIndex,
    ),
  );

  frameElement.scrollTop =
    0;

  counterElement.textContent =
    `${currentSlideIndex + 1} / ${PRESENTATION_CONFIG.slides.length}`;

  previousButton.disabled =
    currentSlideIndex ===
    0;

  nextButton.disabled =
    currentSlideIndex ===
    lastIndex;
}


previousButton.addEventListener(
  "click",
  () =>
    showSlide(
      currentSlideIndex -
        1,
    ),
);

nextButton.addEventListener(
  "click",
  () =>
    showSlide(
      currentSlideIndex +
        1,
    ),
);

root.addEventListener(
  "keydown",
  (event) => {
    const target =
      event.target;

    const tagName =
      target instanceof HTMLElement
        ? target.tagName
        : "";

    if (
      [
        "INPUT",
        "TEXTAREA",
        "SELECT",
        "BUTTON",
        "SUMMARY",
      ].includes(
        tagName,
      )
    ) {
      return;
    }

    if (
      event.key ===
      "ArrowLeft"
    ) {
      event.preventDefault();

      showSlide(
        currentSlideIndex -
          1,
      );
    }

    if (
      event.key ===
      "ArrowRight"
    ) {
      event.preventDefault();

      showSlide(
        currentSlideIndex +
          1,
      );
    }
  },
);


/*
 * ============================================================
 * STARTUP
 * ============================================================
 */
showSlide(0);

fetchSolarAnalysis()
  .then(
    () => {
      if (
        currentSlideIndex ===
        0
      ) {
        showSlide(0);
      }
    },
  )
  .catch(
    (error) => {
      console.error(
        "SunOpt presentation solar analysis failed:",
        error,
      );

      if (
        currentSlideIndex ===
        0
      ) {
        showSlide(0);
      }
    },
  );
