// public/presentation/slides.js

/*
 * ============================================================
 * SUNOPT PRESENTATION PARAMETERS
 * ============================================================
 *
 * Keep presentation-specific assumptions here so the visual can be tuned
 * without changing the rendering code below.
 */
const PRESENTATION_CONFIG = {
  solarAvailability: {
    // Share of the full time circle. Night deliberately occupies the lower half.
    nightPercent: 50,

    // Additional zero-sun share placed symmetrically left/right of the night half.
    winterZeroPercentEachSide: 15,

    // Provisional values: tune these when we extract the final values from data.
    morningEveningLowPercentTotal: 10,
    rainLowPercent: 5,

    // The renderer calculates the remaining percentage as high-output sun.
    highOutputPercent: null,

    labels: {
      night: "Nacht",
      winterMorning: "Winter · Morgen",
      winterEvening: "Winter · Abend",
      morningEvening: "Morgen / Abend",
      rain: "Regen / Bewölkung",
      high: "Hohe PV-Leistung",
    },
  },

  dailyPower: {
    // Schematic reference profile. Noon is explicitly fixed to 500 %.
    referenceEnergyKWh: 1,
    noonPowerPercent: 500,

    // Hours at which the schematic curve becomes > 0 / returns to 0.
    sunriseHour: 6,
    sunsetHour: 18,

    // > 1 makes the peak narrower; < 1 makes it broader.
    curveExponent: 1.45,

    // Resolution of the generated line.
    stepMinutes: 10,
  },

  slides: [
    "Verfügbarkeit der Sonne",
    "Jahressimulation",
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

let currentSlideIndex = 0;


function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}


function percent(value) {
  return `${Number(value.toFixed(1))} %`;
}


function normalizeAvailabilityConfig() {
  const config = PRESENTATION_CONFIG.solarAvailability;

  const fixed =
    config.nightPercent
    + 2 * config.winterZeroPercentEachSide
    + config.morningEveningLowPercentTotal
    + config.rainLowPercent;

  const high = config.highOutputPercent == null
    ? Math.max(0, 100 - fixed)
    : config.highOutputPercent;

  const total = fixed + high;

  if (Math.abs(total - 100) > 0.001) {
    console.warn(
      `SunOpt presentation availability slices total ${total} %, expected 100 %.`,
    );
  }

  return {
    ...config,
    highOutputPercent: high,
  };
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


function svgElement(name, attributes = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }
  return element;
}


function renderSolarWheel(container) {
  const config = normalizeAvailabilityConfig();
  const svg = svgElement("svg", {
    class: "solar-wheel",
    viewBox: "0 0 760 610",
    role: "img",
    "aria-label": "Schematische Verfügbarkeit der Sonne über die Zeit",
  });

  const cx = 380;
  const cy = 300;
  const radius = 205;

  /*
   * Angle convention: 0° = 12 o'clock, 90° = 3 o'clock,
   * 180° = 6 o'clock, 270° = 9 o'clock.
   * Night is intentionally the full lower half: 90°..270°.
   */
  const sideWinterDegrees = config.winterZeroPercentEachSide * 3.6;
  const morningEveningTotalDegrees = config.morningEveningLowPercentTotal * 3.6;
  const morningEveningEachDegrees = morningEveningTotalDegrees / 2;
  const rainDegrees = config.rainLowPercent * 3.6;
  const highDegrees = config.highOutputPercent * 3.6;

  const slices = [
    {
      key: "winterMorning",
      label: config.labels.winterMorning,
      subLabel: `0 kWh · ${percent(config.winterZeroPercentEachSide)}`,
      start: 270,
      end: 270 + sideWinterDegrees,
      fill: "var(--sunopt-zero)",
    },
    {
      key: "morning",
      label: "Morgen",
      subLabel: `> 0 kWh · ${percent(config.morningEveningLowPercentTotal / 2)}`,
      start: 270 + sideWinterDegrees,
      end: 270 + sideWinterDegrees + morningEveningEachDegrees,
      fill: "var(--sunopt-low)",
    },
    {
      key: "rain",
      label: config.labels.rain,
      subLabel: `> 0 kWh · ${percent(config.rainLowPercent)}`,
      start: 270 + sideWinterDegrees + morningEveningEachDegrees,
      end: 270 + sideWinterDegrees + morningEveningEachDegrees + rainDegrees,
      fill: "var(--sunopt-rain)",
    },
    {
      key: "high",
      label: config.labels.high,
      subLabel: `${percent(config.highOutputPercent)}`,
      start: 270 + sideWinterDegrees + morningEveningEachDegrees + rainDegrees,
      end: 270 + sideWinterDegrees + morningEveningEachDegrees + rainDegrees + highDegrees,
      fill: "var(--sunopt-high)",
    },
    {
      key: "evening",
      label: "Abend",
      subLabel: `> 0 kWh · ${percent(config.morningEveningLowPercentTotal / 2)}`,
      start: 450 - sideWinterDegrees - morningEveningEachDegrees,
      end: 450 - sideWinterDegrees,
      fill: "var(--sunopt-low)",
    },
    {
      key: "winterEvening",
      label: config.labels.winterEvening,
      subLabel: `0 kWh · ${percent(config.winterZeroPercentEachSide)}`,
      start: 450 - sideWinterDegrees,
      end: 450,
      fill: "var(--sunopt-zero)",
    },
    {
      key: "night",
      label: config.labels.night,
      subLabel: `0 Sonne · ${percent(config.nightPercent)}`,
      start: 90,
      end: 270,
      fill: "var(--sunopt-night)",
    },
  ];

  // If provisional values overlap, draw top-half pieces in their intended order
  // and let the parameter warning reveal the mismatch rather than silently hiding it.
  for (const slice of slices) {
    const path = svgElement("path", {
      d: describeArc(cx, cy, radius, slice.start, slice.end),
      fill: slice.fill,
      stroke: "#fffaf2",
      "stroke-width": 3,
    });
    svg.appendChild(path);
  }

  const circle = svgElement("circle", {
    cx,
    cy,
    r: radius,
    fill: "none",
    stroke: "#8f806b",
    "stroke-width": 1.2,
  });
  svg.appendChild(circle);

  const labelSettings = {
    winterMorning: { x: 95, y: 155, anchor: "start" },
    morning: { x: 110, y: 80, anchor: "start" },
    rain: { x: 290, y: 45, anchor: "middle" },
    high: { x: 475, y: 45, anchor: "middle" },
    evening: { x: 650, y: 80, anchor: "end" },
    winterEvening: { x: 665, y: 155, anchor: "end" },
    night: { x: 380, y: 585, anchor: "middle" },
  };

  for (const slice of slices) {
    const mid = (slice.start + slice.end) / 2;
    const edge = polarPoint(cx, cy, radius + 4, mid);
    const bend = polarPoint(cx, cy, radius + 34, mid);
    const label = labelSettings[slice.key];

    const lineEndX = label.anchor === "start"
      ? label.x - 8
      : label.anchor === "end"
        ? label.x + 8
        : label.x;
    const lineEndY = label.y - 7;

    const leader = svgElement("path", {
      d: `M ${edge.x} ${edge.y} L ${bend.x} ${bend.y} L ${lineEndX} ${lineEndY}`,
      class: "leader-line",
    });
    svg.appendChild(leader);

    const text = svgElement("text", {
      x: label.x,
      y: label.y,
      "text-anchor": label.anchor,
      class: "wheel-label",
    });
    text.textContent = slice.label;
    svg.appendChild(text);

    const sub = svgElement("text", {
      x: label.x,
      y: label.y + 18,
      "text-anchor": label.anchor,
      class: "wheel-sub-label",
    });
    sub.textContent = slice.subLabel;
    svg.appendChild(sub);
  }

  const clockLabels = [
    { angle: 0, text: "12:00", dx: 0, dy: -2 },
    { angle: 90, text: "18:00", dx: 14, dy: 4 },
    { angle: 180, text: "00:00", dx: 0, dy: 16 },
    { angle: 270, text: "06:00", dx: -14, dy: 4 },
  ];

  for (const item of clockLabels) {
    const point = polarPoint(cx, cy, radius + 12, item.angle);
    const text = svgElement("text", {
      x: point.x + item.dx,
      y: point.y + item.dy,
      "text-anchor": item.angle === 90 ? "start" : item.angle === 270 ? "end" : "middle",
      class: "clock-label",
    });
    text.textContent = item.text;
    svg.appendChild(text);
  }

  container.replaceChildren(svg);
}


function buildDailyPowerPoints() {
  const config = PRESENTATION_CONFIG.dailyPower;
  const points = [];
  const steps = Math.round(24 * 60 / config.stepMinutes);

  for (let i = 0; i <= steps; i += 1) {
    const hour = i * config.stepMinutes / 60;
    let value = 0;

    if (hour >= config.sunriseHour && hour <= config.sunsetHour) {
      const phase = (hour - config.sunriseHour)
        / (config.sunsetHour - config.sunriseHour);
      const base = Math.max(0, Math.sin(Math.PI * phase));
      value = config.noonPowerPercent * Math.pow(base, config.curveExponent);
    }

    points.push({ hour, value });
  }

  return points;
}


function renderPowerChart(container) {
  const config = PRESENTATION_CONFIG.dailyPower;
  const svg = svgElement("svg", {
    class: "power-chart",
    viewBox: "0 0 1000 360",
    role: "img",
    "aria-label": "Schematisches PV-Leistungsprofil über 24 Stunden mit 500 Prozent um 12 Uhr",
  });

  const margin = { left: 72, right: 30, top: 30, bottom: 55 };
  const width = 1000 - margin.left - margin.right;
  const height = 360 - margin.top - margin.bottom;
  const maxY = Math.max(500, config.noonPowerPercent);

  const x = (hour) => margin.left + hour / 24 * width;
  const y = (value) => margin.top + (1 - value / maxY) * height;

  for (const value of [0, 100, 200, 300, 400, 500]) {
    const yy = y(value);
    svg.appendChild(svgElement("line", {
      x1: margin.left,
      y1: yy,
      x2: margin.left + width,
      y2: yy,
      class: "grid-line",
    }));

    const label = svgElement("text", {
      x: margin.left - 12,
      y: yy + 4,
      "text-anchor": "end",
      class: "chart-tick",
    });
    label.textContent = `${value}%`;
    svg.appendChild(label);
  }

  for (const hour of [0, 6, 12, 18, 24]) {
    const xx = x(hour);
    svg.appendChild(svgElement("line", {
      x1: xx,
      y1: margin.top,
      x2: xx,
      y2: margin.top + height,
      class: "grid-line",
    }));

    const label = svgElement("text", {
      x: xx,
      y: margin.top + height + 28,
      "text-anchor": "middle",
      class: "chart-tick",
    });
    label.textContent = `${String(hour % 24).padStart(2, "0")}:00`;
    svg.appendChild(label);
  }

  svg.appendChild(svgElement("line", {
    x1: margin.left,
    y1: margin.top + height,
    x2: margin.left + width,
    y2: margin.top + height,
    class: "axis-line",
  }));

  svg.appendChild(svgElement("line", {
    x1: margin.left,
    y1: margin.top,
    x2: margin.left,
    y2: margin.top + height,
    class: "axis-line",
  }));

  const noonX = x(12);
  svg.appendChild(svgElement("line", {
    x1: noonX,
    y1: margin.top,
    x2: noonX,
    y2: margin.top + height,
    class: "noon-line",
  }));

  const points = buildDailyPowerPoints();
  const linePath = points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${x(point.hour)} ${y(point.value)}`)
    .join(" ");

  const areaPath = [
    `M ${x(points[0].hour)} ${y(0)}`,
    ...points.map((point) => `L ${x(point.hour)} ${y(point.value)}`),
    `L ${x(points[points.length - 1].hour)} ${y(0)}`,
    "Z",
  ].join(" ");

  svg.appendChild(svgElement("path", { d: areaPath, class: "power-area" }));
  svg.appendChild(svgElement("path", { d: linePath, class: "power-line" }));

  const peak = svgElement("circle", {
    cx: noonX,
    cy: y(config.noonPowerPercent),
    r: 6,
    fill: "var(--sunopt-accent)",
    stroke: "#fff",
    "stroke-width": 2,
  });
  svg.appendChild(peak);

  const peakLabel = svgElement("text", {
    x: noonX + 14,
    y: y(config.noonPowerPercent) + 5,
    class: "chart-title-value",
  });
  peakLabel.textContent = `12:00 = ${config.noonPowerPercent} %`;
  svg.appendChild(peakLabel);

  const xLabel = svgElement("text", {
    x: margin.left + width / 2,
    y: 347,
    "text-anchor": "middle",
    class: "chart-label",
  });
  xLabel.textContent = "Tageszeit";
  svg.appendChild(xLabel);

  const yLabel = svgElement("text", {
    x: 18,
    y: margin.top + height / 2,
    transform: `rotate(-90 18 ${margin.top + height / 2})`,
    "text-anchor": "middle",
    class: "chart-label",
  });
  yLabel.textContent = "relative PV-Leistung";
  svg.appendChild(yLabel);

  container.replaceChildren(svg);
}


function createSolarAvailabilitySlide() {
  const config = normalizeAvailabilityConfig();
  const slide = document.createElement("article");
  slide.className = "slide";

  slide.innerHTML = `
    <p class="slide-kicker">Folie 1 · Ausgangslage</p>
    <h1 class="slide-title">Wann ist die Sonne wirklich verfügbar?</h1>
    <p class="slide-lead">
      Solarenergie ist reichlich vorhanden, aber zeitlich konzentriert. Die Darstellung ist bewusst
      schematisch; alle Prozentwerte stehen am Kopf von <code>slides.js</code> und können direkt angepasst werden.
    </p>

    <div class="solar-layout">
      <section class="diagram-card">
        <h2>Zeitliche Verfügbarkeit</h2>
        <div id="solar-wheel" class="solar-wheel-wrap"></div>
      </section>

      <aside class="note-card">
        <h2>Aktuelle Parameter</h2>
        <dl class="metric-list">
          <div class="metric-row"><dt>Nacht · 0 Sonne</dt><dd>${percent(config.nightPercent)}</dd></div>
          <div class="metric-row"><dt>Winter · links · 0 Sonne</dt><dd>${percent(config.winterZeroPercentEachSide)}</dd></div>
          <div class="metric-row"><dt>Winter · rechts · 0 Sonne</dt><dd>${percent(config.winterZeroPercentEachSide)}</dd></div>
          <div class="metric-row"><dt>Morgen + Abend · niedrige Leistung</dt><dd>${percent(config.morningEveningLowPercentTotal)}</dd></div>
          <div class="metric-row"><dt>Regen / Bewölkung · niedrige Leistung</dt><dd>${percent(config.rainLowPercent)}</dd></div>
          <div class="metric-row"><dt>verbleibender Hochleistungsanteil</dt><dd>${percent(config.highOutputPercent)}</dd></div>
        </dl>
      </aside>
    </div>

    <section class="diagram-card power-card">
      <h2>Schematisches Tagesprofil · Referenz ${PRESENTATION_CONFIG.dailyPower.referenceEnergyKWh} kWh</h2>
      <div id="power-chart"></div>
    </section>
  `;

  requestAnimationFrame(() => {
    const wheel = slide.querySelector("#solar-wheel");
    const power = slide.querySelector("#power-chart");
    if (wheel) renderSolarWheel(wheel);
    if (power) renderPowerChart(power);
  });

  return slide;
}


function createPlaceholderSlide(index) {
  const title = PRESENTATION_CONFIG.slides[index];
  const slide = document.createElement("article");
  slide.className = "slide";
  slide.innerHTML = `
    <p class="slide-kicker">Folie ${index + 1}</p>
    <h1 class="slide-title">${title}</h1>
    <p class="slide-lead">Diese Folie ist im Generator bereits angelegt und wird als Nächstes mit den realen SunOpt-Daten aufgebaut.</p>
    <div class="placeholder-grid">
      <div class="placeholder-card"><strong>Inhalt</strong>Reale Diagramme und Kennzahlen aus SunOpt.</div>
      <div class="placeholder-card"><strong>Interaktion</strong>Navigation und spätere Detailansichten bleiben in derselben HTML/JS-Struktur.</div>
      <div class="placeholder-card"><strong>Status</strong>Platzhalter – bewusst noch ohne erfundene Daten.</div>
    </div>
  `;
  return slide;
}


function createSlide(index) {
  return index === 0
    ? createSolarAvailabilitySlide()
    : createPlaceholderSlide(index);
}


function renderTabs() {
  tabsElement.replaceChildren();

  PRESENTATION_CONFIG.slides.forEach((title, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "slide-tab";
    button.textContent = `${index + 1}. ${title}`;
    button.setAttribute("aria-current", index === currentSlideIndex ? "page" : "false");
    button.addEventListener("click", () => showSlide(index));
    tabsElement.appendChild(button);
  });
}


function showSlide(index) {
  const lastIndex = PRESENTATION_CONFIG.slides.length - 1;
  currentSlideIndex = clamp(index, 0, lastIndex);

  renderTabs();
  frameElement.replaceChildren(createSlide(currentSlideIndex));
  frameElement.scrollTop = 0;

  counterElement.textContent = `${currentSlideIndex + 1} / ${PRESENTATION_CONFIG.slides.length}`;
  previousButton.disabled = currentSlideIndex === 0;
  nextButton.disabled = currentSlideIndex === lastIndex;
}


previousButton.addEventListener("click", () => showSlide(currentSlideIndex - 1));
nextButton.addEventListener("click", () => showSlide(currentSlideIndex + 1));

root.addEventListener("keydown", (event) => {
  const tagName = event.target instanceof HTMLElement ? event.target.tagName : "";
  if (["INPUT", "TEXTAREA", "SELECT"].includes(tagName)) return;

  if (event.key === "ArrowLeft") {
    event.preventDefault();
    showSlide(currentSlideIndex - 1);
  }

  if (event.key === "ArrowRight") {
    event.preventDefault();
    showSlide(currentSlideIndex + 1);
  }
});

showSlide(0);
