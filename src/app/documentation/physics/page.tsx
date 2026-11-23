import type { ReactNode } from "react";
import { SunOptDocHeader } from "@/components/ui/sunopt-documentation";

function Formula({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="sunopt-physics-equation" role="group" aria-label={label}>
      <math display="block" aria-label={label}>
        {children}
      </math>
    </div>
  );
}
const flow = [
  {
    title: "PV-Erzeugung und Haushaltsverbrauch",
    text: "Die Wetterdaten bestimmen, wie viel Solarstrom zu jeder Viertelstunde zur Verfügung steht. Die elektrische Haushaltslast hat zunächst Vorrang; übrig gebliebene Leistung kann für die Wärmepumpe genutzt werden.",
  },
  {
    title: "Warmwasserverbrauch",
    text: "Das Verbrauchsprofil beschreibt, wann dem Speicher Wärme entnommen wird. Diese Entnahmen sind wichtig, weil der Speicher trotzdem immer ausreichend warmes Wasser bereitstellen soll.",
  },
  {
    title: "Vorausschau des Reglers",
    text: "Der Regler untersucht den bevorstehenden Bedarf und die erwartete Solarproduktion. Er berücksichtigt auch, dass ein heisser Speicher nach und nach Energie an die Umgebung verliert.",
  },
  {
    title: "Waterline und Peak Shaving",
    text: "Aus den zeitlichen Vorgaben wird abgeleitet, wann und wie stark die Wärmepumpe laufen soll. So wird möglichst viel überschüssiger Solarstrom verwendet, ohne den zukünftigen Warmwasserbedarf zu gefährden.",
  },
  {
    title: "Wärmepumpe und Speicher",
    text: "Die Wärmepumpe wandelt elektrische in thermische Energie um und lädt den Wassertank. Gleichzeitig werden Wärme für das Zapfwasser entnommen und Speicherverluste berücksichtigt. Der verbleibende Energiezustand geht in den nächsten Zeitschritt ein.",
  },
  {
    title: "Jahresbilanz und Speicherauswahl",
    text: "Für jeden Tank wird die gesamte Jahresrechnung aufgestellt: Netzbezug, Einspeisung, Eigenverbrauch, Speicherverluste und Kosten. Daraus kann SunOpt verschiedene Speichergrössen vergleichen.",
  },
] as const;

export default function PhysicsDocumentationPage() {
  return (
    <main className="sunopt-doc-page mx-auto max-w-7xl px-4 py-10 sm:px-6">
      <SunOptDocHeader label="SunOpt · Dokumentation" title="Physikalische Zusammenhänge">
        SunOpt verknüpft Sonnenstrom, elektrische Leistung, Warmwasserbedarf und die Wärmeenergie im Speicher. Der folgende Ablauf zeigt zuerst den Energiefluss, dann die dafür verwendeten Gleichungen.
      </SunOptDocHeader>

      <section className="sunopt-panel p-5 sm:p-7" aria-label="Energiefluss">
        <h2 className="mb-3 text-2xl font-black text-[var(--sunopt-ink)]">Energiefluss durch das Modell</h2>
        <p className="mb-6 text-[var(--sunopt-muted)]">Die Pfeile zeigen die Reihenfolge, in der die Informationen und Energiezustände in der Simulation verarbeitet werden.</p>
        {flow.map((stage, index) => (
          <div key={stage.title}>
            <article className="sunopt-physics-flow-row">
              <div className="sunopt-flow-node">
                <strong>{String(index + 1).padStart(2, "0")} · {stage.title}</strong>
                <span>Schritt {index + 1} von {flow.length}</span>
              </div>
              <p className="leading-7 text-[var(--sunopt-muted)]">{stage.text}</p>
            </article>
            {index < flow.length - 1 && <div className="sunopt-flow-arrow" aria-hidden="true">↓</div>}
          </div>
        ))}
      </section>

      <section className="mt-8" aria-label="Formeln">
        <h2 className="mb-4 text-2xl font-black text-[var(--sunopt-ink)]">Die zugrunde liegenden Gleichungen</h2>
        <div className="grid gap-4 lg:grid-cols-2">
          <article className="sunopt-card p-5">
            <h3 className="mb-3 text-xl font-bold">PV-Energie im Zeitschritt</h3>
            <Formula label="Energie PV gleich Leistung PV mal Zeitintervall"><mrow>
              <msub><mi>E</mi><mi>PV</mi></msub><mo>=</mo><msub><mi>P</mi><mi>PV</mi></msub><mo>·</mo><mi>Δt</mi>
            </mrow></Formula>
            <p className="mt-3 text-[var(--sunopt-muted)]">Die produzierte Energie ergibt sich aus der momentanen Leistung und der Länge eines Zeitschritts. 15 Minuten sind 0,25 Stunden.</p>
          </article>
          <article className="sunopt-card p-5">
            <h3 className="mb-3 text-xl font-bold">Wärmepumpe</h3>
            <Formula label="Thermische Leistung gleich Leistungszahl mal elektrische Leistung"><mrow>
              <msub><mi>P</mi><mi>th</mi></msub><mo>=</mo><mi>COP</mi><mo>·</mo><msub><mi>P</mi><mi>el</mi></msub>
            </mrow></Formula>
            <p className="mt-3 text-[var(--sunopt-muted)]">Die Leistungszahl COP sagt, wie viel Wärme die Wärmepumpe je eingesetzter elektrischer Energie erzeugen kann. Sie hängt von den Betriebsbedingungen ab.</p>
          </article>
          <article className="sunopt-card p-5">
            <h3 className="mb-3 text-xl font-bold">Nutzbare Speicherenergie</h3>
            <Formula label="Nutzbare Energie gleich Wärmekapazität mal Temperaturdifferenz"><mrow>
              <mi>E</mi><mo>=</mo><mi>C</mi><mo>·</mo><mo>(</mo><mi>T</mi><mo>−</mo><msub><mi>T</mi><mi>min</mi></msub><mo>)</mo>
            </mrow></Formula>
            <p className="mt-3 text-[var(--sunopt-muted)]">Je wärmer das Wasser im Vergleich zur Mindesttemperatur ist, desto mehr nutzbare Wärmeenergie ist gespeichert.</p>
          </article>
          <article className="sunopt-card p-5">
            <h3 className="mb-3 text-xl font-bold">Wärmeverlust</h3>
            <Formula label="Wärmeverlust gleich Wärmeleitwert mal Temperaturdifferenz mal Zeitintervall"><mrow>
              <msub><mi>Q</mi><mi>loss</mi></msub><mo>=</mo><mi>G</mi><mo>·</mo><mo>(</mo><mi>T</mi><mo>−</mo><msub><mi>T</mi><mi>amb</mi></msub><mo>)</mo><mo>·</mo><mi>Δt</mi>
            </mrow></Formula>
            <p className="mt-3 text-[var(--sunopt-muted)]">Wärme fliesst vom Speicher zur kühleren Umgebung. Höhere Speichertemperaturen erhöhen diese Verluste.</p>
          </article>
          <article className="sunopt-card p-5">
            <h3 className="mb-3 text-xl font-bold">Energiebilanz des Speichers</h3>
            <Formula label="Energie im nächsten Schritt gleich Energie jetzt plus Wärmepumpenenergie minus Warmwasserentnahme minus Speicherverlust"><mrow>
              <msub><mi>E</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo><msub><mi>E</mi><mi>n</mi></msub><mo>+</mo><msub><mi>E</mi><mi>HP</mi></msub><mo>−</mo><msub><mi>E</mi><mi>DHW</mi></msub><mo>−</mo><msub><mi>Q</mi><mi>loss</mi></msub>
            </mrow></Formula>
            <p className="mt-3 text-[var(--sunopt-muted)]">Die Speicherenergie wächst durch die Wärmepumpe und sinkt durch Warmwasserverbrauch und Wärmeverluste. Die physikalischen Grenzen des Speichers werden zusätzlich eingehalten.</p>
          </article>
          <article className="sunopt-card p-5">
            <h3 className="mb-3 text-xl font-bold">Netzbezug</h3>
            <Formula label="Netzleistung ist das Maximum aus Null und Haushaltslast plus Wärmepumpenleistung minus PV-Leistung"><mrow>
              <msub><mi>P</mi><mi>grid</mi></msub><mo>=</mo><mi>max</mi><mo>(</mo><mn>0</mn><mo>,</mo><msub><mi>P</mi><mi>load</mi></msub><mo>+</mo><msub><mi>P</mi><mi>HP</mi></msub><mo>−</mo><msub><mi>P</mi><mi>PV</mi></msub><mo>)</mo>
            </mrow></Formula>
            <p className="mt-3 text-[var(--sunopt-muted)]">Fehlt Solarleistung, wird Strom aus dem Netz bezogen. Überschüsse können ins Netz eingespeist werden.</p>
          </article>
        </div>
      </section>

      <section className="mt-8 grid gap-4 md:grid-cols-3" aria-label="Modellgrenzen">
        {[{title:"Untergrenze",value:"60 °C",description:"Mindesttemperatur des Warmwasserspeichers im Modell"},
          {title:"Obergrenze",value:"75 °C",description:"Angenommene maximale Speichertemperatur"},
          {title:"Zeitschritt",value:"15 min",description:"35'040 gekoppelte Zustandsübergänge je Jahr"}].map((item) => (
          <article key={item.title} className="sunopt-card p-5">
            <h3 className="text-sm font-bold text-[var(--sunopt-accent-dark)]">{item.title}</h3>
            <p className="mt-2 text-3xl font-black">{item.value}</p>
            <p className="mt-2 text-sm text-[var(--sunopt-muted)]">{item.description}</p>
          </article>
        ))}
      </section>

      <section className="mt-8 sunopt-panel p-5 sm:p-7">
        <h2 className="text-2xl font-bold">Warum ein Tank zeitlich sequenziell berechnet wird</h2>
        <p className="mt-3 max-w-5xl leading-7 text-[var(--sunopt-muted)]">
          Der Speicherzustand nach Schritt n wird im folgenden Schritt erneut benötigt. Deshalb können die 35'040 Zeitpunkte eines Tanks nicht unabhängig voneinander berechnet werden. Verschiedene Tanks können hingegen gleichzeitig simuliert werden.
        </p>
      </section>

      <section className="mt-8 sunopt-card p-5 sm:p-7" aria-label="Legende">
        <h2 className="mb-4 text-2xl font-black">Legende der Formelzeichen</h2>
        <dl className="grid gap-x-8 gap-y-3 sm:grid-cols-2 lg:grid-cols-3">
          {[
            ["P", "Leistung (kW)"], ["E", "Energie (kWh)"], ["Q", "Wärmemenge bzw. Wärmeverlust (kWh)"],
            ["T", "Temperatur (°C)"], ["Δt", "Dauer eines Zeitschritts (h)"], ["COP", "Leistungszahl der Wärmepumpe (dimensionslos)"],
            ["C", "Thermische Speicherkapazität (kWh/K)"], ["G", "Wärmeleitwert des Speichers (kW/K)"],
            ["PV", "Photovoltaikanlage"], ["HP", "Wärmepumpe (Heat Pump)"], ["DHW", "Warmwasserbedarf (Domestic Hot Water)"],
            ["el / th", "elektrisch / thermisch"], ["load", "Elektrische Haushaltslast"], ["grid", "Netzbezug"],
            ["min / amb", "Mindesttemperatur / Umgebungstemperatur"], ["n / n+1", "Aktueller / nächster Zeitschritt"],
          ].map(([term, meaning]) => (
            <div key={term} className="border-b border-[var(--sunopt-border-soft)] pb-2">
              <dt className="font-bold text-[var(--sunopt-ink)]">{term}</dt>
              <dd className="text-sm text-[var(--sunopt-muted)]">{meaning}</dd>
            </div>
          ))}
        </dl>
      </section>
    </main>
  );
}
