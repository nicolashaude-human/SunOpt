// src/app/documentation/physics/page.tsx

const equations = [
  {
    name: "PV-Energie im Zeitschritt",
    equation: "E_PV = P_PV · Δt",
    text:
      "Die Wetter-/PV-Reihe liefert die elektrische PV-Leistung. Bei 15 Minuten ist Δt = 0.25 h.",
  },
  {
    name: "Wärmepumpe",
    equation: "P_th = COP · P_el",
    text:
      "Elektrische Wärmepumpenleistung wird über den COP in thermische Leistung umgerechnet. Die maximale Vorlauftemperatur begrenzt den erreichbaren Speicherzustand.",
  },
  {
    name: "Speicherenergie",
    equation: "E = C · (T − T_min)",
    text:
      "Für den vollständig durchmischten Wasserspeicher wird die nutzbare Energie relativ zur Mindesttemperatur beschrieben. C ist die thermische Speicherkapazität pro Kelvin.",
  },
  {
    name: "Wärmeverlust",
    equation: "Q_loss = G · (T − T_amb) · Δt",
    text:
      "Der Wärmeleitwert G koppelt den Speicher an die Umgebung. Je höher die Speichertemperatur, desto höher der Verlust.",
  },
  {
    name: "Energiebilanz",
    equation: "E_next = E_now + E_HP − E_DHW − Q_loss",
    text:
      "Nach Begrenzung auf die physische Speicherkapazität ergibt sich daraus der Zustand des nächsten Zeitschritts.",
  },
  {
    name: "Netz und Einspeisung",
    equation: "P_grid = max(0, P_load + P_HP − P_PV)",
    text:
      "Überschüssige PV-Leistung wird eingespeist; ein Defizit wird aus dem Netz bezogen.",
  },
] as const;

const chain = [
  {
    title: "PV und Haushaltslast",
    text:
      "Aus PV-Erzeugung und elektrischer Grundlast entsteht die Leistung, die der Wärmepumpe ohne zusätzlichen Netzbezug zur Verfügung steht.",
  },
  {
    title: "Warmwasserbedarf",
    text:
      "Das Verbrauchsprofil erzeugt eine zeitabhängige thermische Entnahme. Die Simulation schützt gleichzeitig eine Mindestreserve für die Versorgung.",
  },
  {
    title: "Forecast",
    text:
      "Der Regler schaut voraus: erwarteter Warmwasserbedarf, PV-Verfügbarkeit und Speicherverluste bestimmen, wie viel Energie künftig im Speicher benötigt wird.",
  },
  {
    title: "Waterline / Peak Shaving",
    text:
      "Der kausale Solver sucht den zulässigen elektrischen Cut. PV-Leistung oberhalb dieser Waterline kann gezielt zum Laden des Speichers verwendet werden.",
  },
  {
    title: "Thermischer Speicher",
    text:
      "Der Speicher koppelt alle Zeitschritte. Er kann PV-Energie zeitlich verschieben, verursacht dabei aber temperaturabhängige Verluste.",
  },
  {
    title: "Jahresziel",
    text:
      "Für jede Speichergrösse werden Netzbezug, Einspeisung, Eigenverbrauch, Speicherverluste, thermische Defizite und Energiekosten integriert.",
  },
] as const;

export default function PhysicsDocumentationPage() {

  return (
    <main className="mx-auto max-w-7xl px-4 py-10">
      <header className="mb-8">
        <p className="text-sm font-bold uppercase tracking-[0.18em] text-[#9a7410]">
          SunOpt · Dokumentation
        </p>
        <h1 className="mt-2 text-4xl font-black text-gray-950">
          Physikalische Zusammenhänge
        </h1>
        <p className="mt-4 max-w-4xl leading-7 text-[#675640]">
          SunOpt optimiert keinen abstrakten Zahlenvektor. Der Controller ist an
          die Energie- und Wärmebilanz von PV-Anlage, Wärmepumpe, Warmwasserbedarf
          und thermischem Speicher gebunden.
        </p>
      </header>

      <section className="grid gap-4 lg:grid-cols-2">
        {equations.map((item) => (
          <article
            key={item.name}
            className="rounded-2xl border border-[#d8c5a5] bg-white p-5 shadow-sm"
          >
            <h2 className="text-xl font-extrabold text-gray-950">
              {item.name}
            </h2>
            <div className="mt-3 rounded-xl bg-[#f3e6d2] px-4 py-3 font-mono text-lg font-bold text-[#594928]">
              {item.equation}
            </div>
            <p className="mt-3 leading-7 text-[#675640]">
              {item.text}
            </p>
          </article>
        ))}
      </section>

      <section className="mt-8 rounded-2xl border border-[#d8c5a5] bg-[#fffaf2] p-6 shadow-sm">
        <h2 className="text-2xl font-black text-gray-950">
          Energiefluss durch das Modell
        </h2>
        <div className="mt-5 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {chain.map((item, index) => (
            <article
              key={item.title}
              className="rounded-xl border border-[#e2d2b7] bg-white p-5"
            >
              <div className="text-sm font-black text-[#9a7410]">
                {String(index + 1).padStart(2, "0")}
              </div>
              <h3 className="mt-1 text-lg font-extrabold text-gray-950">
                {item.title}
              </h3>
              <p className="mt-2 leading-6 text-[#675640]">
                {item.text}
              </p>
            </article>
          ))}
        </div>
      </section>

      <section className="mt-8 grid gap-4 md:grid-cols-3">
        <article className="rounded-2xl border border-[#d8c5a5] bg-white p-5">
          <div className="text-sm font-bold uppercase tracking-wide text-[#9a7410]">
            Untergrenze
          </div>
          <div className="mt-2 text-3xl font-black text-gray-950">
            60 °C
          </div>
          <p className="mt-2 text-sm leading-6 text-[#675640]">
            Mindesttemperatur des Warmwasserspeichers im aktuellen Modell.
          </p>
        </article>

        <article className="rounded-2xl border border-[#d8c5a5] bg-white p-5">
          <div className="text-sm font-bold uppercase tracking-wide text-[#9a7410]">
            Obergrenze
          </div>
          <div className="mt-2 text-3xl font-black text-gray-950">
            75 °C
          </div>
          <p className="mt-2 text-sm leading-6 text-[#675640]">
            Physisch verwendete maximale Speichertemperatur beziehungsweise
            Wärmepumpen-Vorlauftemperatur.
          </p>
        </article>

        <article className="rounded-2xl border border-[#d8c5a5] bg-white p-5">
          <div className="text-sm font-bold uppercase tracking-wide text-[#9a7410]">
            Zeitschritt
          </div>
          <div className="mt-2 text-3xl font-black text-gray-950">
            15 min
          </div>
          <p className="mt-2 text-sm leading-6 text-[#675640]">
            35&apos;040 gekoppelte Zustandsübergänge pro Jahr.
          </p>
        </article>
      </section>

      <section className="mt-8 rounded-2xl border-2 border-[#7b5a00] bg-[#fff3c7] p-6">
        <h2 className="text-2xl font-black text-[#624800]">
          Warum der Speicher die Simulation sequenziell macht
        </h2>
        <p className="mt-3 max-w-5xl leading-7 text-[#66501e]">
          Die Speicherenergie nach Schritt n ist eine Eingangsgrösse von Schritt
          n+1. Damit kann ein einzelner Tank nicht als 35&apos;040 völlig
          unabhängige Zeitpunkte gerechnet werden. Parallelität entsteht zwischen
          unabhängigen Tanks und innerhalb des kausalen GPU-Solvers, nicht durch
          das Aufbrechen dieser Zustandskette.
        </p>
      </section>
    </main>
  );
}
