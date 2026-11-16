// src/app/acceleration-strategy/page.tsx

import Link from "next/link";


type BenchmarkRow = {
  label: string;
  beforeLabel: string;
  beforeSeconds: number;
  afterLabel: string;
  afterSeconds: number;
  note: string;
};


const BENCHMARKS: BenchmarkRow[] = [
  {
    label: "Jahressimulation – generische CPU-Beschleunigung",
    beforeLabel: "TypeScript Baseline",
    beforeSeconds: 198,
    afterLabel: "Oxc / Cranelift",
    afterSeconds: 27,
    note:
      "Gleicher Algorithmus und gleiche Physik; TypeScript-Scope wird erkannt und bis zu nativem Maschinencode herunterkompiliert.",
  },
  {
    label: "28-Tank-Vergleich – GPU-Parallelisierung",
    beforeLabel: "früher Batch-Lauf",
    beforeSeconds: 35.5852,
    afterLabel: "GPU Lockstep",
    afterSeconds: 7.1083,
    note:
      "28 unabhängige Tanks werden parallel verarbeitet. Diese Messreihe ist nicht direkt mit der einzelnen Jahressimulation darüber vergleichbar.",
  },
  {
    label: "Causal Replay – Runtime/Launch-Optimierung",
    beforeLabel: "normales CUDA Replay",
    beforeSeconds: 5.5997081,
    afterLabel: "CUDA Graph Replay",
    afterSeconds: 5.1649067,
    note:
      "Nur noch rund 7.8 % Gewinn. Der Hauptaufwand liegt damit in der kausalen Berechnung selbst und nicht mehr in Host/GPU-Synchronisation oder Kernel-Launches.",
  },
];


function percent(
  value: number,
  maximum: number,
): string {
  return `${Math.max(
    4,
    value / maximum * 100,
  ).toFixed(2)}%`;
}


function speedup(
  before: number,
  after: number,
): string {
  return `${(
    before /
    after
  ).toFixed(2)}×`;
}


export default function AccelerationStrategyPage() {

  const graphGainMs =
    (
      5.5997081 -
      5.1649067
    ) *
    1000;

  const graphGainPercent =
    graphGainMs /
    (
      5.5997081 *
      1000
    ) *
    100;


  return (
    <main className="min-h-screen bg-[#fffaf2] px-6 py-10">

      <div className="mx-auto max-w-7xl">

        <header className="rounded-2xl border border-[#d9c7aa] bg-white p-8 shadow-sm">

          <div className="text-sm font-semibold uppercase tracking-wider text-[#8a734f]">
            SunOpt · Acceleration Strategy
          </div>

          <h1 className="mt-2 text-4xl font-bold tracking-tight text-gray-950">
            Von A zu B
          </h1>

          <p className="mt-4 max-w-5xl text-lg leading-8 text-gray-700">
            Die Beschleunigung von SunOpt wird ab jetzt bewusst in zwei
            grundverschiedene Ansätze getrennt. Ansatz A beschleunigt
            bestehenden Code, ohne den Algorithmus zu verändern. Ansatz B
            nutzt die spezielle Mathematik und Physik des SunOpt-Problems
            und darf deshalb einen anderen, problemspezifischen Solver
            verwenden.
          </p>

          <div className="mt-8 grid gap-5 lg:grid-cols-2">

            <div className="rounded-2xl border border-[#d9c7aa] bg-[#fff8e8] p-6">

              <div className="inline-flex rounded-full bg-[#ddb114] px-3 py-1 text-xs font-bold uppercase tracking-wide text-white">
                Ansatz A · abgeschlossen
              </div>

              <h2 className="mt-4 text-2xl font-bold text-gray-950">
                Generische Code-Beschleunigung
              </h2>

              <p className="mt-3 leading-7 text-gray-700">
                TypeScript wird analysiert, geeignete Scopes werden erkannt
                und derselbe Algorithmus über Oxc, Rust, Cranelift und CUDA
                bis auf nativen Maschinencode heruntergeführt. Optimiert
                werden Ausführung, Speicherzugriffe, Parallelität und
                Runtime — nicht die zugrunde liegende Physik.
              </p>

              <div className="mt-5 rounded-xl border border-[#d9c7aa] bg-white p-4 font-mono text-sm leading-7 text-gray-800">
                TypeScript
                <br />
                ↓ Oxc Parser
                <br />
                ↓ Scope-Erkennung
                <br />
                ↓ gleiche Semantik / gleicher Algorithmus
                <br />
                ↓ Cranelift / Rust / CUDA
                <br />
                ↓ Maschinencode
              </div>

            </div>


            <div className="rounded-2xl border border-violet-200 bg-violet-50 p-6">

              <div className="inline-flex rounded-full bg-violet-700 px-3 py-1 text-xs font-bold uppercase tracking-wide text-white">
                Ansatz B · neue Forschungsphase
              </div>

              <h2 className="mt-4 text-2xl font-bold text-gray-950">
                Problemspezifische High-Speed-Solver
              </h2>

              <p className="mt-3 leading-7 text-gray-700">
                Hier wird Allgemeingültigkeit absichtlich aufgegeben.
                Statt den bestehenden Plasma-Wave-Code nur schneller
                auszuführen, wird die Struktur des thermischen Speichers,
                der Waterline und der kausalen Reservebedingungen direkt
                mathematisch ausgenutzt.
              </p>

              <div className="mt-5 rounded-xl border border-violet-200 bg-white p-4 font-mono text-sm leading-7 text-gray-800">
                SunOpt-Physik
                <br />
                ↓ mathematische Struktur erkennen
                <br />
                ↓ Zustände / Signale komprimieren
                <br />
                ↓ problemspezifischen Solver formulieren
                <br />
                ↓ gegen Plasma Wave verifizieren
                <br />
                ↓ High-Speed CPU / GPU Implementierung
              </div>

            </div>

          </div>

        </header>


        <section className="mt-10 rounded-2xl border border-[#d9c7aa] bg-white p-8 shadow-sm">

          <div className="flex flex-wrap items-end justify-between gap-4">

            <div>

              <div className="text-sm font-semibold uppercase tracking-wider text-[#8a734f]">
                Ansatz A · gemessener Verlauf
              </div>

              <h2 className="mt-2 text-3xl font-bold text-gray-950">
                Wo die generische Beschleunigung gebrochen hat
              </h2>

            </div>

            <div className="rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-950">
              Wichtig: Die drei Balkengruppen sind drei getrennte
              Benchmark-Reihen und werden nicht untereinander als
              identischer Workload verglichen.
            </div>

          </div>


          <div className="mt-8 space-y-8">

            {BENCHMARKS.map(
              (
                benchmark,
              ) => {

                const maximum =
                  Math.max(
                    benchmark.beforeSeconds,
                    benchmark.afterSeconds,
                  );

                return (
                  <article
                    key={benchmark.label}
                    className="rounded-2xl border border-gray-200 bg-gray-50 p-6"
                  >

                    <div className="flex flex-wrap items-start justify-between gap-4">

                      <div>

                        <h3 className="text-lg font-bold text-gray-950">
                          {benchmark.label}
                        </h3>

                        <p className="mt-2 max-w-4xl text-sm leading-6 text-gray-600">
                          {benchmark.note}
                        </p>

                      </div>

                      <div className="rounded-full border border-green-300 bg-green-50 px-4 py-2 text-sm font-bold text-green-800">
                        {speedup(
                          benchmark.beforeSeconds,
                          benchmark.afterSeconds,
                        )}
                      </div>

                    </div>


                    <div className="mt-6 space-y-5">

                      <div>

                        <div className="mb-2 flex flex-wrap justify-between gap-3 text-sm">

                          <span className="font-semibold text-gray-700">
                            {benchmark.beforeLabel}
                          </span>

                          <span className="font-mono font-bold text-gray-950">
                            {benchmark.beforeSeconds.toFixed(
                              benchmark.beforeSeconds >= 100
                                ? 0
                                : 3,
                            )} s
                          </span>

                        </div>

                        <div className="h-8 overflow-hidden rounded-lg bg-gray-200">

                          <div
                            className="flex h-full items-center rounded-lg bg-gray-500 px-3 text-xs font-bold text-white"
                            style={{
                              width:
                                percent(
                                  benchmark.beforeSeconds,
                                  maximum,
                                ),
                            }}
                          >
                            vorher
                          </div>

                        </div>

                      </div>


                      <div>

                        <div className="mb-2 flex flex-wrap justify-between gap-3 text-sm">

                          <span className="font-semibold text-gray-700">
                            {benchmark.afterLabel}
                          </span>

                          <span className="font-mono font-bold text-gray-950">
                            {benchmark.afterSeconds.toFixed(
                              benchmark.afterSeconds >= 100
                                ? 0
                                : 3,
                            )} s
                          </span>

                        </div>

                        <div className="h-8 overflow-hidden rounded-lg bg-[#efe5cf]">

                          <div
                            className="flex h-full items-center rounded-lg bg-[#ddb114] px-3 text-xs font-bold text-white"
                            style={{
                              width:
                                percent(
                                  benchmark.afterSeconds,
                                  maximum,
                                ),
                            }}
                          >
                            nachher
                          </div>

                        </div>

                      </div>

                    </div>

                  </article>
                );
              },
            )}

          </div>


          <div className="mt-8 grid gap-5 lg:grid-cols-[1fr_1.4fr]">

            <div className="rounded-2xl border border-amber-300 bg-amber-50 p-6">

              <div className="text-sm font-semibold uppercase tracking-wider text-amber-800">
                Break point von A
              </div>

              <div className="mt-2 text-4xl font-bold text-amber-950">
                5.165 s
              </div>

              <div className="mt-2 text-sm text-amber-900">
                CUDA Graph Replay
              </div>

              <div className="mt-5 border-t border-amber-200 pt-5">

                <div className="text-sm text-amber-900">
                  Gewinn gegenüber normalem Replay
                </div>

                <div className="mt-1 text-2xl font-bold text-amber-950">
                  {graphGainMs.toFixed(1)} ms · {graphGainPercent.toFixed(1)} %
                </div>

              </div>

            </div>


            <div className="rounded-2xl border border-gray-800 bg-gray-950 p-6 text-white">

              <h3 className="text-xl font-bold">
                Abschlussfolgerung Ansatz A
              </h3>

              <p className="mt-3 leading-7 text-gray-300">
                Das Entfernen beziehungsweise Zusammenfassen der
                Synchronisations- und Launch-Punkte brachte nur noch
                rund 0.435 s. Die GPU-Rechenzeit selbst blieb bei ungefähr
                fünf Sekunden. Damit ist die generische Pipeline nicht
                gescheitert — im Gegenteil: Sie hat den Code bis nahe an
                die Laufzeit des zugrunde liegenden kausalen Problems
                gebracht.
              </p>

              <p className="mt-4 leading-7 text-gray-300">
                Ansatz A bleibt Referenz, Fallback und allgemeine
                Compiler-/Runtime-Technologie. Weitere Verbesserungen
                werden erwartet, aber eher inkrementell. Für einen neuen
                Grössenordnungssprung beginnt Ansatz B.
              </p>

            </div>

          </div>

        </section>


        <section className="mt-10 rounded-2xl border border-violet-200 bg-white p-8 shadow-sm">

          <div className="text-sm font-semibold uppercase tracking-wider text-violet-700">
            Ansatz B · problemspezifische Forschung
          </div>

          <h2 className="mt-2 text-3xl font-bold text-gray-950">
            Physik komprimieren statt denselben Code immer schneller auszuführen
          </h2>

          <p className="mt-4 max-w-5xl text-lg leading-8 text-gray-700">
            Jeder B-Solver muss gegen die bestehende Plasma-Wave-Referenz
            antreten. Exakte Verfahren müssen numerische Parität erreichen.
            Approximative Verfahren dürfen nur übernommen werden, wenn ihr
            Fehler für Waterline, Survival und Jahresresultate kontrolliert
            und akzeptabel ist.
          </p>


          <div className="mt-8 grid gap-5 lg:grid-cols-3">

            <article className="rounded-2xl border border-violet-200 bg-violet-50 p-6">

              <div className="text-xs font-bold uppercase tracking-wide text-violet-700">
                B1 · exakt
              </div>

              <h3 className="mt-2 text-xl font-bold text-gray-950">
                Max-Affine Blockkomposition
              </h3>

              <p className="mt-3 text-sm leading-7 text-gray-700">
                Die eindimensionale Speicherphysik wird als
                zusammensetzbare, stückweise affine Transformation
                formuliert. Ziel ist, einen 192-Schritt-Horizont in
                wenigen Baumebenen zusammenzusetzen, ohne die Physik
                zu approximieren.
              </p>

              <div className="mt-5 rounded-lg bg-white p-4 font-mono text-sm text-gray-800">
                192 Schritte
                <br />
                → 96
                <br />
                → 48
                <br />
                → 24
                <br />
                → …
                <br />
                → ~8 Kompositionsebenen
              </div>

            </article>


            <article className="rounded-2xl border border-violet-200 bg-violet-50 p-6">

              <div className="text-xs font-bold uppercase tracking-wide text-violet-700">
                B2 · exakt angestrebt
              </div>

              <h3 className="mt-2 text-xl font-bold text-gray-950">
                Direkter Cut-Level-Solver
              </h3>

              <p className="mt-3 text-sm leading-7 text-gray-700">
                Statt bis zu 20 Causal-Binary-Search-Runden jeweils über
                den 48-h-Horizont zu simulieren, soll die stückweise
                lineare Abhängigkeit vom Cut-Level direkt ausgewertet
                werden.
              </p>

              <div className="mt-5 rounded-lg bg-white p-4 font-mono text-sm text-gray-800">
                heute:
                <br />
                Cut → 192 Schritte → survives?
                <br />
                × bis zu 20
                <br /><br />
                Ziel:
                <br />
                Hüllkurve → Cut direkt
              </div>

            </article>


            <article className="rounded-2xl border border-violet-200 bg-violet-50 p-6">

              <div className="text-xs font-bold uppercase tracking-wide text-violet-700">
                B3 · approximativ / explorativ
              </div>

              <h3 className="mt-2 text-xl font-bold text-gray-950">
                Fourier-Kompression
              </h3>

              <p className="mt-3 text-sm leading-7 text-gray-700">
                PV-Export und Warmwasserbedarf werden mit deutlich weniger
                Frequenzkoeffizienten dargestellt. Untersucht wird, wie
                stark sich die Signale reduzieren lassen, bevor Cut-Level
                oder Survival-Entscheide von Plasma Wave abweichen.
              </p>

              <div className="mt-5 rounded-lg bg-white p-4 font-mono text-sm text-gray-800">
                192 Zeitwerte
                <br />
                ↓ Fourier
                <br />
                8 / 16 / 24 / 32 / …
                <br />
                Harmonische
                <br />
                ↓
                <br />
                Fehler gegen Referenz
              </div>

            </article>

          </div>


          <div className="mt-8 rounded-2xl border border-violet-300 bg-violet-950 p-6 text-white">

            <h3 className="text-xl font-bold">
              Regel für B
            </h3>

            <p className="mt-3 max-w-5xl leading-7 text-violet-100">
              Problemspezifische Geschwindigkeit zählt erst, wenn der
              Solver gegen Plasma Wave besteht. Deshalb werden Laufzeit,
              maximaler Cut-Level-Fehler, Survival-Mismatches und
              Jahresparität gemeinsam bewertet. Eine schnelle,
              physikalisch falsche Näherung ist kein Erfolg.
            </p>

          </div>

        </section>


        <div className="mt-8 flex flex-wrap gap-3">

          <Link
            href="/project-status"
            className="rounded-lg border border-[#d9c7aa] bg-white px-4 py-2 text-sm font-semibold text-[#66543d] hover:bg-[#f3e6d2]"
          >
            ← Projektstatus
          </Link>

          <Link
            href="/tank-comparison"
            className="rounded-lg bg-[#ddb114] px-4 py-2 text-sm font-semibold text-white hover:brightness-95"
          >
            Tankvergleich
          </Link>

        </div>

      </div>

    </main>
  );
}
