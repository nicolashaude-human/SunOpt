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
    label: "Einzelne vollständige Jahressimulation – Wrapper / Cranelift · Intel Core i7-9700",
    beforeLabel: "TypeScript / vollständige Jahressimulation",
    beforeSeconds: 170,
    afterLabel: "Wrapper / Oxc / Cranelift – sequenzielle Jahressimulation",
    afterSeconds: 27,
    note:
      "Intel Core i7-9700 (8 physische Kerne). Die einzelne vollständige Jahressimulation bleibt kausal/sequenziell. Der erste grosse Gewinn von ungefähr 170 s auf rund 27 s stammt aus Wrapper/Oxc/Rust/Cranelift. Danach wurden die Laufzeiten analysiert und weitere Software-Optimierungen umgesetzt: Compiler-Optimierungen, Pre-calculating wiederverwendbarer Werte und Eliminieren redundanter Berechnungen. Der spätere serielle 28-Tank-Sweep lag dadurch bei 148.455 s gesamt beziehungsweise durchschnittlich 5.269 s pro Tank (Minimum 4.277 s, Maximum 5.533 s). Diese Werte entstehen ohne Multi-Worker-Parallelisierung.",
  },
  {
    label: "Serieller 28-Tank-Sweep – bester CPU-Softwarestand · Intel Core i7-9700",
    beforeLabel: "Gesamter serieller Sweep",
    beforeSeconds: 148.4553,
    afterLabel: "Ø pro Tank / vollständige Jahressimulation",
    afterSeconds: 5.2688,
    note:
      "28 Tanks wurden nacheinander auf dem Intel Core i7-9700 gerechnet. Gesamtzeit 148.455 s, Durchschnitt 5.269 s pro Tank, Minimum 4.277 s, Maximum 5.533 s. Dieser Wert zeigt den besten erreichten CPU-Softwarestand pro einzelner Jahressimulation vor der Worker-Parallelisierung.",
  },
  {
    label: "28-Tank-Sweep – Parallelisierung unabhängiger Jahressimulationen",
    beforeLabel: "Intel Core i5-8400 · 6 CPU-Worker · optimierter Referenzlauf",
    beforeSeconds: 35.5852,
    afterLabel: "GeForce RTX 5060 · CUDA GPU Lockstep",
    afterSeconds: 7.1083,
    note:
      "Nach dem seriellen i7-9700-Sweep mit durchschnittlich 5.269 s pro Tank wurden die 28 voneinander unabhängigen Jahressimulationen parallelisiert. Auf dem Intel Core i7-9700 liefen 6 von 8 Kernen als Worker und der gesamte Sweep sank auf rund 47.03 s. Nach dem Wechsel auf das System mit Intel Core i5-8400 und GeForce RTX 5060 lag der optimierte 6-Worker-CPU-Referenzlauf bei 35.585 s; CUDA GPU Lockstep reduzierte denselben 28-Tank-Vergleich auf 7.108 s.",
  },
  {
    label: "Causal Replay – Runtime/Launch-Optimierung",
    beforeLabel: "normales CUDA Replay",
    beforeSeconds: 5.5997081,
    afterLabel: "CUDA Graph Replay · Synchronisation gebündelt",
    afterSeconds: 5.1649067,
    note:
      "CUDA Graph bündelt die 35'040 Launch-/Synchronisationspunkte experimentell und spart rund 435 ms beziehungsweise 7.8 %. Dieser Gewinn soll als letzter offener A-Schritt noch in den echten Produktionspfad übernommen werden; der Hauptaufwand bleibt danach die kausale Berechnung selbst.", 
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
    <main className="min-h-screen bg-[var(--sunopt-page)] px-6 py-10">

      <div className="mx-auto max-w-7xl">

        <header className="rounded-2xl border border-[var(--sunopt-border)] bg-[var(--sunopt-card)] p-8 shadow-sm">

          <div className="text-sm font-semibold uppercase tracking-wider text-[var(--sunopt-soft-text)]">
            SunOpt · Acceleration Strategy
          </div>

          <h1 className="mt-2 text-4xl font-bold tracking-tight text-[var(--sunopt-ink)]">
            Von A zu B
          </h1>

          <p className="mt-4 max-w-5xl text-lg leading-8 text-[var(--sunopt-muted)]">
            Die Beschleunigung von SunOpt wird ab jetzt bewusst in zwei
            grundverschiedene Ansätze getrennt. Ansatz A beschleunigt
            bestehenden Code, ohne den Algorithmus zu verändern. Ansatz B
            nutzt die spezielle Mathematik und Physik des SunOpt-Problems
            und darf deshalb einen anderen, problemspezifischen Solver
            verwenden.
          </p>

          <div className="mt-8 grid gap-5 lg:grid-cols-2">

            <div className="rounded-2xl border border-[var(--sunopt-border)] bg-[var(--sunopt-bg)] p-6">

              <div className="inline-flex rounded-full bg-[var(--sunopt-accent)] px-3 py-1 text-xs font-bold uppercase tracking-wide text-white">
                Ansatz A · Abschlussphase
              </div>

              <h2 className="mt-4 text-2xl font-bold text-[var(--sunopt-ink)]">
                Generische Code-Beschleunigung
              </h2>

              <p className="mt-3 leading-7 text-[var(--sunopt-muted)]">
                TypeScript wird analysiert, geeignete Scopes werden erkannt
                und derselbe Algorithmus über Oxc, Rust, Cranelift und CUDA
                bis auf nativen Maschinencode heruntergeführt. Optimiert
                werden Ausführung, Speicherzugriffe, Parallelität und
                Runtime — nicht die zugrunde liegende Physik.
              </p>

              <div className="mt-5 rounded-xl border border-[var(--sunopt-border)] bg-[var(--sunopt-card)] p-4 font-mono text-sm leading-7 text-gray-800">
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

              <h2 className="mt-4 text-2xl font-bold text-[var(--sunopt-ink)]">
                Problemspezifische High-Speed-Solver
              </h2>

              <p className="mt-3 leading-7 text-[var(--sunopt-muted)]">
                Hier wird Allgemeingültigkeit absichtlich aufgegeben.
                Statt den bestehenden Plasma-Wave-Code nur schneller
                auszuführen, wird die Struktur des thermischen Speichers,
                der Waterline und der kausalen Reservebedingungen direkt
                mathematisch ausgenutzt.
              </p>

              <div className="mt-5 rounded-xl border border-violet-200 bg-[var(--sunopt-card)] p-4 font-mono text-sm leading-7 text-gray-800">
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


        <section className="mt-10 rounded-2xl border border-[var(--sunopt-border)] bg-[var(--sunopt-card)] p-8 shadow-sm">

          <div className="flex flex-wrap items-end justify-between gap-4">

            <div>

              <div className="text-sm font-semibold uppercase tracking-wider text-[var(--sunopt-soft-text)]">
                Ansatz A · gemessener Verlauf
              </div>

              <h2 className="mt-2 text-3xl font-bold text-[var(--sunopt-ink)]">
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
                    className="rounded-2xl border border-[var(--sunopt-border-soft)] bg-[var(--sunopt-page)] p-6"
                  >

                    <div className="flex flex-wrap items-start justify-between gap-4">

                      <div>

                        <h3 className="text-lg font-bold text-[var(--sunopt-ink)]">
                          {benchmark.label}
                        </h3>

                        <p className="mt-2 max-w-4xl text-sm leading-6 text-[var(--sunopt-muted)]">
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

                          <span className="font-semibold text-[var(--sunopt-muted)]">
                            {benchmark.beforeLabel}
                          </span>

                          <span className="font-mono font-bold text-[var(--sunopt-ink)]">
                            {benchmark.beforeSeconds.toFixed(
                              benchmark.beforeSeconds >= 100
                                ? 0
                                : 3,
                            )} s
                          </span>

                        </div>

                        <div className="h-8 overflow-hidden rounded-lg bg-gray-200">

                          <div
                            className="flex h-full items-center rounded-lg bg-[var(--sunopt-page)]0 px-3 text-xs font-bold text-white"
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

                          <span className="font-semibold text-[var(--sunopt-muted)]">
                            {benchmark.afterLabel}
                          </span>

                          <span className="font-mono font-bold text-[var(--sunopt-ink)]">
                            {benchmark.afterSeconds.toFixed(
                              benchmark.afterSeconds >= 100
                                ? 0
                                : 3,
                            )} s
                          </span>

                        </div>

                        <div className="h-8 overflow-hidden rounded-lg bg-[#efe5cf]">

                          <div
                            className="flex h-full items-center rounded-lg bg-[var(--sunopt-accent)] px-3 text-xs font-bold text-white"
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

          <div className="mt-8 grid gap-5 lg:grid-cols-3">

            <div className="rounded-2xl border border-blue-200 bg-blue-50 p-6">

              <div className="text-sm font-semibold uppercase tracking-wider text-blue-700">
                Phase 1 · CPU-System
              </div>

              <div className="mt-2 text-2xl font-bold text-[var(--sunopt-ink)]">
                Intel Core i7-9700 · 8 Kerne
              </div>

              <p className="mt-3 text-sm leading-7 text-[var(--sunopt-muted)]">
                Die einzelne vollständige Jahressimulation ist
                kausal/sequenziell. Sie wurde nicht über Worker
                parallelisiert. Der erste grosse Sprung von ungefähr
                170 s → 27 s entstand durch Wrapper/Oxc/Cranelift.
                Danach wurde die Laufzeit systematisch analysiert und
                der Code gezielt weiter optimiert: Optimierungen beim
                Compilieren, Pre-calculating wiederverwendbarer Werte
                und Eliminieren redundanter Berechnungen. Dadurch sank
                die Rechenzeit im seriellen 28-Tank-Sweep auf
                durchschnittlich 5.269 s pro Tank beziehungsweise
                vollständiger Jahressimulation (Minimum 4.277 s,
                Maximum 5.533 s). Erst danach wurde der separate
                28-Tank-Sweep über mehrere Worker parallelisiert.
              </p>

            </div>


            <div className="rounded-2xl border border-amber-200 bg-amber-50 p-6">

              <div className="text-sm font-semibold uppercase tracking-wider text-amber-700">
                CPU-Softwareoptimierung · 27 s → 5.2 s
              </div>

              <div className="mt-2 text-2xl font-bold text-[var(--sunopt-ink)]">
                Laufzeit systematisch herunteroptimiert
              </div>

              <ul className="mt-4 space-y-2 text-sm leading-7 text-[var(--sunopt-muted)]">
                <li>• Analyse der Laufzeiten und Identifikation der Hot Paths</li>
                <li>• Optimierungen beim Compilieren</li>
                <li>• Pre-calculating von wiederverwendbaren Werten</li>
                <li>• Eliminieren redundanter Berechnungen</li>
              </ul>

              <p className="mt-4 text-sm leading-7 text-[var(--sunopt-muted)]">
                Ergebnis im seriellen i7-9700-Sweep:
                Ø 5.269 s pro Tank/Jahressimulation,
                Minimum 4.277 s, Maximum 5.533 s.
              </p>

            </div>


            <div className="rounded-2xl border border-cyan-200 bg-cyan-50 p-6">

              <div className="text-sm font-semibold uppercase tracking-wider text-cyan-700">
                Phase 2 · neues System
              </div>

              <div className="mt-2 text-2xl font-bold text-[var(--sunopt-ink)]">
                Intel Core i5-8400 · 6 Kerne
              </div>

              <p className="mt-3 text-sm leading-7 text-[var(--sunopt-muted)]">
                Mit dem später verwendeten RTX-5060-Rechner wechselte
                auch die CPU auf den Intel Core i5-8400. Der optimierte
                28-Tank-Referenzlauf mit 6 CPU-Workern lag hier bei
                35.585 s.
              </p>

            </div>


            <div className="rounded-2xl border border-green-200 bg-green-50 p-6">

              <div className="text-sm font-semibold uppercase tracking-wider text-green-700">
                Phase 3 · GPU
              </div>

              <div className="mt-2 text-2xl font-bold text-[var(--sunopt-ink)]">
                NVIDIA GeForce RTX 5060 · 30 SM
              </div>

              <p className="mt-3 text-sm leading-7 text-[var(--sunopt-muted)]">
                CUDA GPU Lockstep reduzierte den vollständigen
                28-Tank-Vergleich auf 7.108 s. Das reine Causal Replay
                liegt bei etwa 5.600 s und mit CUDA Graph bei 5.165 s.
              </p>

            </div>

          </div>


          <div className="mt-8 rounded-2xl border border-[var(--sunopt-border-soft)] bg-[var(--sunopt-page)] p-6">

            <div className="text-sm font-semibold uppercase tracking-wider text-[var(--sunopt-muted)]">
              Hardware-Zeitlinie
            </div>

            <div className="mt-4 grid gap-4 md:grid-cols-[1fr_auto_1fr_auto_1fr] md:items-center">

              <div className="rounded-xl border border-blue-200 bg-[var(--sunopt-card)] p-4">
                <div className="font-bold text-[var(--sunopt-ink)]">i7-9700</div>
                <div className="mt-1 text-sm text-[var(--sunopt-muted)]">
                  einzelne Jahressimulation · sequenziell
                  <br />
                  Wrapper/Cranelift · 170 → 27 s
                  <br />
                  Laufzeitanalyse + Compile/Pre-calc/Redundanz · Ø 5.269 s
                  <br />
                  28-Tank-Sweep · 6 Worker · 47.03 s
                </div>
              </div>

              <div className="text-center text-2xl text-gray-400">→</div>

              <div className="rounded-xl border border-cyan-200 bg-[var(--sunopt-card)] p-4">
                <div className="font-bold text-[var(--sunopt-ink)]">i5-8400</div>
                <div className="mt-1 text-sm text-[var(--sunopt-muted)]">
                  RTX-5060-System
                  <br />
                  28 Tanks · 6 Worker · 35.585 s
                </div>
              </div>

              <div className="text-center text-2xl text-gray-400">→</div>

              <div className="rounded-xl border border-green-200 bg-[var(--sunopt-card)] p-4">
                <div className="font-bold text-[var(--sunopt-ink)]">RTX 5060</div>
                <div className="mt-1 text-sm text-[var(--sunopt-muted)]">
                  28 Tanks · CUDA Lockstep
                  <br />
                  7.108 s
                </div>
              </div>

            </div>

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
                Abschlussphase Ansatz A
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
                Compiler-/Runtime-Technologie. Ein letzter klar
                quantifizierter Punkt bleibt offen: die im CUDA-Graph-
                Replay nachgewiesenen rund 435 ms Launch-/Synchronisations-
                Gewinn auch im Produktionspfad zu realisieren. Danach gilt
                A als abgeschlossen; der nächste Grössenordnungssprung
                muss aus Ansatz B kommen.
              </p>

            </div>

          </div>

        </section>



          <div className="mt-6 rounded-2xl border border-orange-300 bg-orange-50 p-6">

            <div className="text-sm font-semibold uppercase tracking-wider text-orange-800">
              Letzter offener A-Schritt
            </div>

            <h3 className="mt-2 text-xl font-bold text-orange-950">
              Rund 435 ms Produktions-Overhead noch entfernen
            </h3>

            <p className="mt-3 leading-7 text-orange-950">
              Das Blackwell CUDA-Graph-Experiment zeigte 5.600 s →
              5.165 s, also ungefähr 434.8 ms beziehungsweise 7.8 %.
              Im Experiment werden die vielen Host/GPU-Launch- und
              Synchronisationspunkte in einem Graph gebündelt. Im
              Produktionspfad bestehen die zustandsabhängigen
              Synchronisationspunkte noch. Dieser messbare Restgewinn
              soll noch realisiert werden, ohne die physikalische
              Semantik zu verändern.
            </p>

          </div>

        <section className="mt-10 rounded-2xl border border-violet-200 bg-[var(--sunopt-card)] p-8 shadow-sm">

          <div className="text-sm font-semibold uppercase tracking-wider text-violet-700">
            Ansatz B · problemspezifische Forschung
          </div>

          <h2 className="mt-2 text-3xl font-bold text-[var(--sunopt-ink)]">
            Physik komprimieren statt denselben Code immer schneller auszuführen
          </h2>

          <p className="mt-4 max-w-5xl text-lg leading-8 text-[var(--sunopt-muted)]">
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

              <h3 className="mt-2 text-xl font-bold text-[var(--sunopt-ink)]">
                Max-Affine Blockkomposition
              </h3>

              <p className="mt-3 text-sm leading-7 text-[var(--sunopt-muted)]">
                Die eindimensionale Speicherphysik wird als
                zusammensetzbare, stückweise affine Transformation
                formuliert. Ziel ist, einen 192-Schritt-Horizont in
                wenigen Baumebenen zusammenzusetzen, ohne die Physik
                zu approximieren.
              </p>

              <div className="mt-5 rounded-lg bg-[var(--sunopt-card)] p-4 font-mono text-sm text-gray-800">
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

              <h3 className="mt-2 text-xl font-bold text-[var(--sunopt-ink)]">
                Direkter Cut-Level-Solver
              </h3>

              <p className="mt-3 text-sm leading-7 text-[var(--sunopt-muted)]">
                Statt bis zu 20 Causal-Binary-Search-Runden jeweils über
                den 48-h-Horizont zu simulieren, soll die stückweise
                lineare Abhängigkeit vom Cut-Level direkt ausgewertet
                werden.
              </p>

              <div className="mt-5 rounded-lg bg-[var(--sunopt-card)] p-4 font-mono text-sm text-gray-800">
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

              <h3 className="mt-2 text-xl font-bold text-[var(--sunopt-ink)]">
                Fourier-Kompression
              </h3>

              <p className="mt-3 text-sm leading-7 text-[var(--sunopt-muted)]">
                PV-Export und Warmwasserbedarf werden mit deutlich weniger
                Frequenzkoeffizienten dargestellt. Untersucht wird, wie
                stark sich die Signale reduzieren lassen, bevor Cut-Level
                oder Survival-Entscheide von Plasma Wave abweichen.
              </p>

              <div className="mt-5 rounded-lg bg-[var(--sunopt-card)] p-4 font-mono text-sm text-gray-800">
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
            className="rounded-lg border border-[var(--sunopt-border)] bg-[var(--sunopt-card)] px-4 py-2 text-sm font-semibold text-[var(--sunopt-muted)] hover:bg-[var(--sunopt-panel-hover)]"
          >
            ← Projektstatus
          </Link>

          <Link
            href="/tank-comparison"
            className="rounded-lg bg-[var(--sunopt-accent)] px-4 py-2 text-sm font-semibold text-white hover:brightness-95"
          >
            Tankvergleich
          </Link>

        </div>

      </div>

    </main>
  );
}
