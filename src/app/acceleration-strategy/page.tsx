// src/app/acceleration-strategy/page.tsx



type HistoryStep = {
  stage: string;
  hardware: string;
  tanks: number;
  wallSeconds: number;
  detail: string;
  kind: "single" | "sweep";
};

// Einheitliche Bezugsgrösse für sämtliche Stufen:
// 1 Tank / 200 s = 0.005 Tanks/s; die historische Referenz ist gerundet.
const BASELINE_TANKS_PER_SECOND = 1 / 200;

const HISTORY: HistoryStep[] = [
  {
    stage: "Ausgangslage · TypeScript",
    hardware: "Intel Core i7-9700",
    tanks: 1,
    wallSeconds: 200,
    detail: "Erste vollständige Jahressimulation eines einzelnen Speichers. Historischer Ausgangswert: ungefähr 200 Sekunden.",
    kind: "single",
  },
  {
    stage: "Next.js / TypeScript optimiert",
    hardware: "Intel Core i7-9700",
    tanks: 1,
    wallSeconds: 170,
    detail: "Erste Optimierungen des Next.js-/TypeScript-Ausführungspfades vor dem Einsatz des nativen JIT.",
    kind: "single",
  },
  {
    stage: "Wrapper → Oxc → Rust / Cranelift",
    hardware: "Intel Core i7-9700",
    tanks: 1,
    wallSeconds: 27,
    detail: "Ausgewählte Berechnungsfunktionen werden zur Laufzeit mit Oxc verarbeitet und über Rust/Cranelift in nativen CPU-Maschinencode übersetzt.",
    kind: "single",
  },
  {
    stage: "Weitere CPU-Softwareoptimierungen",
    hardware: "Intel Core i7-9700",
    tanks: 1,
    wallSeconds: 5.2,
    detail: "Ungefähr 5,2 s pro Einzelsimulation durch Profiling der Hot Paths, Compiler-Einstellungen, Vorberechnung und Entfernen redundanter Arbeit. Historischer Näherungswert; nicht mit einem vollständigen Sweep gleichsetzen.",
    kind: "single",
  },
  {
    stage: "Serieller 28-Tank-Sweep",
    hardware: "Intel Core i7-9700 · 1 Worker",
    tanks: 28,
    wallSeconds: 148.4553,
    detail: "148,4553 s Gesamt-Wall-Time für 28 Tanks, darin 147,5373 s Tank-Sweep. Der Sweep selbst erreichte Ø 5,2688 s/Tank (Minimum 4,2773 s; Maximum 5,5325 s). Für den Durchsatz zählt ausschliesslich die Gesamt-Wall-Time.",
    kind: "sweep",
  },
  {
    stage: "Paralleler CPU-Sweep",
    hardware: "Intel Core i7-9700 · 6 Worker",
    tanks: 28,
    wallSeconds: 47.03,
    detail: "Sechs der acht physischen Kerne bearbeiten unabhängige Tank-Jahressimulationen parallel; die Jahreszeitreihe innerhalb eines Tanks bleibt sequenziell.",
    kind: "sweep",
  },
  {
    stage: "CPU-Referenz auf GPU-Rechner",
    hardware: "Intel Core i5-8400 · 6 Worker",
    tanks: 28,
    wallSeconds: 35.5852,
    detail: "Separates System mit i5-8400 und RTX 5060. Optimierter CPU-Referenz-Sweep mit sechs Workern; Hardware und Messumgebung wurden gewechselt.",
    kind: "sweep",
  },
  {
    stage: "CUDA FP64 · früherer Lockstep",
    hardware: "GeForce RTX 5060 · Ansatz A",
    tanks: 28,
    wallSeconds: 7.1083,
    detail: "Vollständiger CUDA-Jahressweep aus einer früheren Messreihe. Nicht als 3480-MHz-Fixed-Clock-Messung ausgeben, sofern der GPU-Takt dort nicht protokolliert wurde.",
    kind: "sweep",
  },
  {
    stage: "B14Q · FP64-Referenz",
    hardware: "GeForce RTX 5060 · 3480 MHz",
    tanks: 28,
    wallSeconds: 6.264,
    detail: "Kontrollierter 3480-MHz-Produktionsvergleich mit Ansatz A. Die Lüfterdrehzahl dieser historischen Messung ist hier nicht belegt.",
    kind: "sweep",
  },
  {
    stage: "B14Q · spezialisierter FP32-Solver",
    hardware: "GeForce RTX 5060 · 3480 MHz",
    tanks: 28,
    wallSeconds: 3.55,
    detail: "Derselbe kontrollierte 28-Tank-Produktionsvergleich mit Ansatz B. Die Lüfterdrehzahl dieser historischen Messung ist hier nicht belegt.",
    kind: "sweep",
  },
];

function tankRate(step: HistoryStep): number {
  return step.tanks / step.wallSeconds;
}

function vsOriginal(step: HistoryStep): number {
  return tankRate(step) / BASELINE_TANKS_PER_SECOND;
}

function secondsLabel(seconds: number): string {
  return seconds >= 100
    ? seconds.toLocaleString("de-CH", { maximumFractionDigits: 3 })
    : seconds.toLocaleString("de-CH", { maximumFractionDigits: 4 });
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
            Die Entwicklung wurde in zwei Ansätze aufgeteilt. Der bestehende
            Ansatz A berechnet die Referenzergebnisse mit hoher Präzision.
            Ansatz B verwendet dagegen eine speziell für SunOpt entwickelte
            mathematische Lösung. Später wurde geprüft, ob sich Methoden aus B
            unabhängig von der konkreten Funktion auf A übertragen lassen.
            Der generische CUDA-Versuch A9–A11 erreichte dieses Ziel nicht.
          </p>

          <div className="mt-8 grid gap-5 lg:grid-cols-2">

            <div className="rounded-2xl border border-[var(--sunopt-border)] bg-[var(--sunopt-bg)] p-6">

              <div className="inline-flex rounded-full bg-[var(--sunopt-accent)] px-3 py-1 text-xs font-bold uppercase tracking-wide text-white">
                Ansatz A · produktive Referenz
              </div>

              <h2 className="mt-4 text-2xl font-bold text-[var(--sunopt-ink)]">
                FP64-Referenz und JIT-Technik
              </h2>

              <p className="mt-3 leading-7 text-[var(--sunopt-muted)]">
                Der CPU-JIT kann dafür ausgewählte TypeScript-Funktionen
                über Oxc und Cranelift in nativen Maschinencode übersetzen.
                Daneben existiert der bewährte FP64-CUDA-Pfad für die
                Jahresberechnung. Der später untersuchte generische
                TypeScript-zu-CUDA-Compiler A9–A11 lieferte zwar korrekte
                Ergebnisse, beschleunigte Original A aber nicht.
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
                ↓ Rust / Cranelift · CPU-JIT
                <br />
                ↓ nativer Maschinencode
                <br />
                CUDA-Jahreslauf: separater FP64-GPU-Pfad
              </div>

            </div>


            <div className="rounded-2xl border border-violet-200 bg-violet-50 p-6">

              <div className="inline-flex rounded-full bg-violet-700 px-3 py-1 text-xs font-bold uppercase tracking-wide text-white">
                Ansatz B · spezialisierter GPU-Solver
              </div>

              <h2 className="mt-4 text-2xl font-bold text-[var(--sunopt-ink)]">
                Problemspezifische High-Speed-Solver
              </h2>

              <p className="mt-3 leading-7 text-[var(--sunopt-muted)]">
                Hier wird Allgemeingültigkeit absichtlich aufgegeben.
                Für Ansatz B wurden die physikalischen Eigenschaften
                des thermischen Speichers, die Waterline und die
                Bedingungen für die kausale Energiereserve im Voraus
                untersucht. Daraus entstand eine mathematisch kompakte
                Berechnung, die gezielt für eine effiziente Ausführung
                auf der Blackwell-GPU vorbereitet wurde. Diese Lösung
                ist nicht auf beliebige Funktionen übertragbar.
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
                ↓ gegen FP64-Referenz verifizieren
                <br />
                ↓ High-Speed CPU / GPU Implementierung
              </div>

            </div>

          </div>

        </header>


        <section className="mt-10 rounded-2xl border border-[var(--sunopt-border)] bg-[var(--sunopt-card)] p-8 shadow-sm">

          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <div className="text-sm font-semibold uppercase tracking-wider text-[var(--sunopt-soft-text)]">
                Ansatz A · gemessener Verlauf
              </div>
              <h2 className="mt-2 text-3xl font-bold text-[var(--sunopt-ink)]">
                Vom ersten Tank mit 200 s zum 28-Tank-Sweep
              </h2>
              <p className="mt-3 max-w-4xl leading-7 text-[var(--sunopt-muted)]">
                Zuerst wurde <strong>eine einzelne Jahressimulation</strong> schneller gemacht:
                von ungefähr 200 s über 170 s und 27 s bis auf rund 5,2 s.
                Erst danach wurde auf 28 Tanks und später auf mehrere CPU-Worker und die GPU erweitert.
                Alle Faktoren in der Tabelle verwenden dieselbe Einheit und denselben Ausgangspunkt.
              </p>
            </div>
            <div className="rounded-xl border border-green-300 bg-green-50 px-4 py-3 text-sm text-green-950">
              Referenz: 1 Tank / 200 s = <strong>0,005 Tanks/s · 1,00×</strong>
            </div>
          </div>

          <div className="mt-6 overflow-x-auto rounded-xl border border-[var(--sunopt-border)]">
            <table className="w-full min-w-[880px] border-collapse text-left text-sm">
              <thead className="bg-[var(--sunopt-panel)] text-[var(--sunopt-ink)]">
                <tr>
                  <th className="px-4 py-3">Entwicklungsstand</th>
                  <th className="px-4 py-3">Hardware / Kontext</th>
                  <th className="px-4 py-3 text-right">Tanks</th>
                  <th className="px-4 py-3 text-right">Wall-Time</th>
                  <th className="px-4 py-3 text-right">Tanks/s</th>
                  <th className="px-4 py-3 text-right">Faktor zu 200 s</th>
                </tr>
              </thead>
              <tbody>
                {HISTORY.map((step, index) => (
                  <tr key={step.stage} className={`border-t border-[var(--sunopt-border-soft)] ${index % 2 ? "bg-[var(--sunopt-card)]" : "bg-[var(--sunopt-page)]"}`}>
                    <td className="px-4 py-3 align-top">
                      <div className="font-semibold text-[var(--sunopt-ink)]">{step.stage}</div>
                      <div className="mt-1 text-xs leading-5 text-[var(--sunopt-muted)]">{step.detail}</div>
                    </td>
                    <td className="px-4 py-3 align-top text-[var(--sunopt-muted)]">{step.hardware}</td>
                    <td className="px-4 py-3 text-right align-top tabular-nums">{step.tanks}</td>
                    <td className="px-4 py-3 text-right align-top tabular-nums">{secondsLabel(step.wallSeconds)} s</td>
                    <td className="px-4 py-3 text-right align-top font-mono tabular-nums">{tankRate(step).toFixed(4)}</td>
                    <td className="px-4 py-3 text-right align-top font-bold tabular-nums text-green-800">{vsOriginal(step).toFixed(2)}×</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-5 rounded-xl border border-green-300 bg-green-50 p-5 text-sm leading-7 text-green-950">
            <h3 className="font-bold">Produktive A/B-Messungen · Zielprofil der RTX 5060</h3>
            Beide produktiven CUDA-Wege fordern vor dem Sweep denselben SM-Takt von
            <strong> 3480 MHz </strong>und eine manuelle Lüftervorgabe von
            <strong> 70 % </strong>an. Ein separater Guard liest GPU-Takt und Temperatur
            während der Rechnung und setzt Takt- und Lüfterregelung danach wieder zurück.
            Kann der Windows-Treiber eine Einstellung nicht übernehmen oder bleibt
            der gemessene SM-Takt unter Last ausserhalb der Toleranz, wird der Lauf
            nicht als gültiger Fixed-Clock-Produktionsbenchmark akzeptiert.
            Die bereits archivierten B14Q-Werte wurden bei 3480 MHz gemessen;
            für ihre damalige Lüftereinstellung liegt hier kein Protokoll vor.
          </div>

          <div className="mt-4 rounded-xl border border-amber-300 bg-amber-50 px-5 py-4 text-sm leading-6 text-amber-950">
            <strong>Gleiches Mass, unterschiedliche Versuchsbedingungen:</strong> Durchsatz = Anzahl Tanks / gesamte Wall-Time.
            Faktoren = (Tanks/s) / 0,005. Eine einzelne Jahressimulation und ein kompletter Sweep
            sind damit rechnerisch vergleichbar, aber Messungen mit anderer Hardware und anderer
            Umgebung sind keine isolierten Optimierungsnachweise. Die Werte 200 s, 170 s und 5,2 s sind historische Näherungswerte.
            CUDA-Graph-Replays sind ein separater Prüfstand und werden nicht in die Tabelle hineingerechnet.
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
                kausal/sequenziell. Ihre Zeitschritte wurden nicht über Worker
                parallelisiert. Nach ersten Next.js-Optimierungen von ungefähr 200 s auf 170 s
                brachte Wrapper/Oxc/Cranelift die Laufzeit auf rund 27 s.
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
                28-Tank-Vergleich in einer frühen Messung auf 7.108 s.
                Im späteren B14Q-Fixed-Clock-Vergleich bei 3480 MHz
                benötigte A 6.264 s und B 3.550 s (jeweils 28 Tanks).
                Der künftige Produktions-Guard fordert zusätzlich 70 % Lüfter an
                und protokolliert den tatsächlichen SM-Takt. Historische Messungen
                werden dadurch nicht rückwirkend umetikettiert.
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
                  Next.js · ~200 → 170 s; Wrapper/Cranelift · 170 → 27 s
                  <br />
                  Laufzeitanalyse + Compile/Pre-calc/Redundanz · Ø 5.269 s
                  <br />
                  28 Tanks · seriell 148.455 s; 6 Worker 47.03 s
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


            <div className="rounded-2xl border border-[var(--sunopt-border)] bg-[var(--sunopt-panel)] p-6 text-[var(--sunopt-ink)]">

              <h3 className="text-xl font-bold">
                Ergebnis des Ansatz-A-Vergleichs
              </h3>

              <p className="mt-3 leading-7 text-[var(--sunopt-muted)]">
                Das CUDA-Graph-Replay sparte in seinem eigenen Prüfstand
                rund 0.435 s. Dieser isolierte Messwert darf nicht als
                Verbesserung des echten Jahreslaufs ausgegeben werden.
                Der eigentliche Produktionspfad A blieb die Referenz.
              </p>

              <p className="mt-4 leading-7 text-[var(--sunopt-muted)]">
                Der anschliessend entwickelte generische CUDA-Pfad A9–A11
                war numerisch korrekt, benötigte aber 182.099 s für den
                28-Tank-Jahreslauf gegenüber 7.356 s mit Original A.
                Er war damit rund 24.8-mal langsamer und wurde nicht
                produktiv übernommen. Die schnellere B-Lösung beruht auf
                der gezielten Aufbereitung der SunOpt-Physik.
              </p>

            </div>

          </div>

        </section>



          <div className="mt-6 rounded-2xl border border-orange-300 bg-orange-50 p-6">

            <div className="text-sm font-semibold uppercase tracking-wider text-orange-800">
              Historisches CUDA-Graph-Experiment
            </div>

            <h3 className="mt-2 text-xl font-bold text-orange-950">
              435 ms im Replay gespart – nicht im Jahreslauf belegt
            </h3>

            <p className="mt-3 leading-7 text-orange-950">
              Das Blackwell CUDA-Graph-Experiment zeigte 5.600 s →
              5.165 s, also ungefähr 434.8 ms beziehungsweise 7.8 %.
              In diesem Prüfstand wurden Launch- und Synchronisationskosten
              durch einen CUDA Graph reduziert. Für den produktiven Ablauf
              mit zustandsabhängigen Zeitschritten liegt jedoch kein
              entsprechender Nachweis vor. Eine automatische Übernahme
              dieses Speedups wäre eine unbelegte Aussage.
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
            Jeder B-Solver wurde gegen den bestehenden FP64-Referenzpfad
            geprüft. Exakte Verfahren müssen numerische Parität erreichen.
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
                oder Survival-Entscheide vom Referenzpfad abweichen.
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
              Solver gegen den Referenzpfad besteht. Deshalb werden Laufzeit,
              maximaler Cut-Level-Fehler, Survival-Mismatches und
              Jahresparität gemeinsam bewertet. Eine schnelle,
              physikalisch falsche Näherung ist kein Erfolg.
            </p>

          </div>

        </section>




      </div>

    </main>
  );
}
