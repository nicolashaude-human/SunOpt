// src/app/approach-b/page.tsx

type MeasurementRow = {
  stage: string;
  measurement: string;
  result: string;
  meaning: string;
};

type Lesson = {
  title: string;
  body: string;
  transfer: "Runtime" | "Compiler" | "Methodik";
};


const measurementRows: MeasurementRow[] = [
  {
    stage: "B14C",
    measurement: "Exact warp32 sample",
    result: "1.779 s sample kernel",
    meaning:
      "Die erste exakte Warp-Abbildung war korrekt, aber noch deutlich zu langsam. Korrektheit allein ist keine GPU-Optimierung.",
  },
  {
    stage: "B14E / B14F",
    measurement: "Warp parallel + precheck",
    result: "756 ms → 706.6 ms sample kernel",
    meaning:
      "Mehr Parallelität und billige Vorprüfungen helfen, lösen aber keine grundsätzlich serielle Suchstruktur.",
  },
  {
    stage: "B14G",
    measurement: "Warp-binary",
    result: "52.31 ms sample kernel; 13.56× vs. B14F",
    meaning:
      "Der grosse Sprung kam aus der algorithmischen Umformung der Suche, nicht aus Mikro-Tuning.",
  },
  {
    stage: "B14H",
    measurement: "Full annual offline replay",
    result: "894.37 ms causal kernel",
    meaning:
      "Zeigt das Potential des komprimierten Solvers bei grosser, bereits vorbereiteter Arbeit. Kein Produktions-Endwert.",
  },
  {
    stage: "B14I → B14K",
    measurement: "Production-shaped sequential replay",
    result: "5.257 s → 4.547 s wall",
    meaning:
      "Bei nur 28 Jobs pro Zeitschritt wird Occupancy zum Problem. Speculative depth-2 parallelisiert abhängige Entscheidungen.",
  },
  {
    stage: "B14L",
    measurement: "Packed transport",
    result: "3.363 s wall",
    meaning:
      "Nach dem Kernel wird Datenbewegung und Synchronisation zum dominanten Kostenblock.",
  },
  {
    stage: "B14M / B14P",
    measurement: "FP32 + mapped output",
    result: "1.398 s median production-shaped replay",
    meaning:
      "FP32 bringt viel, aber nur zusammen mit einer End-to-End-Paritätsprüfung darf daraus Produktionscode werden.",
  },
  {
    stage: "B14Q",
    measurement: "Real annual production A/B",
    result: "6.264 s → 3.550 s sweep; 1.765×",
    meaning:
      "Entscheidender Produktionsnachweis: echter Lockstep-Jahreslauf, fester Takt, gleiche Optima, praktisch identische Jahresbilanz.",
  },
];


const lessons: Lesson[] = [
  {
    title: "Algorithmus vor Arithmetik optimieren",
    body:
      "Der wichtigste Sprung entstand durch die Umformung der seriellen Causal-Suche in eine kompakte, komponierbare Zustandsabbildung. Ein schnellerer Kernel für denselben schlechten Algorithmus hätte diesen Gewinn nicht geliefert.",
    transfer: "Compiler",
  },
  {
    title: "Device-resident State ist Teil des Algorithmus",
    body:
      "Konstante Jahresdaten und wiederverwendeter Zustand gehören dauerhaft in den GPU-Speicher. Ein JIT, der jeden Funktionsaufruf isoliert betrachtet, verschenkt einen grossen Teil des GPU-Vorteils.",
    transfer: "Runtime",
  },
  {
    title: "Wall time ist wichtiger als Kernel time",
    body:
      "H2D, D2H, Host-Wait und Synchronisationspunkte können einen sehr schnellen Kernel fast vollständig neutralisieren. Optimiert wird deshalb die End-to-End-Latenz, nicht eine einzelne CUDA-Zahl.",
    transfer: "Methodik",
  },
  {
    title: "Kleine Batches brauchen eine andere Parallelisierung",
    body:
      "28 Tanks pro Zeitschritt reichen auf 30 SMs nicht für gute Auslastung. B14K zeigte, dass abhängige Entscheidungen spekulativ parallel ausgewertet werden können, um mehr Arbeit pro Block zu erzeugen.",
    transfer: "Compiler",
  },
  {
    title: "Pinned Memory ist keine universelle Optimierung",
    body:
      "Unter RTX 5060 + WDDM war Pageable im Fixed-Clock-Test klar besser: 3045.57 ms Median gegenüber 4430.08 ms für Pinned. Die Runtime muss Speicherstrategien messen oder autotunen statt dogmatisch wählen.",
    transfer: "Runtime",
  },
  {
    title: "FP32 nur mit physikalischem End-to-End-Vertrag",
    body:
      "Einzelne FP32-Causal-Jobs hatten messbare Cut-Abweichungen. Erst B14Q zeigte, dass daraus im vollständigen Jahreslauf nur minimale Abweichungen entstehen und die relevanten Optima identisch bleiben.",
    transfer: "Methodik",
  },
  {
    title: "Fallback-Pfad während der Einführung behalten",
    body:
      "Approach B blieb neben dem FP64-Pfad A aktivierbar. Das ermöglichte echte A/B-Parität im selben Produktionssystem und verhindert, dass Performance-Optimierung die Referenz zerstört.",
    transfer: "Runtime",
  },
  {
    title: "Benchmarks brauchen kontrollierte Hardwarebedingungen",
    body:
      "Warmup, 5 Wiederholungen, alternierende Reihenfolge, Median und Fixed-Clock-Telemetrie sind Teil der Messmethode. Unterschiede unter ungefähr 10 % werden nicht als realer Gewinn interpretiert.",
    transfer: "Methodik",
  },
];


const transferToA = [
  {
    title: "A1 · Runtime",
    items: [
      "Persistente GPU-Kontexte und device-residente Arrays.",
      "Kernel-Aufrufe über mehrere Calls hinweg planen statt jeden Call isoliert behandeln.",
      "Mapped Output / Event-Polling dort einsetzen, wo dadurch Synchronisation entfällt.",
      "Pageable vs. Pinned, Blockgrösse und Transferstrategie automatisch benchmarken.",
      "FP32/FP64-Spezialisierungen mit einer Referenzvariante koppeln.",
    ],
  },
  {
    title: "A2 · Compiler",
    items: [
      "Kernel fusion und Batch-Bildung über Call-Grenzen hinweg.",
      "Kleine, abhängige Jobs durch speculative execution verbreitern.",
      "Warp-level reductions und scans als eigene IR-Operationen modellieren.",
      "Wiederkehrende Schleifen und Recurrences erkennen, nicht nur Syntax zu CUDA übersetzen.",
      "Kostenmodell auf End-to-End-Walltime statt nur theoretischer FLOP-Zahl ausrichten.",
    ],
  },
  {
    title: "A3 · Algebraic Compression",
    items: [
      "Nach kleinen Zustandsabbildungen suchen, die einen langen Schleifenabschnitt zusammenfassen.",
      "Prüfen, ob diese Zustände assoziativ komponiert werden können.",
      "Wenn ja: serielle Schleife in tree reduction oder parallel scan umformen.",
      "Clamps, affine Teilstücke und monotone Suchräume als Optimierungsmuster erkennen.",
      "Nur anwenden, wenn die Transformation semantisch bewiesen oder gegen eine Referenz validiert werden kann.",
    ],
  },
];


const notAutomaticallyTransferable = [
  "Die konkrete Max-Affine-Zustandsform {a, b, c, max_input} ist SunOpt-spezifisch.",
  "Der exakte B14G/B14K-Suchbaum darf nicht als allgemeines Muster für beliebigen TypeScript-Code angenommen werden.",
  "Die akzeptable FP32-Fehlertoleranz ist eine Eigenschaft dieser physikalischen Anwendung, kein globaler Compiler-Default.",
  "Ein generischer Compiler kann Problemwissen nicht zuverlässig erfinden; er kann aber erkannte algebraische Strukturen ausnutzen.",
];


function MetricCard(
  props: {
    label: string;
    value: string;
    detail: string;
  },
) {
  return (
    <div
      className="rounded-2xl border border-[#d9c7aa] bg-white p-5 shadow-sm"
    >
      <div
        className="text-xs font-bold uppercase tracking-[0.18em] text-[#8b6e3f]"
      >
        {props.label}
      </div>

      <div
        className="mt-2 text-3xl font-black tracking-tight text-[#2f281e]"
      >
        {props.value}
      </div>

      <div
        className="mt-2 text-sm leading-6 text-[#6c5c47]"
      >
        {props.detail}
      </div>
    </div>
  );
}


function TransferBadge(
  props: {
    value: Lesson["transfer"];
  },
) {
  const className =
    props.value === "Runtime"
      ? "bg-[#e8f1e7] text-[#315b35]"
      : props.value === "Compiler"
        ? "bg-[#e7edf7] text-[#34547b]"
        : "bg-[#f6ead5] text-[#76551f]";

  return (
    <span
      className={[
        "inline-flex rounded-full px-2.5 py-1",
        "text-xs font-bold",
        className,
      ].join(" ")}
    >
      {props.value}
    </span>
  );
}


export default function ApproachBPage() {
  return (
    <main
      className="min-h-screen bg-[#fffaf2] text-[#2f281e]"
    >
      <div
        className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8"
      >
        <section
          className="overflow-hidden rounded-3xl border border-[#d9c7aa] bg-[#2f281e] px-6 py-9 text-[#fffaf2] shadow-sm sm:px-9"
        >
          <div
            className="max-w-4xl"
          >
            <div
              className="text-xs font-bold uppercase tracking-[0.22em] text-[#ddb114]"
            >
              SunOpt GPU Research · Approach B
            </div>

            <h1
              className="mt-4 text-4xl font-black tracking-tight sm:text-5xl"
            >
              Physics Compression:
              Lessons Learned
            </h1>

            <p
              className="mt-5 max-w-3xl text-base leading-7 text-[#eadfcf] sm:text-lg"
            >
              Approach B beschleunigt nicht einfach denselben Code.
              Er nutzt die mathematische Struktur des Causal-Problems,
              komprimiert lange Zustandsfolgen und reduziert Datenbewegung.
              Die wichtigste Frage nach B14Q lautet deshalb:
              Welche Erkenntnisse sind SunOpt-spezifisch – und welche
              können wir in den allgemeinen Approach A zurückführen?
            </p>
          </div>
        </section>


        <section
          className="mt-7 grid gap-4 md:grid-cols-2 xl:grid-cols-4"
        >
          <MetricCard
            label="Produktions-Sweep A"
            value="6.264 s"
            detail="FP64-Referenz, 28 Tanks × 35'040 Schritte, Fixed Clock 3480 MHz."
          />

          <MetricCard
            label="Produktions-Sweep B"
            value="3.550 s"
            detail="FP32 Approach B im echten GPU-Lockstep-Jahreslauf."
          />

          <MetricCard
            label="End-to-End Speedup"
            value="1.765×"
            detail="Vergleich des echten Jahres-Sweeps A gegen B, nicht eines isolierten Kernels."
          />

          <MetricCard
            label="Physik-Parität"
            value="praktisch identisch"
            detail="Gleiche relevanten Optima; unmet thermal demand in A und B = 0."
          />
        </section>


        <section
          className="mt-10 rounded-3xl border border-[#d9c7aa] bg-white p-6 shadow-sm sm:p-8"
        >
          <div
            className="flex flex-col gap-2"
          >
            <div
              className="text-xs font-bold uppercase tracking-[0.18em] text-[#8b6e3f]"
            >
              B14Q · finaler Produktionsnachweis
            </div>

            <h2
              className="text-2xl font-black tracking-tight"
            >
              Performance wurde akzeptiert, weil die Physik gleich blieb
            </h2>
          </div>

          <div
            className="mt-6 grid gap-4 lg:grid-cols-3"
          >
            <div
              className="rounded-2xl bg-[#f8f1e5] p-5"
            >
              <div
                className="text-sm font-bold"
              >
                Gleiche Optima
              </div>

              <p
                className="mt-2 text-sm leading-6 text-[#6c5c47]"
              >
                Minimale Energiekosten: 1400 L in A und B.
                Minimaler Netzbezug: 2000 L in A und B.
                Damit ändert FP32 die relevante Speicherentscheidung nicht.
              </p>
            </div>

            <div
              className="rounded-2xl bg-[#f8f1e5] p-5"
            >
              <div
                className="text-sm font-bold"
              >
                Grösste Jahresabweichung
              </div>

              <p
                className="mt-2 text-sm leading-6 text-[#6c5c47]"
              >
                Feed-in / self-consumption maximal 0.00461 kWh.
                Grid import maximal 0.0000504 kWh.
                Net energy cost maximal CHF 0.000369.
              </p>
            </div>

            <div
              className="rounded-2xl bg-[#f8f1e5] p-5"
            >
              <div
                className="text-sm font-bold"
              >
                Kontrollierter Benchmark
              </div>

              <p
                className="mt-2 text-sm leading-6 text-[#6c5c47]"
              >
                A und B liefen beide mit 3480 MHz SM clock,
                0 MHz Span. Der V/F-Zustand war damit keine
                versteckte Variable des Vergleichs.
              </p>
            </div>
          </div>
        </section>


        <section
          className="mt-10"
        >
          <div
            className="mb-5"
          >
            <div
              className="text-xs font-bold uppercase tracking-[0.18em] text-[#8b6e3f]"
            >
              Entwicklung
            </div>

            <h2
              className="mt-2 text-2xl font-black tracking-tight"
            >
              Von B14C bis B14Q
            </h2>

            <p
              className="mt-2 max-w-4xl text-sm leading-6 text-[#6c5c47]"
            >
              Diese Werte sind absichtlich mit ihrem Messkontext angegeben.
              Sample-Kernel, Offline-Replay, production-shaped Replay und
              echter Produktionslauf sind nicht direkt austauschbar.
            </p>
          </div>

          <div
            className="overflow-x-auto rounded-3xl border border-[#d9c7aa] bg-white shadow-sm"
          >
            <table
              className="min-w-[900px] w-full border-collapse text-left"
            >
              <thead
                className="bg-[#f3e6d2]"
              >
                <tr>
                  <th className="px-5 py-4 text-sm font-black">
                    Stage
                  </th>
                  <th className="px-5 py-4 text-sm font-black">
                    Messung
                  </th>
                  <th className="px-5 py-4 text-sm font-black">
                    Ergebnis
                  </th>
                  <th className="px-5 py-4 text-sm font-black">
                    Erkenntnis
                  </th>
                </tr>
              </thead>

              <tbody>
                {measurementRows.map(
                  (
                    row,
                    index,
                  ) => (
                    <tr
                      key={row.stage}
                      className={
                        index ===
                        measurementRows.length - 1
                          ? ""
                          : "border-b border-[#eee0ca]"
                      }
                    >
                      <td
                        className="whitespace-nowrap px-5 py-4 align-top text-sm font-black"
                      >
                        {row.stage}
                      </td>

                      <td
                        className="px-5 py-4 align-top text-sm text-[#5e513f]"
                      >
                        {row.measurement}
                      </td>

                      <td
                        className="whitespace-nowrap px-5 py-4 align-top text-sm font-bold"
                      >
                        {row.result}
                      </td>

                      <td
                        className="px-5 py-4 align-top text-sm leading-6 text-[#6c5c47]"
                      >
                        {row.meaning}
                      </td>
                    </tr>
                  ),
                )}
              </tbody>
            </table>
          </div>
        </section>


        <section
          className="mt-10"
        >
          <div
            className="mb-5"
          >
            <div
              className="text-xs font-bold uppercase tracking-[0.18em] text-[#8b6e3f]"
            >
              Lessons learned
            </div>

            <h2
              className="mt-2 text-2xl font-black tracking-tight"
            >
              Was Approach B tatsächlich gezeigt hat
            </h2>
          </div>

          <div
            className="grid gap-4 md:grid-cols-2"
          >
            {lessons.map(
              lesson => (
                <article
                  key={lesson.title}
                  className="rounded-2xl border border-[#d9c7aa] bg-white p-6 shadow-sm"
                >
                  <div
                    className="flex items-start justify-between gap-4"
                  >
                    <h3
                      className="text-lg font-black leading-6"
                    >
                      {lesson.title}
                    </h3>

                    <TransferBadge
                      value={lesson.transfer}
                    />
                  </div>

                  <p
                    className="mt-3 text-sm leading-6 text-[#6c5c47]"
                  >
                    {lesson.body}
                  </p>
                </article>
              ),
            )}
          </div>
        </section>


        <section
          className="mt-10"
        >
          <div
            className="mb-5"
          >
            <div
              className="text-xs font-bold uppercase tracking-[0.18em] text-[#8b6e3f]"
            >
              Rücktransfer
            </div>

            <h2
              className="mt-2 text-2xl font-black tracking-tight"
            >
              Was Approach A von B übernehmen soll
            </h2>

            <p
              className="mt-2 max-w-4xl text-sm leading-6 text-[#6c5c47]"
            >
              Ziel ist nicht, die SunOpt-Sonderlösung in den allgemeinen
              Compiler zu kopieren. Ziel ist, die wiederverwendbaren
              Optimierungsmuster als Runtime- und Compilerfähigkeiten
              in Approach A einzubauen.
            </p>
          </div>

          <div
            className="grid gap-5 lg:grid-cols-3"
          >
            {transferToA.map(
              section => (
                <article
                  key={section.title}
                  className="rounded-3xl border border-[#d9c7aa] bg-white p-6 shadow-sm"
                >
                  <h3
                    className="text-xl font-black"
                  >
                    {section.title}
                  </h3>

                  <ul
                    className="mt-4 space-y-3"
                  >
                    {section.items.map(
                      item => (
                        <li
                          key={item}
                          className="flex gap-3 text-sm leading-6 text-[#6c5c47]"
                        >
                          <span
                            className="mt-[0.58rem] h-1.5 w-1.5 shrink-0 rounded-full bg-[#ddb114]"
                          />

                          <span>
                            {item}
                          </span>
                        </li>
                      ),
                    )}
                  </ul>
                </article>
              ),
            )}
          </div>
        </section>


        <section
          className="mt-10 grid gap-5 lg:grid-cols-[1.15fr_0.85fr]"
        >
          <article
            className="rounded-3xl border border-[#d9c7aa] bg-[#2f281e] p-7 text-[#fffaf2] shadow-sm"
          >
            <div
              className="text-xs font-bold uppercase tracking-[0.18em] text-[#ddb114]"
            >
              Generalisierbares Kernmuster
            </div>

            <h2
              className="mt-3 text-2xl font-black"
            >
              Lange Recurrence → kleiner State → Compose → Parallel Scan
            </h2>

            <div
              className="mt-5 rounded-2xl bg-black/20 p-5 font-mono text-sm leading-7 text-[#f4eadb]"
            >
              <div>TypeScript loop / recurrence</div>
              <div>↓</div>
              <div>state-transform analysis</div>
              <div>↓</div>
              <div>compact composable state</div>
              <div>↓</div>
              <div>compose / scan / speculative tree</div>
              <div>↓</div>
              <div>CUDA execution</div>
            </div>

            <p
              className="mt-5 text-sm leading-6 text-[#e4d7c5]"
            >
              Genau hier liegt das grösste Forschungsresultat aus B:
              Ein allgemeiner TS→GPU-Compiler sollte nicht nur Code
              übersetzen, sondern versuchen, wiederkehrende
              Zustandsabbildungen algebraisch zu erkennen und
              parallelisierbar zu machen.
            </p>
          </article>

          <article
            className="rounded-3xl border border-[#d9c7aa] bg-white p-7 shadow-sm"
          >
            <div
              className="text-xs font-bold uppercase tracking-[0.18em] text-[#8b6e3f]"
            >
              Nicht automatisch übertragbar
            </div>

            <h2
              className="mt-3 text-2xl font-black"
            >
              Grenze zwischen A und B
            </h2>

            <ul
              className="mt-5 space-y-4"
            >
              {notAutomaticallyTransferable.map(
                item => (
                  <li
                    key={item}
                    className="flex gap-3 text-sm leading-6 text-[#6c5c47]"
                  >
                    <span
                      className="mt-2 h-2 w-2 shrink-0 rounded-sm bg-[#8b6e3f]"
                    />

                    <span>
                      {item}
                    </span>
                  </li>
                ),
              )}
            </ul>
          </article>
        </section>


        <section
          className="mt-10 rounded-3xl border border-[#d9c7aa] bg-[#f3e6d2] p-7"
        >
          <div
            className="text-xs font-bold uppercase tracking-[0.18em] text-[#8b6e3f]"
          >
            Entscheidung nach B14Q
          </div>

          <h2
            className="mt-3 text-2xl font-black"
          >
            Approach B ist Produktionsstandard; Approach A bleibt Referenz und Forschungsplattform
          </h2>

          <p
            className="mt-4 max-w-5xl text-sm leading-6 text-[#5e513f]"
          >
            B14Q klassifiziert A und B als praktisch identisch.
            Die relevanten Speicheroptima bleiben gleich, während
            der echte 28-Tank-Jahres-Sweep von 6.264 s auf 3.550 s
            sinkt. Der FP64-Pfad A bleibt deshalb als expliziter
            Referenz- und Debug-Fallback erhalten. Die allgemeinen
            Erkenntnisse aus B werden schrittweise in A zurückgeführt,
            ohne die problem-spezifische Physik-Kompression als
            universelle Compilerregel auszugeben.
          </p>
        </section>
      </div>
    </main>
  );
}
