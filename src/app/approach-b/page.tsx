// src/app/approach-b/page.tsx

// Messkontexte bleiben ausdrücklich getrennt: Ein GPU-Kernel-Timer ist
// keine Wall-Time des vollständigen 28-Tank-Jahreslaufs.
type MeasurementRow = {
  stage: string;
  context: string;
  measurement: string;
  reference: string;
  result: string;
  meaning: string;
};

type Lesson = {
  title: string;
  body: string;
  transfer: "Runtime" | "Algorithmus" | "Methodik";
};

const measurementRows: MeasurementRow[] = [
  {
    stage: "B14C",
    context: "Sample · nur Kernel",
    measurement: "Exact warp32",
    reference: "Keine gleichartige Vergleichsmessung dokumentiert",
    result: "1.779 s Kernelzeit",
    meaning:
      "Die erste exakte Warp-Abbildung lieferte korrekte Resultate, nutzte die GPU aber noch nicht effizient. Der Wert ist ein Ausgangspunkt innerhalb der Sample-Reihe, keine Jahreslaufzeit.",
  },
  {
    stage: "B14E → B14F",
    context: "Sample · nur Kernel",
    measurement: "Warp parallel + precheck",
    reference: "B14E: 756 ms Kernelzeit",
    result: "B14F: 706.6 ms Kernelzeit",
    meaning:
      "Zusätzliche Parallelität und günstige Vorprüfungen senkten die Sample-Kernelzeit um rund 6.5 %. Das ist ein kleiner Fortschritt, der unter einer konservativen 10-%-Schwelle nicht als robuster Gewinn gelten sollte. Die teure Suchstruktur blieb bestehen.",
  },
  {
    stage: "B14G",
    context: "Sample · nur Kernel",
    measurement: "Warp-binary",
    reference: "B14F: 706.6 ms Kernelzeit",
    result: "B14G: 52.31 ms Kernelzeit",
    meaning:
      "Bei dieser Sample-Messung beträgt das Verhältnis 706.6 / 52.31 = 13.51×. Der entscheidende Schritt war die Umformung der Sucharbeit, nicht nur eine höhere Taktrate oder ein anderer Speicherpuffer. Nicht auf den Jahreslauf hochrechnen.",
  },
  {
    stage: "B14H",
    context: "Offline-Replay · nur Kernel",
    measurement: "Full annual offline replay",
    reference: "Keine gepaarte Offline-Referenz dokumentiert",
    result: "894.37 ms Causal-Kernel",
    meaning:
      "Viele vorbereitete Causal-Jobs konnten in grossen GPU-Batches abgearbeitet werden. Das illustriert die Rechenleistung unter günstiger Parallelität, enthält aber weder die schrittweise Jahressteuerung noch deren Host-Wartezeiten.",
  },
  {
    stage: "B14I → B14K",
    context: "Production-shaped Replay · Wall",
    measurement: "Sequential replay / speculative depth-2",
    reference: "B14I: 5.257 s Wall-Time",
    result: "B14K: 4.547 s Wall-Time",
    meaning:
      "Der Replay-Aufbau bearbeitet nur 28 Tanks je Zeitschritt. Dadurch sind selbst auf 30 SMs nicht beliebig viele Jobs gleichzeitig verfügbar. Spekulative Auswertung unabhängiger Entscheidungszweige erhöht die Arbeit pro Schritt und spart hier rund 13.5 % Wall-Time.",
  },
  {
    stage: "B14L",
    context: "Production-shaped Replay · Wall",
    measurement: "Packed transport",
    reference: "B14K: 4.547 s (Replay-Orientierung)",
    result: "B14L: 3.363 s Wall-Time",
    meaning:
      "Nach der algorithmischen Beschleunigung gewinnen Datentransfers, Aufrufhäufigkeit und Host-/GPU-Synchronisation an Gewicht. Durch das Packen der Übertragungen sinkt die gemessene Replay-Wall-Time. Da kein vollständig gepaarter Messlauf dokumentiert ist, ist die Differenz zu B14K kein isolierter Transfer-Speedup.",
  },
  {
    stage: "B14M / B14P",
    context: "Production-shaped Replay · Median",
    measurement: "FP32 + mapped output",
    reference: "B14L: 3.363 s (nur zeitliche Orientierung)",
    result: "B14P: 1.398 s Replay-Median",
    meaning:
      "Der schnellere FP32-Rechenpfad und die veränderte Ausgabe sind gemeinsam im Ergebnis enthalten; ein sauberer Einzelbeitrag lässt sich daraus nicht ableiten. Der Vergleich zu B14L ist nicht als gepaarter A/B-Benchmark ausgewiesen. Die Genauigkeit musste erst im echten Jahreslauf geprüft werden.",
  },
  {
    stage: "B14Q",
    context: "Produktion · 28-Tank-Jahres-Sweep · Wall",
    measurement: "Echter A/B-Jahresvergleich · Fixed Clock",
    reference: "Original A · FP64: 6.264 s Sweep-Wall",
    result: "Approach B · FP32: 3.550 s Sweep-Wall",
    meaning:
      "Hier stimmen Workload, Messgrösse und fixierter GPU-Takt überein: 28 Tanks × 35'040 Viertelstunden, 3480 MHz. B ist 1.765× schneller. Gleichzeitig blieben die ausgewiesenen Speicheroptima gleich und die Jahresabweichungen sehr klein. Nur diese Zeile belegt den Produktionsgewinn.",
  },
];

const lessons: Lesson[] = [
  {
    title: "Die Suchstruktur war wichtiger als der Takt",
    body:
      "B14G reduzierte den Aufwand einer Sample-Berechnung drastisch, weil die Suche anders organisiert wurde. Eine GPU mit mehr Recheneinheiten allein hätte die wiederholte, abhängige Arbeit nicht beseitigt. Der gemessene Gewinn gehört zu genau dieser mathematischen Struktur und ist keine allgemeine Eigenschaft beliebiger Funktionen.",
    transfer: "Algorithmus",
  },
  {
    title: "Daten im GPU-Speicher wiederverwenden",
    body:
      "Historische Eingabereihen ändern sich innerhalb eines Jahreslaufs nicht. Sie bei jedem Viertelstundenschritt erneut zu übertragen wäre unnötig. Device-residente Daten und kompakte dynamische Eingaben verringern Speicherverkehr; das ist eine grundsätzlich nützliche Laufzeittechnik, garantiert aber noch keinen schnelleren Gesamtlauf.",
    transfer: "Runtime",
  },
  {
    title: "Kernelzeit und Wall-Time strikt trennen",
    body:
      "Eine Kernelmessung zählt nur die GPU-Ausführung. Die Wall-Time enthält auch das Bereitstellen der Daten, Treiberarbeit, Aufrufe, Synchronisation und Warten auf Resultate. B14H sieht deshalb sehr schnell aus, ist aber kein Beweis für einen Jahreslauf unter einer Sekunde. Der entscheidende Vergleich ist B14Q gegen A im gleichen Produktionsablauf.",
    transfer: "Methodik",
  },
  {
    title: "Nur 28 Tanks gleichzeitig: Auslastung ist schwierig",
    body:
      "Ein Jahreslauf muss seine Zeitschritte nacheinander bearbeiten, auch wenn die 28 Tanks innerhalb eines Schritts unabhängig sind. Auf der RTX 5060 mit 30 SMs kann das zu wenig gleichzeitig nutzbare Arbeit sein. B14K erzeugte zusätzliche Parallelität durch die spekulative Prüfung möglicher Suchentscheidungen – mit zusätzlichem Rechenaufwand, der sich in diesem Fall gelohnt hat.",
    transfer: "Algorithmus",
  },
  {
    title: "Pinned Memory war auf diesem Testsystem langsamer",
    body:
      "Im kontrollierten Test mit RTX 5060 unter Windows/WDDM lag der Median für Pageable bei 3045.57 ms und für Pinned bei 4430.08 ms. Pinned kann grössere Transfers beschleunigen und echte asynchrone Kopien ermöglichen, verursacht aber je nach Zugriffsmuster zusätzliche Kosten. Aus diesem Messpaar lässt sich die genaue Ursache des Rückstands nicht isolieren; eine pauschale Pinned-Regel wäre falsch.",
    transfer: "Runtime",
  },
  {
    title: "FP32 muss am Jahresergebnis beurteilt werden",
    body:
      "Einzelne FP32-Causal-Entscheidungen wichen vom FP64-Wert ab. Das war zunächst ein Warnsignal, kein Freibrief. B14Q überprüfte deshalb die für die Anwendung wichtigen Jahresgrössen und Speicheroptima. Diese kontrollierte Toleranz gehört zum SunOpt-Modell und darf nicht als allgemeine Genauigkeitszusage für andere Funktionen gelten.",
    transfer: "Methodik",
  },
  {
    title: "Original A als unveränderte Referenz erhalten",
    body:
      "Ein schnellerer Algorithmus braucht einen verlässlichen Vergleichspartner. Der FP64-Produktionspfad blieb deshalb erhalten, während Approach B erprobt wurde. So konnten identische Eingaben, Jahresbilanzen und gewählte Speichervolumina gegenübergestellt werden, ohne das geprüfte Verhalten von A aufzugeben.",
    transfer: "Runtime",
  },
  {
    title: "Messbedingungen dokumentieren statt Zahlen sammeln",
    body:
      "GPU-Takt, Warmup, Wiederholungszahl, Messreihenfolge, Datengrösse und gemessene Zeitart beeinflussen das Ergebnis. Beim Fixed-Clock-Vergleich wurden 3480 MHz verwendet. Ein Sample-Kernel, ein Offline-Replay und ein echter Jahres-Sweep sind drei unterschiedliche Prüfstände und dürfen nicht zu einer einzigen Fortschrittskurve verbunden werden.",
    transfer: "Methodik",
  },
];

const transferReview = [
  {
    title: "Die eigentliche Forschungsfrage",
    items: [
      "Enthielten die B-Experimente zufällig Methoden, die nicht von der konkreten Funktion f(x) abhängen?",
      "Kandidaten waren insbesondere Laufzeitorganisation, Datenübertragung, Warp-Operationen und die Erkennung geeigneter algebraischer Strukturen.",
      "Funktionsunabhängigkeit bedeutet nicht, dass eine Methode jede beliebige Funktion beschleunigt.",
    ],
  },
  {
    title: "Was der generische Versuch ergab",
    items: [
      "A9 übertrug TypeScript-Scopes in einen generischen CUDA-Ausführungspfad. Die Ergebnisse waren korrekt, die Jahresausführung aber sehr langsam.",
      "A10 (schneller Integer-Rest) verlangsamte den gemessenen GPU-Kurztest: 1.938 s → 2.272 s.",
      "A11 beschleunigte den generischen Kurztest von 1.980 s auf 1.761 s. Im vollständigen Jahr benötigte A11 trotzdem 182.099 s statt 7.356 s mit Original A.",
    ],
  },
  {
    title: "Entscheidung: kein erfolgreicher Rücktransfer",
    items: [
      "Der getestete generische Weg war im Jahreslauf 24.8-mal langsamer als Original A.",
      "B14Q belegt eine Beschleunigung des problemspezifischen B-Pfads – nicht des generischen Compilers.",
      "Die Experimente A9–A11 wurden deshalb nicht als neue produktive A-Implementierung übernommen. Original A und sein Schalter bleiben unverändert.",
    ],
  },
];

const notAutomaticallyTransferable = [
  "Die konkrete Max-Affine-Zustandsform und der B14G/B14K-Suchablauf verwenden Eigenschaften der SunOpt-Berechnung.",
  "Eine allgemeine AST-Übersetzung erkennt diese mathematischen Eigenschaften nicht automatisch.",
  "FP32-Abweichungen sind nur nach Prüfung der für SunOpt relevanten Jahresgrössen akzeptabel.",
  "Auch allgemein verwendbare CUDA-Techniken sind erst dann ein Gewinn für A, wenn ein vergleichbarer Jahreslauf das nachweist.",
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
      className="rounded-2xl border border-[var(--sunopt-border)] bg-[var(--sunopt-card)] p-5 shadow-sm"
    >
      <div
        className="text-xs font-bold uppercase tracking-[0.18em] text-[var(--sunopt-soft-text)]"
      >
        {props.label}
      </div>

      <div
        className="mt-2 text-3xl font-black tracking-tight text-[var(--sunopt-ink)]"
      >
        {props.value}
      </div>

      <div
        className="mt-2 text-sm leading-6 text-[var(--sunopt-muted)]"
      >
        {props.detail}
      </div>
    </div>
  );
}


function TopicBadge(
  props: {
    value: Lesson["transfer"];
  },
) {
  const className =
    props.value === "Runtime"
      ? "bg-[var(--sunopt-green-background)] text-[var(--sunopt-green-ink)]"
      : props.value === "Algorithmus"
        ? "bg-[#e7edf7] text-[#34547b]"
        : "bg-[var(--sunopt-panel)] text-[#76551f]";

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
      className="min-h-screen bg-[var(--sunopt-page)] text-[var(--sunopt-ink)]"
    >
      <div
        className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8"
      >
        <section
          className="overflow-hidden rounded-3xl border border-[var(--sunopt-border)] bg-[var(--sunopt-panel)] px-6 py-9 text-[var(--sunopt-ink)] shadow-sm sm:px-9"
        >
          <div
            className="max-w-4xl"
          >
            <div
              className="text-xs font-bold uppercase tracking-[0.22em] text-[var(--sunopt-accent-dark)]"
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
              className="mt-5 max-w-3xl text-base leading-7 text-[var(--sunopt-muted)] sm:text-lg"
            >
              Approach B beschleunigt nicht einfach denselben Code.
              Er nutzt die mathematische Struktur des Causal-Problems,
              komprimiert lange Zustandsfolgen und reduziert Datenbewegung.
              Die B14-Versuche zeigen, warum spezielle Mathematik
              und sorgfältig gemessene GPU-Ausführung schneller sein
              können als eine direkte Übersetzung des ursprünglichen Codes.
              Der spätere Versuch, die Erkenntnisse auf einen generischen
              Compiler zu übertragen, erreichte das Leistungsziel nicht.
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
          className="mt-10 rounded-3xl border border-[var(--sunopt-border)] bg-[var(--sunopt-card)] p-6 shadow-sm sm:p-8"
        >
          <div
            className="flex flex-col gap-2"
          >
            <div
              className="text-xs font-bold uppercase tracking-[0.18em] text-[var(--sunopt-soft-text)]"
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
              className="rounded-2xl bg-[var(--sunopt-bg)] p-5"
            >
              <div
                className="text-sm font-bold"
              >
                Gleiche Optima
              </div>

              <p
                className="mt-2 text-sm leading-6 text-[var(--sunopt-muted)]"
              >
                Minimale Energiekosten: 1400 L in A und B.
                Minimaler Netzbezug: 2000 L in A und B.
                Damit ändert FP32 die relevante Speicherentscheidung nicht.
              </p>
            </div>

            <div
              className="rounded-2xl bg-[var(--sunopt-bg)] p-5"
            >
              <div
                className="text-sm font-bold"
              >
                Grösste Jahresabweichung
              </div>

              <p
                className="mt-2 text-sm leading-6 text-[var(--sunopt-muted)]"
              >
                Feed-in / self-consumption maximal 0.00461 kWh.
                Grid import maximal 0.0000504 kWh.
                Net energy cost maximal CHF 0.000369.
              </p>
            </div>

            <div
              className="rounded-2xl bg-[var(--sunopt-bg)] p-5"
            >
              <div
                className="text-sm font-bold"
              >
                Kontrollierter Benchmark
              </div>

              <p
                className="mt-2 text-sm leading-6 text-[var(--sunopt-muted)]"
              >
                Der historische B14Q-Vergleich lief mit 3480 MHz SM-Takt
                und 0 MHz Taktspanne. Eine feste Lüftereinstellung ist für
                diese ältere Messung nicht dokumentiert. Künftige produktive
                A/B-Läufe fordern 3480 MHz und 70 % Lüfterdrehzahl mit
                NVML an und protokollieren den SM-Takt unter GPU-Last.
                Wenn Windows/WDDM die Einstellungen verweigert oder der
                tatsächliche Takt nicht bestätigt wird, zählt der Lauf
                ausdrücklich nicht als gültiger Fixed-Clock-Vergleich.
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
              className="text-xs font-bold uppercase tracking-[0.18em] text-[var(--sunopt-soft-text)]"
            >
              Entwicklung
            </div>

            <h2
              className="mt-2 text-2xl font-black tracking-tight"
            >
              Von B14C bis B14Q
            </h2>

            <p
              className="mt-2 max-w-4xl text-sm leading-6 text-[var(--sunopt-muted)]"
            >
              Jede Zeile nennt jetzt eine Messgrösse und eine Referenz.
              Nur Messungen mit demselben Arbeitsumfang und derselben Zeitart
              dürfen direkt verglichen werden. Eine GPU-Kernelzeit ist keine
              Wall-Time; ein vorbereitetes Offline-Replay ist kein
              vollständiger Jahreslauf. Nicht belegte Referenzen bleiben
              ausdrücklich als solche gekennzeichnet.
            </p>
          </div>

          <div className="mb-5 grid gap-3 md:grid-cols-3">
            <div className="rounded-2xl border border-[var(--sunopt-border)] bg-[var(--sunopt-card)] p-4">
              <h3 className="font-black">Sample-Kernel</h3>
              <p className="mt-2 text-sm leading-6 text-[var(--sunopt-muted)]">
                Misst einen ausgewählten GPU-Rechenkern mit vorbereiteten Daten.
                Kein vollständiger Jahreslauf.
              </p>
            </div>
            <div className="rounded-2xl border border-[var(--sunopt-border)] bg-[var(--sunopt-card)] p-4">
              <h3 className="font-black">Replay-Wall-Time</h3>
              <p className="mt-2 text-sm leading-6 text-[var(--sunopt-muted)]">
                Misst den gesamten Ablauf des jeweiligen Replay-Prüfstands.
                Ähnliche Replays sind nicht automatisch identische Messungen.
              </p>
            </div>
            <div className="rounded-2xl border border-[var(--sunopt-green-border)] bg-[var(--sunopt-green-background)] p-4">
              <h3 className="font-black text-[var(--sunopt-green-ink)]">Produktions-Sweep · Wall-Time</h3>
              <p className="mt-2 text-sm leading-6 text-[var(--sunopt-green-ink)]">
                Misst die tatsächliche Jahressimulation aller 28 Tanks.
                Nur der kontrollierte B14Q-A/B-Vergleich belegt hier 1.765×.
              </p>
            </div>
          </div>

          <div
            className="overflow-x-auto rounded-3xl border border-[var(--sunopt-border)] bg-[var(--sunopt-card)] shadow-sm"
          >
            <table
              className="min-w-[1240px] w-full border-collapse text-left"
            >
              <thead
                className="bg-[var(--sunopt-panel-hover)]"
              >
                <tr>
                  <th className="px-5 py-4 text-sm font-black">
                    Stage
                  </th>
                  <th className="px-5 py-4 text-sm font-black">
                    Messkontext / Messung
                  </th>
                  <th className="px-5 py-4 text-sm font-black">
                    Referenz
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
                          : "border-b border-[var(--sunopt-border)]"
                      }
                    >
                      <td
                        className="whitespace-nowrap px-5 py-4 align-top text-sm font-black"
                      >
                        {row.stage}
                      </td>

                      <td className="min-w-[195px] px-5 py-4 align-top text-sm text-[var(--sunopt-muted)]">
                        <span className="mb-2 inline-block rounded-md border border-[var(--sunopt-border)] bg-[var(--sunopt-panel)] px-2 py-1 text-xs font-semibold text-[var(--sunopt-soft-text)]">
                          {row.context}
                        </span>
                        <div>{row.measurement}</div>
                      </td>

                      <td className="min-w-[185px] px-5 py-4 align-top text-sm leading-6 text-[var(--sunopt-muted)]">
                        {row.reference}
                      </td>

                      <td className="min-w-[160px] px-5 py-4 align-top text-sm font-bold">
                        {row.result}
                      </td>

                      <td className="min-w-[310px] px-5 py-4 align-top text-sm leading-6 text-[var(--sunopt-muted)]">
                        {row.meaning}
                      </td>
                    </tr>
                  ),
                )}
              </tbody>
            </table>
          </div>
        </section>


        <section className="mt-10 rounded-3xl border border-[var(--sunopt-border)] bg-[var(--sunopt-card)] p-6 shadow-sm sm:p-8">
          <div className="text-xs font-bold uppercase tracking-[0.18em] text-[var(--sunopt-soft-text)]">
            Datentransfer und Windows-Treiber
          </div>
          <h2 className="mt-2 text-2xl font-black">Pinned, Pageable und WDDM – was steckt dahinter?</h2>
          <p className="mt-3 max-w-5xl text-sm leading-7 text-[var(--sunopt-muted)]">
            Die GPU hat eigenen Grafikspeicher (VRAM). Die Daten im Arbeitsspeicher
            des Computers müssen für viele CUDA-Aufgaben zuerst übertragen werden.
            Wie der CPU-Speicher bereitgestellt wird und wie Windows die GPU-Aufträge
            verwaltet, beeinflusst die tatsächlich gemessene Zeit.
          </p>
          <div className="mt-5 grid gap-4 lg:grid-cols-2">
            <article className="rounded-2xl border border-[var(--sunopt-border)] bg-[var(--sunopt-page)] p-5">
              <h3 className="text-lg font-black">Pageable Memory – normaler Arbeitsspeicher</h3>
              <p className="mt-2 text-sm leading-7 text-[var(--sunopt-muted)]">
                Windows darf diese Speicherseiten bei Bedarf verschieben oder auslagern.
                Für einen üblichen Transfer zur GPU richtet der CUDA-Treiber deshalb
                einen Zwischenschritt über temporär fixierten Speicher ein.
                Das kostet eine zusätzliche Kopie, ist bei kleinen Transfers oder
                bestimmten Host-Abläufen aber nicht zwangsläufig langsamer.
              </p>
            </article>
            <article className="rounded-2xl border border-[var(--sunopt-border)] bg-[var(--sunopt-page)] p-5">
              <h3 className="text-lg font-black">Pinned Memory – festgehaltener Arbeitsspeicher</h3>
              <p className="mt-2 text-sm leading-7 text-[var(--sunopt-muted)]">
                Diese Speicherseiten dürfen während ihrer Verwendung nicht ausgelagert
                werden. Die GPU kann Daten damit in geeigneten Fällen effizienter
                übertragen; für echte asynchrone CUDA-Kopien ist page-locked Host-Speicher
                der übliche Weg. Reservierung, Registrierung, Zugriffsmuster und
                Synchronisation können den Vorteil jedoch wieder aufzehren.
              </p>
            </article>
          </div>
          <div className="mt-4 rounded-2xl border border-[var(--sunopt-green-border)] bg-[var(--sunopt-green-background)] p-5">
            <h3 className="text-base font-black text-[var(--sunopt-green-ink)]">Unser gemessener Vergleich</h3>
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              <div className="rounded-xl bg-[var(--sunopt-card)] p-4">
                <div className="text-sm font-semibold text-[var(--sunopt-muted)]">Pageable · Median</div>
                <div className="mt-1 text-2xl font-black">3045.57 ms</div>
              </div>
              <div className="rounded-xl bg-[var(--sunopt-card)] p-4">
                <div className="text-sm font-semibold text-[var(--sunopt-muted)]">Pinned · Median</div>
                <div className="mt-1 text-2xl font-black">4430.08 ms</div>
              </div>
            </div>
            <p className="mt-3 text-sm leading-7 text-[var(--sunopt-green-ink)]">
              Im getesteten Fixed-Clock-Ablauf war Pageable rund 31.3 % kürzer in der
              Wall-Time. Das widerspricht nicht dem grundsätzlichen Bandbreitenvorteil
              von Pinned Memory bei grossen Transfers: Wir haben einen vollständigen
              Ablauf gemessen, nicht ausschliesslich die PCIe-Transferbandbreite.
              Welcher Einzelaufwand hier ausschlaggebend war, wurde nicht isoliert.
            </p>
          </div>
          <div className="mt-5 rounded-2xl border border-[var(--sunopt-border)] bg-[var(--sunopt-panel)] p-5">
            <h3 className="text-lg font-black">WDDM – Windows Display Driver Model</h3>
            <p className="mt-2 text-sm leading-7 text-[var(--sunopt-muted)]">
              WDDM ist die Windows-Infrastruktur, die Grafikspeicher und GPU-Aufträge
              zwischen Anwendungen verwaltet. Auch CUDA-Arbeit auf unserer GeForce
              RTX 5060 läuft unter diesem Treibermodell. Die Verwaltung sorgt für
              gemeinsame Nutzung und Stabilität, kann aber Warteschlangen,
              Launch-Latenzen und Synchronisationskosten verursachen. Gerade bei
              35&apos;040 aufeinanderfolgenden Zeitschritten sind viele kleine
              Aufträge deshalb relevant. Dass WDDM allein den Unterschied zwischen
              Pinned und Pageable verursacht, ist durch unseren Test nicht belegt.
            </p>
          </div>
          <p className="mt-4 text-xs leading-6 text-[var(--sunopt-soft-text)]">
            Technische Grundlagen: <a className="underline" href="https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html" target="_blank" rel="noreferrer">NVIDIA CUDA Programming Guide – Host Memory</a>{" · "}
            <a className="underline" href="https://developer.nvidia.com/blog/leveling-up-cuda-performance-on-wsl2-with-new-enhancements/" target="_blank" rel="noreferrer">NVIDIA – WDDM und Launch-Overhead</a>{" · "}
            <a className="underline" href="https://learn.microsoft.com/en-us/windows-hardware/drivers/display/benefits-of-the-windows-vista-and-later-display-driver-model" target="_blank" rel="noreferrer">Microsoft – WDDM</a>.
          </p>
        </section>


        <section
          className="mt-10"
        >
          <div
            className="mb-5"
          >
            <div
              className="text-xs font-bold uppercase tracking-[0.18em] text-[var(--sunopt-soft-text)]"
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
                  className="rounded-2xl border border-[var(--sunopt-border)] bg-[var(--sunopt-card)] p-6 shadow-sm"
                >
                  <div
                    className="flex items-start justify-between gap-4"
                  >
                    <h3
                      className="text-lg font-black leading-6"
                    >
                      {lesson.title}
                    </h3>

                    <TopicBadge
                      value={lesson.transfer}
                    />
                  </div>

                  <p
                    className="mt-3 text-sm leading-6 text-[var(--sunopt-muted)]"
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
              className="text-xs font-bold uppercase tracking-[0.18em] text-[var(--sunopt-soft-text)]"
            >
              Forschungsbilanz A9–A11
            </div>

            <h2
              className="mt-2 text-2xl font-black tracking-tight"
            >
              Welche Erkenntnisse aus B auf A übertragbar waren
            </h2>

            <p
              className="mt-2 max-w-4xl text-sm leading-6 text-[var(--sunopt-muted)]"
            >
              Unsere Frage war, ob die Experimente von B zufällig
              Methoden aufgedeckt hatten, die unabhängig von der konkreten
              Funktion f(x) funktionieren und Original A beschleunigen.
              Wir haben das untersucht – aber der generische CUDA-Pfad
              wurde im vollständigen Jahreslauf nicht schneller als A.
              Die folgenden Karten dokumentieren den Versuch und seine
              Entscheidung, keine neue Entwicklungszusage.
            </p>
          </div>

          <div
            className="grid gap-5 lg:grid-cols-3"
          >
            {transferReview.map(
              section => (
                <article
                  key={section.title}
                  className="rounded-3xl border border-[var(--sunopt-border)] bg-[var(--sunopt-card)] p-6 shadow-sm"
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
                          className="flex gap-3 text-sm leading-6 text-[var(--sunopt-muted)]"
                        >
                          <span
                            className="mt-[0.58rem] h-1.5 w-1.5 shrink-0 rounded-full bg-[var(--sunopt-accent)]"
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
            className="rounded-3xl border border-[var(--sunopt-border)] bg-[var(--sunopt-card)] p-7 text-[var(--sunopt-ink)] shadow-sm"
          >
            <div
              className="text-xs font-bold uppercase tracking-[0.18em] text-[var(--sunopt-accent-dark)]"
            >
              Forschungshypothese – nicht als allgemeiner Compiler nachgewiesen
            </div>

            <h2
              className="mt-3 text-2xl font-black"
            >
              Wann lässt sich eine serielle Berechnung parallelisieren?
            </h2>

            <div
              className="mt-5 rounded-2xl bg-[var(--sunopt-panel-hover)] p-5 font-mono text-sm leading-7 text-[var(--sunopt-ink)]"
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
              className="mt-5 text-sm leading-6 text-[var(--sunopt-muted)]"
            >
              Für bestimmte mathematische Zustandsabbildungen kann
              diese Transformation sehr wirksam sein. Unsere A9–A11-
              Experimente haben aber nicht gezeigt, dass ein generischer
              Compiler sie aus beliebigem TypeScript zuverlässig erkennen
              und schneller als Original A ausführen kann. Das ist eine
              mögliche Forschungshypothese, kein erreichtes Ergebnis.
            </p>
          </article>

          <article
            className="rounded-3xl border border-[var(--sunopt-border)] bg-[var(--sunopt-card)] p-7 shadow-sm"
          >
            <div
              className="text-xs font-bold uppercase tracking-[0.18em] text-[var(--sunopt-soft-text)]"
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
                    className="flex gap-3 text-sm leading-6 text-[var(--sunopt-muted)]"
                  >
                    <span
                      className="mt-2 h-2 w-2 shrink-0 rounded-sm bg-[var(--sunopt-soft-text)]"
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
          className="mt-10 rounded-3xl border border-[var(--sunopt-border)] bg-[var(--sunopt-panel-hover)] p-7"
        >
          <div
            className="text-xs font-bold uppercase tracking-[0.18em] text-[var(--sunopt-soft-text)]"
          >
            Entscheidung nach B14Q
          </div>

          <h2
            className="mt-3 text-2xl font-black"
          >
            B beschleunigt SunOpt; die generische Übertragung auf A blieb erfolglos
          </h2>

          <p
            className="mt-4 max-w-5xl text-sm leading-6 text-[var(--sunopt-muted)]"
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
