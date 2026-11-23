import {
  SunOptDocHeader,
  SunOptFlowStep,
  SunOptPanel,
} from "@/components/ui/sunopt-documentation";

// The explanations intentionally describe responsibilities rather than asserting
// that every simulation call goes through the CPU JIT. The production CUDA path
// below is separate and the A/B routing remains unchanged.
const jitStages = [
  {
    title: "Historische Vektoren",
    subtitle: "Next.js · Daten über die Zeit",
    file: "Wetter-, PV- und Verbrauchszeitreihen",
    description: "Bevor SunOpt rechnen kann, braucht es eine zeitlich geordnete Liste von Werten. Für jedes Zeitintervall steht darin zum Beispiel, wie viel Solarstrom verfügbar war und wie viel Energie verbraucht wurde. Diese Listen nennt man Vektoren. So kann die Simulation einen ganzen Tag oder ein ganzes Jahr Schritt für Schritt nachvollziehen.",
  },
  {
    title: "Funktions-Wrapper",
    subtitle: "Die passende Berechnung auswählen",
    file: "src/simulation/rust-acceleration-scopes.ts",
    description: "Nicht das gesamte Programm muss beschleunigt werden. Ein Wrapper bildet eine klar definierte Schnittstelle zu einer bestimmten Berechnung: Er nimmt die benötigten Werte entgegen und gibt das Ergebnis in einer einheitlichen Form zurück. Der übrige Anwendungscode kann dieselbe Aufgabe weiterhin verstehen, ohne die technischen Details der Ausführung zu kennen.",
  },
  {
    title: "Worker Manager",
    subtitle: "Rechenaufträge organisieren",
    file: "src/workers/",
    description: "Der Worker Manager verteilt geeignete Aufgaben an Rechenprozesse und sammelt ihre Ergebnisse. Damit kann die Benutzeroberfläche von aufwendigen Berechnungen getrennt bleiben. Ob Aufgaben tatsächlich gleichzeitig laufen können, hängt davon ab, ob ihre Eingaben voneinander unabhängig sind.",
  },
  {
    title: "Native Schnittstelle",
    subtitle: "JavaScript spricht mit Rust",
    file: "rust/sunopt-node/ · N-API",
    description: "Ein Webprogramm verwendet normalerweise JavaScript. Für rechenintensive Teile kann es eine native Bibliothek aufrufen. Die N-API-Schnittstelle übermittelt die nötigen Argumente an die Rust-Laufzeit und stellt das Ergebnis der Anwendung wieder zur Verfügung. Der Aufruf dieser Schnittstelle ist nicht selbst schon eine Beschleunigung.",
  },
  {
    title: "Oxc-Parser und AST",
    subtitle: "Quelltext als Syntaxbaum verstehen",
    file: "Oxc · Abstract Syntax Tree",
    description: "Der Parser liest eine unterstützte Funktion und zerlegt ihre Anweisungen in eine strukturierte Baumdarstellung. Dort steht beispielsweise, welche Werte addiert werden oder unter welcher Bedingung eine Schleife endet. Dieser Syntaxbaum ist noch kein ausführbarer Maschinencode: Er ist die Grundlage für den nächsten Übersetzungsschritt.",
  },
  {
    title: "Cranelift – JIT",
    subtitle: "Während der Laufzeit übersetzen",
    file: "Rust-Laufzeit · Cranelift IR",
    description: "Aus dem Syntaxbaum wird zunächst eine Zwischenrepräsentation erzeugt, die Cranelift verarbeiten kann. Cranelift übersetzt sie zur Laufzeit in Maschinenanweisungen für den Prozessor. JIT bedeutet „Just in Time“: Die Übersetzung findet erst statt, wenn das Programm die Funktion benötigt. Kompilierte Varianten können danach wiederverwendet werden.",
  },
  {
    title: "Ausführung und Ergebnis",
    subtitle: "Maschinencode auf der CPU ausführen",
    file: "JIT-Cache · Rückgabe an Next.js",
    description: "Die Maschine führt die übersetzte Berechnung mit den aktuellen Eingabedaten aus. Das Resultat gelangt über die native Schnittstelle zurück zur Anwendung und kann dort für weitere Simulationsschritte oder Diagramme verwendet werden. Eine bereits kompilierte Funktion muss bei gleichen Voraussetzungen nicht jedes Mal neu übersetzt werden.",
  },
] as const;

const lockstepStages = [
  {
    title: "Jahresdaten und API",
    text: "Die Benutzeroberfläche legt Parameter wie Wärmepumpenleistung, PV-Anlage und Speichervolumen fest. Die API bereitet Wetter- und Verbrauchsdaten für die 15-Minuten-Schritte vor.",
    file: "src/app/api/tank-comparison/route.ts",
  },
  {
    title: "GPU-Lockstep und Zustand",
    text: "Der Jahrescontroller berechnet die 28 Tankgrössen in jedem Zeitschritt parallel, benötigt aber für den nächsten Schritt jeweils die neuen Speicherzustände aus dem vorherigen Schritt.",
    file: "src/simulation/tank-comparison-gpu-lockstep.ts",
  },
  {
    title: "Nativer CUDA-Solver",
    text: "Die produktiven Buttons A und B verwenden ihre vorhandenen CUDA-Ausführungspfade. Der experimentelle, aus TypeScript generierte A9–A11-Pfad gehört ausdrücklich nicht zu diesen Buttons.",
    file: "rust/sunopt-node/",
  },
] as const;

export default function CodePathDocumentationPage() {
  return (
    <main className="sunopt-doc-page px-4 py-10 sm:px-6">
      <div className="mx-auto max-w-7xl">
        <SunOptDocHeader label="SunOpt · Dokumentation" title="Wie wird Programmcode schneller ausgeführt?">
          <p>Die linken grünen Boxen zeigen den Weg der Informationen durch das Programm. Rechts steht zu jedem Schritt, was tatsächlich passiert – ohne vorauszusetzen, dass man sich mit Softwareentwicklung auskennt.</p>
          <p className="mt-2">Zuerst zeigen wir den allgemeinen CPU-JIT-Codepfad über Oxc und Cranelift. Die produktive GPU-Jahressimulation ist ein eigenständiger Ablauf und wird darunter erklärt.</p>
        </SunOptDocHeader>
        <section className="sunopt-panel px-4 py-6 sm:px-7" aria-label="CPU-JIT-Codepfad">
          <h2 className="mb-6 text-2xl font-black text-[var(--sunopt-ink)]">Vom Messwert zum Maschinencode</h2>
          {jitStages.map((stage, index) => (
            <div key={stage.title}>
              <SunOptFlowStep number={index + 1} title={stage.title} subtitle={stage.subtitle}
                file={stage.file} description={<p>{stage.description}</p>} />
              {index < jitStages.length - 1 && <div className="sunopt-flow-arrow" aria-hidden="true">↓</div>}
            </div>
          ))}
          <p className="mt-6 text-sm leading-7 text-[var(--sunopt-muted)]">Hinweis: Dies ist eine verständliche Architekturübersicht. Je nach Berechnung können Vorbereitung, Worker-Aufruf und Cache-Nutzung variieren; nicht jede Rechenaufgabe durchläuft alle Schritte.</p>
        </section>
        <section className="mt-8" aria-label="GPU-Jahressimulation">
          <h2 className="mb-4 text-2xl font-black text-[var(--sunopt-ink)]">Der separate CUDA-Jahreslauf</h2>
          <p className="mb-5 max-w-5xl leading-7 text-[var(--sunopt-muted)]">Für den Tankvergleich läuft ein speziell optimierter GPU-Controller. Dessen Datenfluss darf nicht mit dem allgemeinen CPU-JIT verwechselt werden.</p>
          <div className="grid gap-4 lg:grid-cols-3">
            {lockstepStages.map((stage) => (
              <SunOptPanel key={stage.title} title={stage.title}>
                <p>{stage.text}</p>
                <code className="mt-3 block break-all rounded-lg bg-[var(--sunopt-panel-hover)] px-3 py-2 text-xs">{stage.file}</code>
              </SunOptPanel>
            ))}
          </div>
        </section>
        <section className="mt-8 grid gap-4 md:grid-cols-2">
          <SunOptPanel title="Button A – unverändert">
            <p>Der bestehende exakte FP64-CUDA-Pfad bleibt die Leistungs- und Genauigkeitsreferenz. Diese UI-Überarbeitung verändert seine Implementierung oder Auswahl nicht.</p>
          </SunOptPanel>
          <SunOptPanel title="Button B – unverändert">
            <p>Der bestehende spezialisierte B14P-Pfad verwendet FP32 dort, wo die erforderliche Ergebnisgenauigkeit nachgewiesen wurde. Auch seine produktive Ausführung bleibt unverändert.</p>
          </SunOptPanel>
        </section>
      </div>
    </main>
  );
}
