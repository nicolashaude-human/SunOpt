import {
  SunOptDocHeader,
  SunOptFlowStep,
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

const cudaStages = [
  {
    title: "Parameter und Jahresdaten",
    subtitle: "Next.js · Ausgangslage festlegen",
    file: "src/app/tank-comparison/page.tsx",
    description: "Man wählt die PV-Anlage, die Wärmepumpe und die Kostenannahmen. Die Anwendung verwendet dazu historische Wetter- und Verbrauchsreihen. Für jeden der 35'040 Viertelstundenschritte stehen so die benötigten zeitlichen Eingaben bereit.",
  },
  {
    title: "API und Datenaufbereitung",
    subtitle: "Eingaben prüfen und vorbereiten",
    file: "src/app/api/tank-comparison/route.ts",
    description: "Eine serverseitige Schnittstelle übernimmt die Anfrage. Sie bereitet den Jahreslauf für die unterschiedlichen Speichergrössen vor, bevor die zeitaufwendige Simulation startet. Diese Vorbereitungsphase ist vom eigentlichen GPU-Rechenkern getrennt.",
  },
  {
    title: "GPU-Jahrescontroller",
    subtitle: "28 Speicher, ein gemeinsamer Zeitschritt",
    file: "src/simulation/tank-comparison-gpu-lockstep.ts",
    description: "Der Controller hält die 28 unabhängigen Tanks zeitlich zusammen. Er berechnet für alle Tanks zunächst denselben Zeitpunkt. Erst wenn die Speicherzustände feststehen, geht er zur nächsten Viertelstunde weiter. Eine einzelne Jahreskurve bleibt dadurch zeitlich abhängig.",
  },
  {
    title: "GPU-Speicher und Datentransport",
    subtitle: "Konstante Vektoren bleiben auf der Grafikkarte",
    file: "rust/sunopt-node/ · CUDA-Runtime",
    description: "Wiederverwendete Jahresdaten und vorbereitete Arbeitsbereiche werden möglichst auf der GPU gehalten. Pro Zeitschritt müssen nur die veränderlichen Zustandswerte ausgetauscht werden. Das spart Übertragungen zwischen Hauptspeicher und Grafikkarte.",
  },
  {
    title: "CUDA-Solver A oder B",
    subtitle: "Zwei produktive Berechnungswege",
    file: "rust/sunopt-node/ · native CUDA-Funktionen",
    description: "Der unveränderte Schalter im Tankvergleich wählt den präziseren FP64-Referenzpfad A oder den spezialisierten FP32-Pfad B. Beide ermitteln unter den Modellbedingungen die nötige Heizleistung beziehungsweise den zulässigen Cut. Der experimentelle generische A9–A11-Compiler ist kein dritter produktiver Button.",
  },
  {
    title: "Neuer Speicherzustand",
    subtitle: "Energiefluss je Zeitschritt bilanzieren",
    file: "GPU-Lockstep · Zustandsaktualisierung",
    description: "Die simulierte Wärmepumpe lädt den Speicher, Warmwasserverbrauch und Wärmeverluste entnehmen Energie. Daraus entsteht die Temperatur beziehungsweise der Energiezustand für den nächsten Schritt. Genau deshalb kann man nicht einfach sämtliche Zeitpunkte einer Tankkurve unabhängig berechnen.",
  },
  {
    title: "Jahresbilanz und Diagramme",
    subtitle: "28 Tankgrössen miteinander vergleichen",
    file: "Next.js · Tankvergleich",
    description: "Am Ende werden Netzbezug, Eigenverbrauch, Einspeisung, Speicherverluste und Energiekosten über das gesamte Jahr ausgewertet. Die Ergebnisse gehen über die API zurück an die Oberfläche, wo die Speichergrössen verglichen werden.",
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
        <section className="sunopt-panel mt-8 px-4 py-6 sm:px-7" aria-label="GPU-Jahressimulation">
          <h2 className="mb-4 text-2xl font-black text-[var(--sunopt-ink)]">Vom Jahresdatensatz zum CUDA-Ergebnis</h2>
          <p className="mb-6 max-w-5xl leading-7 text-[var(--sunopt-muted)]">
            Auch hier zeigen die grünen Kästen links den technischen Schritt und die Erklärungen rechts dessen Zweck. Anders als beim CPU-JIT werden im CUDA-Jahreslauf bereits vorhandene native GPU-Funktionen verwendet; er übersetzt den TypeScript-Code nicht bei jedem Aufruf neu.
          </p>
          {cudaStages.map((stage, index) => (
            <div key={stage.title}>
              <SunOptFlowStep number={index + 1} title={stage.title} subtitle={stage.subtitle}
                file={stage.file} description={<p>{stage.description}</p>} />
              {index < cudaStages.length - 1 && <div className="sunopt-flow-arrow" aria-hidden="true">↓</div>}
            </div>
          ))}
          <p className="mt-6 text-sm leading-7 text-[var(--sunopt-muted)]">
            Die Buttons A und B im Tankvergleich bleiben unverändert. A ist die FP64-Referenz und B der spezialisierte Produktionspfad. Die dokumentierten Beschleunigungsexperimente sind davon getrennt.
          </p>
        </section>
      </div>
    </main>
  );
}
