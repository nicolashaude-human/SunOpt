// src/app/documentation/code-path/page.tsx

const stages = [
  {
    title: "1 · Browser / Tankvergleich",
    file: "src/app/tank-comparison/page.tsx",
    text:
      "Die UI sammelt Wärmepumpenleistung, PV-Leistung, Tarife und den Solver-Pfad A oder B. Der Request geht an /api/tank-comparison.",
  },
  {
    title: "2 · API-Orchestrierung",
    file: "src/app/api/tank-comparison/route.ts",
    text:
      "Die Route lädt die Wetterreihe, erzeugt den Warmwasserbedarf, resampelt auf 15 Minuten und startet genau einen GPU-Lockstep-Jahreslauf über alle 28 Speichergrössen.",
  },
  {
    title: "3 · Jahres-Lockstep",
    file: "src/simulation/tank-comparison-gpu-lockstep.ts",
    text:
      "Alle Speicher teilen sich Wetter- und Lastreihen. Pro 15-Minuten-Schritt werden die dynamischen Zustände vorbereitet, der kausale Solver ausgeführt und danach der physische Speicherzustand fortgeschrieben.",
  },
  {
    title: "4A · Approach A",
    file: "konsolidierter A1–A7-Pfad",
    text:
      "A ist der exakte FP64-Referenz- und Produktionspfad. Hinter dem Button A gehört der konsolidierte Stand der validierten Approach-A-Optimierungen, nicht der alte nackte Direct-FP64-Pfad.",
  },
  {
    title: "4B · Approach B",
    file: "B14P FP32 + FP64-Fallback",
    text:
      "B nutzt den problem-spezifisch komprimierten B14P-FP32-Pfad. Nicht unterstützte oder fehlerhafte B-Fälle fallen kontrolliert auf FP64 zurück.",
  },
  {
    title: "5 · N-API / Rust",
    file: "rust/sunopt-node/src/cuda_context.rs",
    text:
      "Rust hält die persistenten nativen Kontexte, statischen Tankparameter und dynamischen Schrittwerte. Hier muss der tatsächlich aktive Pfad eindeutig diagnostizierbar sein.",
  },
  {
    title: "6 · CUDA",
    file: "rust/sunopt-node/cuda/cuda_context.cu",
    text:
      "CUDA verarbeitet die unabhängigen Tank-Jobs eines Zeitschritts parallel. Der nächste Jahreszeitschritt kann erst nach dem Host-Feedback des aktuellen Schritts vorbereitet werden.",
  },
] as const;

const feedback = [
  "Speicherenergie zu Beginn des Schritts",
  "Forecast / Mindestreserve / angeforderte HP-Energie",
  "GPU bestimmt den kausalen Cut / Waterline",
  "Wärmepumpe und Speicher werden physisch fortgeschrieben",
  "neue Speicherenergie wird Zustand des nächsten 15-Minuten-Schritts",
] as const;

export default function CodePathDocumentationPage() {

  return (
    <main className="mx-auto max-w-7xl px-4 py-10">
      <header className="mb-8">
        <p className="text-sm font-bold uppercase tracking-[0.18em] text-[#9a7410]">
          SunOpt · Dokumentation
        </p>
        <h1 className="mt-2 text-4xl font-black text-gray-950">
          Codepfad der Jahressimulation
        </h1>
        <p className="mt-4 max-w-4xl leading-7 text-[#675640]">
          Diese Seite beschreibt den produktiven Weg vom Browser bis zum CUDA-Kernel.
          Entscheidend ist die Trennung zwischen dem konsolidierten exakten Approach A
          und dem komprimierten Approach B.
        </p>
      </header>

      <section className="grid gap-4 lg:grid-cols-2">
        {stages.map((stage) => (
          <article
            key={stage.title}
            className="rounded-2xl border border-[#d8c5a5] bg-[#fffaf2] p-5 shadow-sm"
          >
            <h2 className="text-xl font-extrabold text-gray-950">
              {stage.title}
            </h2>
            <code className="mt-2 block rounded-lg bg-[#f3e6d2] px-3 py-2 text-sm text-[#5d4c36]">
              {stage.file}
            </code>
            <p className="mt-3 leading-7 text-[#675640]">
              {stage.text}
            </p>
          </article>
        ))}
      </section>

      <section className="mt-8 rounded-2xl border border-[#d8c5a5] bg-white p-6 shadow-sm">
        <h2 className="text-2xl font-black text-gray-950">
          Die kausale Rückkopplung
        </h2>
        <div className="mt-5 grid gap-3 md:grid-cols-5">
          {feedback.map((item, index) => (
            <div
              key={item}
              className="rounded-xl border border-[#e2d2b7] bg-[#fffaf2] p-4"
            >
              <div className="mb-2 text-sm font-black text-[#9a7410]">
                {index + 1}
              </div>
              <p className="text-sm leading-6 text-[#675640]">
                {item}
              </p>
            </div>
          ))}
        </div>
        <p className="mt-5 leading-7 text-[#675640]">
          Deshalb ist ein kompletter Jahreslauf pro Tank zeitlich sequenziell:
          der Zustand des nächsten Schritts existiert erst nach dem physischen
          Ergebnis des aktuellen Schritts. Parallelisiert werden die unabhängigen
          Tanks beziehungsweise Jobs innerhalb desselben Zeitschritts.
        </p>
      </section>

      <section className="mt-8 grid gap-4 md:grid-cols-2">
        <article className="rounded-2xl border-2 border-[#365b8c] bg-[#eef4fb] p-6">
          <h2 className="text-2xl font-black text-[#244268]">
            Button A
          </h2>
          <p className="mt-3 leading-7 text-[#3e5570]">
            Konsolidierter Approach A1–A7, exakte FP64-Physik. A darf nicht auf
            den alten unoptimierten FP64-Zweig zeigen.
          </p>
        </article>

        <article className="rounded-2xl border-2 border-[#7b5a00] bg-[#fff3c7] p-6">
          <h2 className="text-2xl font-black text-[#624800]">
            Button B
          </h2>
          <p className="mt-3 leading-7 text-[#66501e]">
            B14P FP32 als Produktionssolver, mit kontrolliertem exaktem
            FP64-Fallback für nicht unterstützte Fälle.
          </p>
        </article>
      </section>
    </main>
  );
}
