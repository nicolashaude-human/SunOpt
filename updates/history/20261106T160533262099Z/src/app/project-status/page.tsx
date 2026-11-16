// src/app/project-status/page.tsx

"use client";

import {
  useMemo,
  useState,
} from "react";


interface StatusItem {
  title: string;
  description: string;
  done: boolean;
}


interface StatusSection {
  title: string;
  description: string;
  items: StatusItem[];
}


const initialSections: StatusSection[] = [
  {
    title: "Grundlagen und Daten",

    description:
      "Basisdaten und physikalische Modelle der Warmwasser-Simulation.",

    items: [
      {
        title: "PVGIS Wetter- und PV-Daten integriert",

        description:
          "Stündliche Wetter- und PV-Ertragsdaten werden geladen und auf gemeinsame 15-Minuten-Schritte aufgelöst.",

        done: true,
      },

      {
        title: "Warmwasserverbrauch modelliert",

        description:
          "Der Warmwasserbedarf wird aus Personenanzahl, Tagesverbrauch, Kaltwassertemperatur, Solltemperatur und Tagesprofil berechnet.",

        done: true,
      },

      {
        title: "Wärmepumpenmodell integriert",

        description:
          "Thermische Leistung, elektrische Leistung, COP und maximale Vorlauftemperatur werden in der Simulation berücksichtigt.",

        done: true,
      },

      {
        title: "Thermischer Speicher modelliert",

        description:
          "Speicherinhalt, Temperatur, nutzbare Kapazität und thermische Verluste werden physikalisch berechnet.",

        done: true,
      },

      {
        title: "Speicherisolierung berücksichtigt",

        description:
          "Wärmeverluste werden aus Speichergeometrie, Dämmstärke, Wärmeleitfähigkeit und Umgebungstemperatur bestimmt.",

        done: true,
      },
    ],
  },


  {
    title: "Simulation und gemeinsame Physik",

    description:
      "Alle Regelstrategien laufen über dieselbe physikalische 15-Minuten-Simulation.",

    items: [
      {
        title: "15-Minuten-Simulation",

        description:
          "PV-Erzeugung, Warmwasserbedarf, Wärmepumpe, Speicher, Speicherverluste und Netzbezug werden in 15-Minuten-Schritten simuliert.",

        done: true,
      },

      {
        title: "PV-Überschuss nach Haushaltsverbrauch",

        description:
          "Für die Wärmepumpe steht nur PV-Leistung oberhalb der konfigurierten Haushaltsreserve zur Verfügung.",

        done: true,
      },

      {
        title: "Netzbezug separat bilanziert",

        description:
          "Elektrischer Netzbezug der Wärmepumpe wird für jeden Zeitschritt und über das gesamte Jahr bilanziert.",

        done: true,
      },

      {
        title: "Versorgungssicherheit separat bilanziert",

        description:
          "Nicht gedeckter thermischer Warmwasserbedarf wird unabhängig vom Netzbezug erfasst.",

        done: true,
      },

      {
        title:
          "Speicherphysik für alle Controller vereinheitlicht",

        description:
          "Existing Controller, Quantum Gravity sowie kWh_user und kWh_noSunshine verwenden dieselbe kanonische Speicherberechnung. Die Regelstrategien entscheiden nur über die Wärmepumpenanforderung und besitzen keine eigene abweichende Speicherphysik.",

        done: true,
      },

      {
        title: "Speicherverlust-Bug korrigiert",

        description:
          "Eine doppelte Anwendung der Speicherverluste pro Zeitschritt wurde entfernt. Zusätzlich wurde die Zylindergeometrie korrigiert, sodass Deckel und Boden nicht doppelt gezählt werden. Damit verwenden alle Modelle dieselbe korrigierte Verlustrechnung.",

        done: true,
      },
    ],
  },


  {
    title: "Zentrale Konfiguration",

    description:
      "Technische Konstanten und Versuchsparameter wurden aus den einzelnen Modellen in eine gemeinsame Konfiguration zusammengeführt.",

    items: [
      {
        title: "Gemeinsame CONFIG-Struktur",

        description:
          "Simulations-, PV-, Warmwasser-, Wärmepumpen-, Speicher- und Forecast-Parameter werden zentral in config.ts verwaltet.",

        done: true,
      },

      {
        title: "Zeitschritt zentral konfiguriert",

        description:
          "Die Simulation verwendet zentral einen Zeitschritt von 0.25 h beziehungsweise 15 Minuten.",

        done: true,
      },

      {
        title: "PV-Testanlage zentral konfiguriert",

        description:
          "Aktueller Referenzfall: 10 kWp, Azimut 0° nach PVGIS-Konvention, 30° Neigung und 14 % Systemverluste.",

        done: true,
      },

      {
        title:
          "Warmwasserparameter zentral konfiguriert",

        description:
          "Aktueller Referenzfall: 2 Personen, 50 Liter pro Person und Tag, 10 °C Kaltwasser und 60 °C Warmwassertemperatur.",

        done: true,
      },

      {
        title:
          "Wärmepumpenparameter zentral konfiguriert",

        description:
          "Aktueller Referenzfall: 5 kW thermische Leistung, maximal 75 °C Vorlauf und COP 3.",

        done: true,
      },

      {
        title:
          "Speicherparameter zentral konfiguriert",

        description:
          "Aktueller Referenzfall: 3000 Liter, 60–75 °C, PUR-Dämmung mit λ = 0.030 W/(m·K), 100 mm Dämmstärke, 1.70 m Zylinderhöhe und 20 °C Umgebung.",

        done: true,
      },

      {
        title:
          "Forecast-Parameter zentral konfiguriert",

        description:
          "Haushaltsreserve, Forecast-Horizonte, Sicherheitsfaktor und Zieltemperatur werden über gemeinsame Konfigurationswerte geführt. Die aktuelle Haushaltsreserve beträgt 2 kW.",

        done: true,
      },
    ],
  },


  {
    title: "Regelstrategien und Experimente",

    description:
      "Mehrere Regelungsansätze werden mit derselben physikalischen Simulation verglichen.",

    items: [
      {
        title: "Existing Controller als Referenz",

        description:
          "Der bestehende Forecast-Regler dient als quantitative Baseline.",

        done: true,
      },

      {
        title: "Rolling Demand Forecast",

        description:
          "Zukünftiger Warmwasserbedarf und Speicherverluste werden über einen gleitenden Zeithorizont berechnet.",

        done: true,
      },

      {
        title: "Rolling PV Forecast",

        description:
          "Verfügbare PV-Energie oberhalb der Haushaltsreserve wird separat prognostiziert.",

        done: true,
      },

      {
        title: "Quantum Gravity v1",

        description:
          "Eine Vorwärtssuche bis zu einem zukünftigen PV-Recovery-Punkt wurde implementiert und getestet.",

        done: true,
      },

      {
        title: "Zeropoint Generator",

        description:
          "Ein rückwärts gerechneter Minimalenergie-Ansatz wurde als experimentelles Modell untersucht.",

        done: true,
      },

      {
        title: "Gemeinsame Controller-Schnittstelle",

        description:
          "Die verschiedenen Regelstrategien greifen über eine gemeinsame Schnittstelle auf dieselbe Wärmepumpen- und Speicherphysik zu.",

        done: true,
      },
    ],
  },


  {
    title: "kWh_user – 48-h Forecast",

    description:
      "Kurzfristiger Forecast für den Energiebedarf der nächsten 48 Stunden. (Plasma Wave Accelerator)",

    items: [
      {
        title: "kWh_user Forecast",

        description:
          "Der zukünftige Warmwasserbedarf wird in 15-Minuten-Paketen über einen rollierenden 48-Stunden-Horizont prognostiziert.",

        done: true,
      },

      {
        title: "48-h Waterline implementiert",

        description:
          "Die verfügbare PV-Leistung oberhalb der Haushaltsreserve wird als Leistungsprofil betrachtet. Eine horizontale Waterline bestimmt, welche PV-Leistung zum Laden des Speichers verwendet wird.",

        done: true,
      },

      {
        title:
          "Variable Wärmepumpenleistung oberhalb der Waterline",

        description:
          "Die geplante elektrische Wärmepumpenleistung entspricht der PV-Leistung oberhalb der Waterline, begrenzt durch die maximale elektrische Wärmepumpenleistung.",

        done: true,
      },

      {
        title:
          "Waterline alle 15 Minuten neu berechnet",

        description:
          "Alle 15 Minuten wird die Waterline aus dem aktuellen Speicherzustand sowie dem verschobenen 48-Stunden-Bedarfs- und PV-Fenster neu bestimmt.",

        done: true,
      },

      {
        title:
          "Kausalitätsfehler im Scheduler korrigiert",

        description:
          "Der ursprüngliche Scheduler erfüllte nur die gesamte 48-h-Energiebilanz. Dadurch konnte spätere PV-Energie rechnerisch früheren Bedarf decken. Der korrigierte Scheduler prüft die Speicherbilanz chronologisch für jedes 15-Minuten-Segment.",

        done: true,
      },

      {
        title: "Chronologische Survival-Prüfung",

        description:
          "Eine Waterline ist nur zulässig, wenn der simulierte Speicher alle zukünftigen Bedarfs- und Verlustsegmente bis zum jeweiligen Zeitpunkt übersteht. Spätere PV-Erzeugung kann damit keinen früheren Bedarf mehr kompensieren.",

        done: true,
      },

      {
        title: "Waterline mathematisch validiert",

        description:
          "Die Wärmepumpenleistung folgt P_HP = min(P_HP,max, max(0, P_export − Waterline)). Diagnoseläufe bestätigen die erwartete Differenz zwischen PV-Export und Waterline.",

        done: true,
      },

      {
        title: "Mandatory Reserve bleibt unabhängig",

        description:
          "Die bestehende Mindestreserve- und Netzlogik bleibt erhalten. Bei unvermeidbaren Winterdefiziten darf die Wärmepumpe deshalb unabhängig von der PV-Waterline aus dem Netz laden.",

        done: true,
      },
    ],
  },


  {
    title: "kWh_noSunshine – 7-Tage-Forecast",

    description:
      "Langfristiger Forecast zur Vorbereitung auf mehrtägige sonnenarme Perioden. (Plasma Wave Accelerator)",

    items: [
      {
        title: "7-Tage-PV-Horizont",

        description:
          "Für kWh_noSunshine wird die PV-Verfügbarkeit über insgesamt sieben Tage beziehungsweise 168 Stunden betrachtet.",

        done: true,
      },

      {
        title: "Keine Doppelzählung mit kWh_user",

        description:
          "kWh_noSunshine erzeugt zusätzliche Bedarfs-Buckets nur für Stunden 48–168. Die ersten 48 Stunden werden bereits durch kWh_user abgedeckt.",

        done: true,
      },

      {
        title: "Backward-Bucket-Simulation",

        description:
          "Die Bedarfs- und Verlustpakete der zusätzlichen fünf Tage werden rückwärts auf frühere verfügbare PV-Zeitfenster abgebildet.",

        done: true,
      },

      {
        title:
          "Bestehendes Kühlmodell wiederverwendet",

        description:
          "Die Rückwärtsrechnung verwendet das gemeinsame Speichermodell zur Ermittlung der notwendigen Energie vor einem Zeitschritt. Es wurde keine zweite unabhängige Verlustformel eingeführt.",

        done: true,
      },

      {
        title:
          "7-Tage-Waterline als Diagnose verfügbar",

        description:
          "kWh_noSunshine erzeugt bereits eine eigene Waterline und einen geplanten Wärmepumpenverlauf. Damit kann geprüft werden, wann eine zukünftige Schlechtwetterperiode eine frühere Speicherladung erzwingt.",

        done: true,
      },

      {
        title:
          "Schlechtwetterfall August identifiziert",

        description:
          "Der Test 24.–30. August ergibt rund 54.23 kWh_th kWh_noSunshine für die zusätzlichen fünf Tage. Die nutzbare Speicherenergie von 60 auf 75 °C beträgt beim 3000-Liter-Speicher 52.325 kWh_th.",

        done: true,
      },

      {
        title:
          "kWh_user und kWh_noSunshine kombiniert",

        description:
          "Die 48-h- und 7-Tage-Waterlines sind gekoppelt. Der kombinierte Scheduler verwendet jeweils die tiefere Waterline und bestimmt damit die tatsächlich anzuwendende Wärmepumpenanforderung.",

        done: true,
      },

      {
        title:
          "Kombinierten Regler über ein Jahr validiert",

        description:
          "Die kombinierte Regelstrategie läuft über die vollständige Jahressimulation. Im aktuellen Referenzlauf liegt der Netzbezug rund 5 kWh_el unter dem Existing Controller. Das frühere morgendliche Nachheizen wurde als Abkühlungs- und Reserveproblem identifiziert.",

        done: true,
      },
    ],
  },


  {
    title: "Ergebnisse",

    description:
      "Bisherige Jahresergebnisse und daraus abgeleitete technische Erkenntnisse.",

    items: [
      {
        title: "Existing Controller gemessen",

        description:
          "577.69 kWh_el/a Netzbezug, 1008.30 kWh_el/a PV-Nutzung durch die Wärmepumpe und 1318.18 kWh_th/a Speicherverluste im damaligen Vergleichslauf.",

        done: true,
      },

      {
        title: "Quantum Gravity v1 gemessen",

        description:
          "587.48 kWh_el/a Netzbezug, 245.42 kWh_el/a PV-Nutzung und 235.06 kWh_th/a Speicherverluste im damaligen Vergleichslauf.",

        done: true,
      },

      {
        title:
          "Versorgung bei beiden Vergleichsmodellen vollständig",

        description:
          "Der nicht gedeckte thermische Bedarf betrug in den damaligen Vergleichsläufen 0.00 kWh_th/a.",

        done: true,
      },

      {
        title: "Zielkonflikt identifiziert",

        description:
          "Der Existing Controller nutzt viel PV, erzeugt aber hohe Speicherverluste. Quantum Gravity reduziert die Verluste stark, nutzt jedoch zu wenig verfügbare PV-Energie.",

        done: true,
      },

      {
        title:
          "Jahreskennzahlen nach Physik-Fix neu bestimmen",

        description:
          "Da die Speicherverlustrechnung inzwischen korrigiert und vereinheitlicht wurde, müssen ältere Jahreskennzahlen vor einer finalen Bewertung nochmals mit dem aktuellen Modellstand erzeugt werden.",

        done: false,
      },
    ],
  },


  {
    title: "Frontend und Visualisierung",

    description:
      "Weboberflächen zur Analyse, Modellbewertung und Projektdokumentation.",

    items: [
      {
        title:
          "Debug-Oberfläche für Warmwasser-Simulation",

        description:
          "Monatsweise Analyse von Speicherenergie, Speichertemperatur, Wärmepumpe, Netzbezug, PV-Nutzung und Forecast-Werten.",

        done: true,
      },

      {
        title: "Synchronisierte Zeitreihenplots",

        description:
          "Mehrere Diagramme verwenden synchronisierte horizontale Scrollbars. Eine Bildschirmbreite entspricht 24 Stunden.",

        done: true,
      },

      {
        title: "48-h-Waterline visualisiert",

        description:
          "Die kWh_user-Waterline und die darauf geplante elektrische Wärmepumpenleistung können im Debug-Plot mit dem verfügbaren PV-Export verglichen werden.",

        done: true,
      },

      {
        title: "7-Tage- und kombinierte Waterline visualisiert",

        description:
          "kWh_user, kWh_noSunshine und die effektive kombinierte Waterline werden im Debug-Plot gemeinsam mit PV-Export und Wärmepumpenleistung dargestellt.",

        done: true,
      },

      {
        title: "Model-Comparison-Seite überarbeiten",

        description:
          "Die bestehende Model-Comparison-Seite stürzt mit dem aktuellen Modellstand ab. Sie wird nach der Rust-Performance-Optimierung an die kombinierte 48-h-/7-Tage-Regelung und die aktuellen Jahresergebnisse angepasst.",

        done: false,
      },

      {
        title: "Projektstatus für Management",

        description:
          "Technischer Entwicklungsstand, abgeschlossene Arbeitspakete und nächste Schritte werden auf einer eigenen Statusseite zusammengeführt.",

        done: true,
      },

      {
        title: "Interaktive Status-Häkchen",

        description:
          "Arbeitspakete können auf der Statusseite vor einer Besprechung direkt als erledigt oder offen markiert werden.",

        done: true,
      },

      {
        title:
          "Markdown-Export für externen Austausch",

        description:
          "Der aktuelle Projektstatus kann als eigenständige Markdown-Datei exportiert und ohne Zugriff auf den lokalen Entwicklungsserver weitergegeben werden.",

        done: true,
      },
    ],
  },


  // SUNOPT_ACCELERATION_AB_STATUS_BEGIN
  {
    title: "Beschleunigungsstrategie A – generische Beschleunigung",

    description:
      "Ansatz A beschleunigt den bestehenden TypeScript-/Plasma-Wave-Algorithmus ohne Änderung der Physik oder Regelungssemantik. Dieser Projektabschnitt gilt nach CPU-, CUDA- und Runtime-Optimierung als erfolgreich abgeschlossen.",

    items: [
      {
        title: "TypeScript bis Maschinencode beschleunigt",

        description:
          "Auf dem Intel Core i5-8400 erkennt Oxc geeignete TypeScript-Scopes; Rust und Cranelift führen denselben Algorithmus nativ aus. Die frühe Jahressimulation wurde damit von ungefähr 198 s auf ungefähr 27 s reduziert. Im 48-h-Scheduler sank die gemessene Zeit von 17.10 s auf 2.70 s; die causalSearch selbst von 16.31 s auf 2.01 s.",

        done: true,
      },

      {
        title: "CUDA für parallele Tankberechnung integriert",

        description:
          "Statische Jahresdaten und Tankparameter werden GPU-resident verarbeitet. Der frühere 28-Tank-Batchlauf von 35.585 s wurde auf einen gemessenen GPU-Lockstep-Gesamtlauf von 7.108 s reduziert.",

        done: true,
      },

      {
        title: "Blackwell Causal Replay als stabile Referenz etabliert",

        description:
          "Der aufgezeichnete Blackwell-Jahresdatensatz umfasst 35'040 Schritte, 28 Tanks und 981'120 Causal-Jobs. Der Replay-Checksum 218451794297e021 dient als numerische Referenz für weitere Beschleunigungsversuche.",

        done: true,
      },

      {
        title: "Synchronisations- und Launch-Hypothese geprüft",

        description:
          "Das normale CUDA Replay benötigte median 5.600 s; CUDA Graph reduzierte dies auf 5.165 s. Der Gewinn von rund 435 ms beziehungsweise 7.8 % zeigt, dass Host-/Launch-Overhead nicht mehr der Hauptflaschenhals ist.",

        done: true,
      },

      {
        title: "Ansatz A als Baseline stabilisiert",

        description:
          "Generische Compiler-, CPU-, CUDA- und Runtime-Beschleunigung bleibt Referenz und Fallback. Der Hauptpfad ist abgeschlossen; als letzter A-Punkt soll der im CUDA-Graph-Replay gemessene Launch-/Synchronisationsgewinn von rund 435 ms noch in den Produktionspfad übertragen werden.",

        done: true,
      },


      {
        title: "CUDA-Graph-/Synchronisationsgewinn in Produktionspfad übernehmen",

        description:
          "Das Blackwell Graph Replay reduzierte die Causal-Replay-Walltime von ungefähr 5.600 s auf 5.165 s und weist damit rund 435 ms beziehungsweise 7.8 % vermeidbaren Launch-/Synchronisationsaufwand nach. Im echten zustandsabhängigen Jahreslauf bestehen diese Synchronisationspunkte noch. Dieser letzte A-Schritt soll noch realisiert werden.",

        done: false,
      },

    ],
  },


  {
    title: "Beschleunigungsstrategie B – problemspezifische Solver",

    description:
      "Ansatz B gibt Allgemeingültigkeit bewusst auf und nutzt die konkrete Mathematik von Warmwasserspeicher, Waterline, Reserve und Forecast, um neue High-Speed-Solver zu entwickeln.",

    items: [
      {
        title: "A/B-Architektur klar getrennt",

        description:
          "A bedeutet gleicher Algorithmus mit schnellerer Ausführung. B darf den Algorithmus durch eine mathematisch äquivalente oder kontrolliert approximative SunOpt-Speziallösung ersetzen.",

        done: true,
      },

      {
        title: "Max-Affine Blockkomposition als exakter Kandidat identifiziert",

        description:
          "Die Speicheranforderung kann für geeignete Bereiche als zusammensetzbare max-affine Transformation formuliert werden. Ziel ist, den 192-Schritt-Horizont über einen parallelen Reduktionsbaum statt als lange serielle Zustandskette auszuwerten.",

        done: true,
      },

      {
        title: "Fourier-Kompression als explorativer Kandidat definiert",

        description:
          "PV-Export und Warmwasserbedarf sollen mit reduzierten Frequenzkomponenten rekonstruiert und gegen Plasma Wave auf Signalfehler, Cut-Level-Fehler und Survival-Mismatches geprüft werden.",

        done: true,
      },

      {
        title: "Problemspezifische Solver gegen Plasma Wave benchmarken",

        description:
          "Max-Affine- und Fourier-Varianten werden mit denselben aufgezeichneten Blackwell-Zuständen gegen die kanonische Plasma-Wave-Referenz geprüft. Entscheidend sind Laufzeit, Cut-Level-Parität und Survival-Parität.",

        done: false,
      },

      {
        title: "Direkten piecewise-affinen Cut-Level-Solver untersuchen",

        description:
          "Langfristiges Ziel ist, die wiederholten Causal-Survival-Simulationen der Binary Search durch eine problemspezifische Hüllkurven- beziehungsweise direkte Cut-Level-Lösung zu ersetzen.",

        done: false,
      },
    ],
  },
  // SUNOPT_ACCELERATION_AB_STATUS_END


  {
    title: "Speicher- und Wirtschaftlichkeitsoptimierung",

    description:
      "Nach der Performance-Optimierung wird die kombinierte Regelstrategie für verschiedene Anlagenkonfigurationen ausgewertet und daraus das energetische sowie wirtschaftliche Optimum bestimmt.",

    items: [
      {
        title: "Speichergrössen automatisiert vergleichen",

        description:
          "Der validierte Regler wird für unterschiedliche Speichervolumen automatisiert ausgewertet. Beim aktuellen 3000-Liter-Referenzspeicher zeigt sich bereits die 5-kW-thermische Wärmepumpenleistung als zunehmender Flaschenhals.",

        done: false,
      },

      {
        title: "Energetisches Optimum bestimmen",

        description:
          "Speichervolumen und weitere relevante Anlagenparameter werden anhand von Netzbezug, PV-Eigenverbrauch, Speicherverlusten und Versorgungssicherheit energetisch optimiert.",

        done: false,
      },

      {
        title: "Wirtschaftliches Optimum bestimmen",

        description:
          "Auf Basis des energetischen Optimums werden Investitionskosten, Stromkosten und Einsparungen ergänzt, um die wirtschaftlich sinnvollste Anlagenkonfiguration zu bestimmen.",

        done: false,
      },
    ],
  },
];


function StatusCheckbox(
  props: {
    done: boolean;
    onClick: () => void;
  },
) {

  return (
    <button
      type="button"
      onClick={props.onClick}
      className={
        props.done
          ? "flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-green-600 bg-green-600 text-sm font-bold text-white"
          : "flex h-7 w-7 shrink-0 items-center justify-center rounded-md border-2 border-gray-300 bg-white"
      }
      aria-label={
        props.done
          ? "Als offen markieren"
          : "Als erledigt markieren"
      }
      title={
        props.done
          ? "Als offen markieren"
          : "Als erledigt markieren"
      }
    >
      {props.done ? "✓" : ""}
    </button>
  );
}


function StatusSectionCard(
  props: {
    section: StatusSection;
    sectionIndex: number;
    onToggle: (
      sectionIndex: number,
      itemIndex: number,
    ) => void;
  },
) {

  const completed =
    props.section.items.filter(
      (item) => item.done,
    ).length;


  return (
    <section className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">

      <div className="mb-5 flex flex-wrap items-start justify-between gap-4">

        <div>

          <h2 className="text-xl font-semibold text-gray-900">
            {props.section.title}
          </h2>

          <p className="mt-1 max-w-3xl text-sm text-gray-600">
            {props.section.description}
          </p>

        </div>


        <div className="rounded-full bg-gray-100 px-4 py-2 text-sm font-medium text-gray-700">
          {completed}
          {" / "}
          {props.section.items.length}
          {" erledigt"}
        </div>

      </div>


      <div className="space-y-3">

        {props.section.items.map(
          (item, itemIndex) => (

            <div
              key={item.title}
              className={
                item.done
                  ? "flex gap-4 rounded-lg border border-green-100 bg-green-50 p-4"
                  : "flex gap-4 rounded-lg border border-gray-200 bg-gray-50 p-4"
              }
            >

              <StatusCheckbox
                done={item.done}
                onClick={() =>
                  props.onToggle(
                    props.sectionIndex,
                    itemIndex,
                  )
                }
              />


              <div>

                <h3 className="font-semibold text-gray-900">
                  {item.title}
                </h3>

                <p className="mt-1 text-sm leading-6 text-gray-600">
                  {item.description}
                </p>

              </div>

            </div>

          ),
        )}

      </div>

    </section>
  );
}


function createMarkdown(
  sections: StatusSection[],
): string {

  const allItems =
    sections.flatMap(
      (section) => section.items,
    );


  const completed =
    allItems.filter(
      (item) => item.done,
    ).length;


  const progress =
    allItems.length > 0
      ? Math.round(
          completed / allItems.length * 100,
        )
      : 0;


  const lines: string[] = [
    "# SunOpt – Projektstatus",
    "",
    "## PV-Warmwasserspeicher Optimierung",
    "",
    "Entwicklungsstand der Simulation und Regelstrategie für die optimale Nutzung von Photovoltaikenergie zur Warmwasserbereitung.",
    "",
    `**Projektfortschritt:** ${progress} %`,
    "",
    `**Arbeitspakete:** ${completed} von ${allItems.length} abgeschlossen`,
    "",
    "## Aktueller technischer Schwerpunkt",
    "",
    "**kWh_user** deckt den kurzfristigen rollierenden 48-h-Horizont ab.",
    "",
    "**kWh_noSunshine** erweitert die Planung um sonnenarme Perioden innerhalb eines 7-Tage-Horizonts.",
    "",
    "Beide Verfahren werden im Code unter dem internen Arbeitsnamen *(Plasma Wave Accelerator)* entwickelt.",
    "",
    "Die 48-h- und 7-Tage-Waterlines sind implementiert und validiert. Die generische Beschleunigungsphase A mit Oxc/Cranelift, CUDA und CUDA Graph gilt als abgeschlossen. Aktueller Schwerpunkt ist Phase B: problemspezifische Max-Affine-, piecewise-affine und Fourier-Solver, die gegen Plasma Wave verifiziert werden.",
    "",
  ];


  for (const section of sections) {

    lines.push(
      `## ${section.title}`,
      "",
      section.description,
      "",
    );


    for (const item of section.items) {

      lines.push(
        `- [${item.done ? "x" : " "}] **${item.title}**`,
        `  ${item.description}`,
        "",
      );
    }
  }


  lines.push(
    "## Aktuelles Entwicklungsziel",
    "",
    "kWh_user stellt die kurzfristige kausale Versorgung innerhalb der nächsten 48 Stunden sicher.",
    "",
    "kWh_noSunshine erkennt zusätzlich, wann vor einer längeren sonnenarmen Periode Energie früher gespeichert werden muss.",
    "",
    "Beide Horizonte sind gekoppelt und die Regelstrategie ist funktional fertig. Ansatz A der generischen Beschleunigung ist mit einer Blackwell-Graph-Referenz von rund 5.165 s abgeschlossen; aktueller Schwerpunkt ist Ansatz B mit problemspezifischer Physik- und Signal-Kompression.",
    "",
    "Die neuen B-Solver werden zuerst gegen Plasma Wave auf numerische Parität beziehungsweise kontrollierte Fehler geprüft. Danach folgen die weitere Speichergrössenoptimierung sowie die Bestimmung des energetischen und wirtschaftlichen Optimums.",
    "",
  );


  return lines.join("\n");
}


export default function ProjectStatusPage() {

  const [
    sections,
    setSections,
  ] = useState<StatusSection[]>(
    initialSections,
  );


  const allItems =
    useMemo(
      () =>
        sections.flatMap(
          (section) => section.items,
        ),
      [sections],
    );


  const completed =
    allItems.filter(
      (item) => item.done,
    ).length;


  const total =
    allItems.length;


  const progress =
    total > 0
      ? Math.round(completed / total * 100)
      : 0;


  function toggleItem(
    sectionIndex: number,
    itemIndex: number,
  ) {

    setSections(
      (currentSections) =>
        currentSections.map(
          (
            section,
            currentSectionIndex,
          ) => {

            if (
              currentSectionIndex !==
              sectionIndex
            ) {
              return section;
            }


            return {
              ...section,

              items:
                section.items.map(
                  (
                    item,
                    currentItemIndex,
                  ) => {

                    if (
                      currentItemIndex !==
                      itemIndex
                    ) {
                      return item;
                    }


                    return {
                      ...item,
                      done: !item.done,
                    };
                  },
                ),
            };
          },
        ),
    );
  }


  function downloadMarkdown() {

    const markdown =
      createMarkdown(sections);


    const blob =
      new Blob(
        [markdown],
        {
          type:
            "text/markdown;charset=utf-8",
        },
      );


    const url =
      URL.createObjectURL(blob);


    const link =
      document.createElement("a");


    link.href = url;

    link.download =
      "SunOpt-Projektstatus.md";


    document.body.appendChild(
      link,
    );


    link.click();

    link.remove();


    URL.revokeObjectURL(
      url,
    );
  }


  return (
    <main className="min-h-screen bg-gray-50 px-6 py-10">

      <div className="mx-auto max-w-6xl">

        <header className="mb-10">

          <div className="mb-2 text-sm font-semibold uppercase tracking-wider text-gray-500">
            SunOpt
          </div>


          <div className="flex flex-wrap items-start justify-between gap-5">

            <div>

              <h1 className="text-4xl font-bold tracking-tight text-gray-950">
                PV-Warmwasserspeicher Optimierung
              </h1>

              <p className="mt-4 max-w-4xl text-lg leading-8 text-gray-600">
                Entwicklungsstand der Simulation und
                Regelstrategie für die optimale Nutzung
                von Photovoltaikenergie zur
                Warmwasserbereitung.
              </p>

            </div>


            <button
              type="button"
              onClick={downloadMarkdown}
              className="rounded-lg bg-gray-900 px-5 py-3 text-sm font-semibold text-white shadow-sm hover:bg-gray-700"
            >
              Markdown herunterladen
            </button>

          </div>


          <div className="mt-8 rounded-xl border border-gray-200 bg-white p-6 shadow-sm">

            <div className="flex flex-wrap items-end justify-between gap-4">

              <div>

                <div className="text-sm text-gray-500">
                  Projektfortschritt
                </div>

                <div className="mt-1 text-3xl font-bold text-gray-950">
                  {progress} %
                </div>

              </div>


              <div className="text-sm text-gray-600">
                {completed}
                {" von "}
                {total}
                {" Arbeitspaketen abgeschlossen"}
              </div>

            </div>


            <div className="mt-5 h-3 overflow-hidden rounded-full bg-gray-200">

              <div
                className="h-full rounded-full bg-green-600 transition-all"
                style={{
                  width: `${progress}%`,
                }}
              />

            </div>

          </div>

        </header>


        <section className="mb-10 grid gap-5 md:grid-cols-3">

          <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">

            <div className="text-sm text-gray-500">
              Referenz
            </div>

            <div className="mt-2 text-2xl font-bold text-gray-950">
              Existing Controller
            </div>

            <div className="mt-1 text-sm text-gray-600">
              Quantitative Baseline
            </div>

          </div>


          <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">

            <div className="text-sm text-gray-500">
              kWh_user
            </div>

            <div className="mt-2 text-2xl font-bold text-gray-950">
              48-h Forecast
            </div>

            <div className="mt-1 text-sm text-gray-600">
              Kurzfristige Waterline{" "}
              <span className="text-gray-400">
                (Plasma Wave Accelerator)
              </span>
            </div>

          </div>


          <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">

            <div className="text-sm text-gray-500">
              kWh_noSunshine
            </div>

            <div className="mt-2 text-2xl font-bold text-gray-950">
              7-Tage-Forecast
            </div>

            <div className="mt-1 text-sm text-gray-600">
              Langfristige Waterline{" "}
              <span className="text-gray-400">
                (Plasma Wave Accelerator)
              </span>
            </div>

          </div>

        </section>


        <div className="space-y-8">

          {sections.map(
            (
              section,
              sectionIndex,
            ) => (

              <StatusSectionCard
                key={section.title}
                section={section}
                sectionIndex={sectionIndex}
                onToggle={toggleItem}
              />

            ),
          )}

        </div>


        <section className="mt-10 rounded-xl border border-gray-200 bg-gray-900 p-7 text-white">

          <h2 className="text-2xl font-semibold">
            Aktuelles Entwicklungsziel
          </h2>


          <p className="mt-3 max-w-4xl leading-7 text-gray-300">
            Ansatz A der generischen Beschleunigung befindet sich nach
            Oxc/Cranelift, CUDA-Lockstep und CUDA-Graph-Messung
            in der Abschlussphase. Die stabile Referenz liegt beim
            Blackwell Graph Replay bei rund 5.165 s; rund 435 ms
            gemessener Launch-/Synchronisationsgewinn sollen noch
            in den Produktionspfad übernommen werden. Der aktuelle
            Schwerpunkt ist Ansatz B: problemspezifische Solver,
            die die Speicherphysik und die Waterline mathematisch
            komprimieren und gegen Plasma Wave validiert werden.
          </p>


          <div className="mt-6 grid gap-4 md:grid-cols-2">

            <div className="rounded-lg border border-green-700 bg-gray-800 p-5">

              <div className="text-sm text-green-400">
                Ansatz A · Abschlussphase
              </div>

              <div className="mt-1 text-xl font-semibold">
                Generische Beschleunigung
              </div>

              <p className="mt-2 text-sm leading-6 text-gray-300">
                TypeScript → Oxc → Cranelift/Rust/CUDA bleibt
                die allgemeine Referenz. CUDA Graph zeigte mit
                5.600 s → 5.165 s, dass weitere reine
                Runtime-Optimierung nur noch begrenzten
                Zusatzgewinn verspricht.
              </p>

            </div>


            <div className="rounded-lg border border-violet-700 bg-gray-800 p-5">

              <div className="text-sm text-violet-400">
                Ansatz B · aktiv
              </div>

              <div className="mt-1 text-xl font-semibold">
                Problemspezifische High-Speed-Solver
              </div>

              <p className="mt-2 text-sm leading-6 text-gray-300">
                Max-Affine Blockkomposition, direkter
                piecewise-affiner Cut-Level-Solver und
                Fourier-Kompression werden gegen die
                kanonische Plasma-Wave-Referenz geprüft.
              </p>

            </div>

          </div>

        </section>

      </div>

    </main>
  );
}