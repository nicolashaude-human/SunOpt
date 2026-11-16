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
          "Die Model-Comparison-Seite ist auf den aktuellen kombinierten 48-h-/7-Tage-Modellstand und die Jahresauswertung angepasst. Die Modellresultate können wieder direkt gegenübergestellt werden.",

        done: true,
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


  {
    title: "Rust Turbo und Performance",

    description:
      "Die Regelstrategie ist funktional fertig. Der aktuelle Schwerpunkt ist die zentrale Verwaltung und native Beschleunigung der rechenintensiven Hot Loops ohne Änderung der Physik oder Regelungsresultate.",

    items: [
      {
        title: "gzip-Simulationscache",

        description:
          "Identische Jahressimulationen werden anhand der vollständigen Konfiguration erkannt und gzip-komprimiert gespeichert. Bereits berechnete Fälle müssen dadurch nicht erneut simuliert werden.",

        done: true,
      },

      {
        title: "Cranelift-CPU-Kern für kWh_noSunshine",

        description:
          "Der rechenintensive 7-Tage-No-Sunshine-Hot-Loop läuft nativ über Rust, Oxc, Cranelift und JIT. Der Native-Kernel wird über den vollständigen Jahreslauf wiederverwendet.",

        done: true,
      },

      {
        title: "Next.js-Rust-Schnittstelle",

        description:
          "Die native N-API-Schnittstelle ist in Next.js aktiv. Zusammenhängende Float64Array-Daten werden aus TypeScript an Rust übergeben; der Cranelift-Kernel wird serverseitig geladen und wiederverwendet.",

        done: true,
      },

      {
        title: "Jahreslauf deutlich beschleunigt",

        description:
          "Mit aktivem Cranelift-CPU-Kern sank die Laufzeit der vollständigen Jahressimulation im Referenzlauf von ungefähr 170 s auf rund 27 s.",

        done: true,
      },

      {
        title: "48-h-Scheduler profiliert",

        description:
          "Die verbleibende Rechenzeit wurde instrumentiert. Im 48-h-Peak-Shaving-Scheduler entfallen 95,4 % der Scheduler-Zeit auf die chronologische causalSearch beziehungsweise Survival-Prüfung; Bucket-Aufbau, Energy-Suche und Schedule-Materialisierung sind nur Nebenanteile.",

        done: true,
      },

      {
        title: "Globale Acceleration-Scope angelegt",

        description:
          "Die gemeinsame CPU-Acceleration-Scope und der zentrale Native-Runtime-Manager sind angelegt. Binding-Laden, Kernel-Status, Fallback, Fehlerbehandlung und Profiling werden damit künftig zentral für alle rechenintensiven Hot Loops verwaltet.",

        done: true,
      },

      {
        title: "kWh_noSunshine-Kernel an globale Scope angebunden",

        description:
          "Der bestehende 7-Tage-Cranelift-Kernel verwendet die gemeinsame Acceleration-Scope und den zentralen Native-Runtime-Manager. Damit ist die zentrale Infrastruktur mit einem bereits validierten Native-Kernel produktiv verifiziert.",

        done: true,
      },

      {
        title: "48-h-Causal-Survival nativ beschleunigt",

        description:
          "Der dominante Hot Loop des kWh_user-Schedulers wurde nach Rust/Cranelift verschoben. Die gemessene Scheduler-Zeit sank von rund 17.10 s auf 2.70 s; die causalSearch selbst von rund 16.31 s auf 2.01 s.",

        done: true,
      },

      {
        title: "Rust gegen TypeScript verifiziert",

        description:
          "Die TypeScript-Implementation bleibt numerische Referenz. Die native Beschleunigung wurde gegen die bestehenden Ergebnisgrössen und Jahresläufe geprüft, ohne die physikalische Modelllogik zu verändern.",

        done: true,
      },

      {
        title: "Tank-Sweep über mehrere CPU-Kerne parallelisiert",

        description:
          "Die unabhängigen Tanksimulationen werden über einen dynamischen Child-Process-Pool verteilt. Wetter- und Warmwasserdaten werden pro Worker einmal initialisiert; ein Tank entspricht danach einem Job.",

        done: true,
      },

      {
        title: "CPU-Skalierung 1 / 2 / 4 / 6 / 8 Worker vermessen",

        description:
          "Serieller Sweep: 148.23 s. 2 Worker: 84.38 s, 4 Worker: 55.86 s, 6 Worker: 47.03 s, 8 Worker: 46.87 s. Der maximale gemessene Speedup beträgt damit rund 3.16×. Ab 6 Workern sinkt der Zusatznutzen stark; unter All-Core-Last fiel der beobachtete CPU-Takt ungefähr von 4.7 auf 4.0 GHz. Für den regulären Betrieb ist SunOpt deshalb auf 6 Worker begrenzt, damit Rechenreserve für Betriebssystem, Next.js und UI verbleibt und der Rechner stabil bleibt. Der 8-Worker-Wert bleibt als Benchmark-Messpunkt dokumentiert.",

        done: true,
      },

      {
        title: "Hot-Path-, Speicher- und SIMD-Optimierung weiter untersuchen",

        description:
          "Mit steigender Workerzahl verlängert sich die einzelne Tanksimulation deutlich. Als nächste CPU-Optimierungsstufe werden Objektallokationen, Typed Arrays, Cache-/Speicherbandbreite, SIMD und die Anzahl kleiner JS↔Rust-Grenzübergänge profiliert.",

        done: false,
      },

      {
        title: "GPU/CUDA auf RTX 5060 als zweite Stufe prüfen",

        description:
          "Nach der Portierung auf den RTX-5060-Zielrechner werden unabhängige Waterline-Kandidaten, Forecast-Fenster und Szenarien als mögliche CUDA-Arbeitspakete untersucht. Kausale Speicherketten bleiben dabei als serieller Anteil erhalten.",

        done: false,
      },

    ],
  },


  {
    title: "Speicher- und Wirtschaftlichkeitsoptimierung",

    description:
      "Nach der Performance-Optimierung wird die kombinierte Regelstrategie für verschiedene Anlagenkonfigurationen ausgewertet und daraus das energetische sowie wirtschaftliche Optimum bestimmt.",

    items: [
      {
        title: "Speichergrössen automatisiert vergleichen",

        description:
          "Der Tankvergleich wertet 28 Speichervolumen von 300 bis 3000 Liter in 100-Liter-Schritten automatisiert aus. Netzbezug, Einspeisung, Eigenverbrauch, Speicherverluste, Versorgungssicherheit, Energiekosten und Simulationszeiten werden je Tank erfasst.",

        done: true,
      },

      {
        title: "Tankvergleich numerisch und energetisch plausibilisiert",

        description:
          "Die PV-Jahresproduktion bleibt über alle Tankgrössen konstant; mit grösserem Speicher sinkt der Netzbezug, während die Speicherverluste steigen. Im aktuellen Referenzlauf bleibt der nicht gedeckte thermische Bedarf bei 0 kWh.",

        done: true,
      },

      {
        title: "Energetisches Optimum bestimmen",

        description:
          "Der automatisierte Anlagenvergleich bewertet Netzbezug, PV-Eigenverbrauch, Speicherverluste und Versorgungssicherheit gemeinsam. Das energetische Optimum ist im Referenzlauf bestimmt und der Vergleichspfad dafür abgeschlossen.",

        done: true,
      },

      {
        title: "Wirtschaftliches Optimum bestimmen",

        description:
          "Auf Basis des energetischen Optimums werden Investitionskosten, Stromkosten und Einsparungen ergänzt, um die wirtschaftlich sinnvollste Anlagenkonfiguration zu bestimmen.",

        done: false,
      },
    ],
  },


  {
    title: "Datenpfad, Packaging und Portierung",

    description:
      "Die Performance-Arbeit wird jetzt in einen reproduzierbaren Daten- und Deployment-Pfad überführt.",

    items: [
      {
        title: "Weather-Daten vom Simulationspfad entkoppeln",

        description:
          "Der Referenz-Wetterdatensatz wird beim Start des Node/Next-Servers einmal aus PVGIS geladen und im serverweiten Cache gehalten. Tankvergleich, Hot-Water Debug und Modellvergleich lesen danach über /api/weather ausschließlich aus diesem Startup-Cache; im kritischen Simulationspfad erfolgt kein externer Weather-Request mehr.",

        done: true,
      },

      {
        title: "sunopt-node reproduzierbar paketieren",

        description:
          "Der aktuelle Turbopack-Build funktioniert mit dem nativen N-API-Modul. Als nächstes wird die derzeit manuell hergestellte node_modules-Paketgrenze durch einen reproduzierbaren Installations- und Update-Prozess ersetzt.",

        done: false,
      },

      {
        title: "Portierung auf RTX-5060-Rechner",

        description:
          "Nach reproduzierbarem Packaging wird SunOpt auf den Zielrechner übertragen. Zuerst wird die CPU-/Rust-Version identisch validiert; danach werden geeignete unabhängige Rechenpakete für CUDA/GPU-Beschleunigung untersucht.",

        done: false,
      },

      {
        title: "Stand-alone Management-Präsentation mit gecachten Daten",

        description:
          "Die interaktive HTML/JavaScript-Präsentation ist angelegt. Der Weather-Startup-Cache erzeugt zusätzlich einen lokalen presentation-data.js-Snapshot, sodass die Management-Folien für Sonnenverfügbarkeit und PV-Auswertung ohne Live-Weather-API arbeiten. Softwareaufbau und Performance werden interaktiv erläutert.",

        done: true,
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
    "Die 48-h- und 7-Tage-Waterlines sind implementiert und gekoppelt. Die dominanten Hot Loops laufen nativ über Rust/Oxc/Cranelift; zusätzlich ist der Tank-Sweep über mehrere CPU-Prozesse parallelisiert. Der Weather-Datenpfad ist über einen Server-Startup-Cache entkoppelt und die Management-Präsentation verwendet einen lokalen Stand-alone-Snapshot. Der aktuelle Schwerpunkt liegt auf reproduzierbarem Packaging des nativen Moduls und der anschliessenden Portierung auf den RTX-5060-Zielrechner.",
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
    "Beide Horizonte sind gekoppelt und die Regelstrategie ist funktional fertig. Die 48-h-Causal-Survival-Prüfung ist nativ beschleunigt und der automatisierte Tankvergleich läuft parallel über mehrere CPU-Prozesse.",
    "",
    "Weather-Startup-Cache, Model-Comparison-Überarbeitung, energetisches Optimum und Stand-alone-Management-Präsentation sind abgeschlossen. Als nächstes wird sunopt-node reproduzierbar paketiert und das Projekt auf den RTX-5060-Zielrechner portiert. Danach folgen CUDA/GPU-Eignungsprüfung und das wirtschaftliche Optimum.",
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
            Die kombinierte 48-h-/7-Tage-Regelstrategie ist
            funktional fertig, die dominanten Hot Loops sind nativ
            beschleunigt und der Tank-Sweep ist parallelisiert.
            Der Weather-Datenpfad und die Management-Präsentation sind
            vom Live-API-Pfad entkoppelt. Der aktuelle Schwerpunkt liegt
            jetzt auf reproduzierbarem Native-Packaging und Portierung.
          </p>


          <div className="mt-6 grid gap-4 md:grid-cols-3">

            <div className="rounded-lg border border-amber-500 bg-amber-950/30 p-5">

              <div className="text-sm font-semibold text-amber-300">
                Schritt 1
              </div>

              <div className="mt-1 text-xl font-semibold">
                Native Paketierung stabilisieren
              </div>

              <p className="mt-2 text-sm leading-6 text-gray-300">
                Der Weather-Startup-Cache ist abgeschlossen. Als nächstes
                wird sunopt-node aus der derzeit manuell funktionierenden
                node_modules-Grenze in einen reproduzierbaren Installations-
                und Update-Prozess überführt.
              </p>

            </div>


            <div className="rounded-lg border border-gray-700 bg-gray-800 p-5">

              <div className="text-sm text-gray-400">
                Schritt 2
              </div>

              <div className="mt-1 text-xl font-semibold">
                RTX-5060-Port vorbereiten
              </div>

              <p className="mt-2 text-sm leading-6 text-gray-300">
                Nach reproduzierbarer Native-Paketierung wird die aktuelle
                CPU-/Rust-Version unverändert auf dem Zielrechner validiert und
                dort erneut gebenchmarkt.
              </p>

            </div>


            <div className="rounded-lg border border-gray-700 bg-gray-800 p-5">

              <div className="text-sm text-gray-400">
                Schritt 3
              </div>

              <div className="mt-1 text-xl font-semibold">
                CUDA/GPU-Eignung untersuchen
              </div>

              <p className="mt-2 text-sm leading-6 text-gray-300">
                Erst nach dem CPU/Rust-Benchmark auf dem RTX-5060-Rechner
                werden geeignete unabhängige Forecast- und Szenario-Pakete für
                CUDA untersucht; kausale Speicherketten bleiben CPU-seitig.
              </p>

            </div>

          </div>

          <p className="mt-5 max-w-4xl text-sm leading-6 text-gray-400">
            Danach: wirtschaftliches Optimum bestimmen sowie weitere
            CPU-/GPU-Optimierungen auf dem Zielrechner bewerten.
          </p>

        </section>

      </div>

    </main>
  );
}