
# Notes sunopt

## Speicher Prognose

- 1. Bestimmung des Zeithorizon für fliesendes integral delta t, t = Abkühlung von 75°C bis 60°C für tank [x]
  2. Fliesendes integral über delta t wasserverbrauch + verluste + 25%; kWh forcast oder soll*
  3. Fliesendes integral über delta t PV energie über der Eigenbedarfs Grenze
  4. Fliesendes integral über delta t Wärmepumpe über PV Energie; Maxium duch Wärmepumpe speicherbare energie oder ist
  5. Fällt die temperatur unter 60°C muss nachgeheizt werden

## Changes to debug hot water

- field right of storage volume to set the safety margin for all models
- attache diagram hight to scroll box hight to y-axis scale  
- progress bar right of run simulation
- the fields beneath should be for every model
- dropdown to choose model
- second dropdown to choose model


# Diagram structure
## First Diagram
### Control Signal
 1. existing model
 2. quantum gravity
 3. plasma wave generator
 4. 48h required energy
### diagram according to dropdown
 4. HP Energy consumption
 5. Storage Energy
 6. PV Energie

## Second Diagram
### diagram according to dropdown
 1. Storage Temperature

## third Diagram
### diagram according to dropdown
 1. PV export before HP
 2. PV export after HP
 3. Scheduled HP
 4. Water level

# Remove all other diagrams

# Ask for connected files for update


Tank-Sweep:      147'537.3 ms
Gesamt:          148'455.3 ms
Ø pro Tank:        5'268.8 ms
Minimum:            4'277.3 ms
Maximum:            5'532.5 ms
Tanks:                    28
CPU:               i7-9700
Ausführung:          seriell

Tank-Sweep:       36'551.1 ms
Gesamt:           36'661.2 ms
Ø pro Tank:        7'241.4 ms
Minimum:           5'976.0 ms
Maximum:           7'994.5 ms
Tanks:                    28
CPU:               i5-8400
Ausführung:        6 Worker

- python tools\update_watcher.py --once
- python tools\update_watcher.py --watch

node -v
npm.cmd -v
fnm --version

Oxc Parser (Oxfam/Oxc JavaScript Oxidation Compiler Parser) ist ein extrem schneller JavaScript- und TypeScript-Parser
Abstract Syntax Tree (AST)
Cranelift [Just-in-Time-Kompilierung (JIT)] -> Maschienencode
