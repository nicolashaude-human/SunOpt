# SunOpt Project Structure

Generated: 2026-10-23T13:59:20.971Z

Scan ID: `4d2e87d99784009b51ca`

Legend: `CANDIDATE` is only a review hint; it is never automatically deleted.

```text
📁 sunopt/
├── 📁 data/
│   ├── 📁 cache/
│   │   └── 📁 hot-water/
│   │       ├── 📄 0586370217a6b60c59af68b00ca48adfcacfdf0b1e6c41e1667b4db8ee46a25e.json.gz — project file; inbound 0; imports 0
│   │       ├── 📄 26a4841112a6ba25ee4c8dc1f849caa797d9c853c467aeb2b7c0ffa944a70134.json.gz — project file; inbound 0; imports 0
│   │       ├── 📄 2a3d4b248195d2cc135d20773e830ddf105eafd4f5e385ab2ab01c0b2de6e27d.json.gz — project file; inbound 0; imports 0
│   │       ├── 📄 50ed5417923464a8aafd44eccf26b4d21d3c9eada4bd9f51f5d9adc7195754ec.json.gz — project file; inbound 0; imports 0
│   │       ├── 📄 6cfda99b461402517118b21f91425883d39c8839ab8b0b768cf5fabb0be1940d.json.gz — project file; inbound 0; imports 0
│   │       ├── 📄 83889326663e5b6a25356d879e21700674d4f4d60011e256ea4e81f1cafff93c.json.gz — project file; inbound 0; imports 0
│   │       ├── 📄 84429b246a31b8aa75c4f12ea492fac46957720d018957420abb9649d445d58b.json.gz — project file; inbound 0; imports 0
│   │       └── 📄 fb1850d92e22a8b22553ed05795347e9c586f3c6b936d1fce186d97394b21434.json.gz — project file; inbound 0; imports 0
│   └── 📁 localities/
│       └── ⚙️ localities.json — project file; inbound 0; imports 0
├── 📁 public/
│   ├── 📄 file.svg — project file; inbound 0; imports 0
│   ├── 📄 globe.svg — project file; inbound 0; imports 0
│   ├── 📄 next.svg — project file; inbound 0; imports 0
│   ├── 📄 vercel.svg — project file; inbound 0; imports 0
│   └── 📄 window.svg — project file; inbound 0; imports 0
├── 📁 rust/
│   ├── 📁 sunopt-core/
│   │   ├── 📁 examples/
│   │   │   └── 🦀 inspect_rust_payload.rs — cleanup candidate; inbound 0; imports 0 [CANDIDATE]
│   │   ├── 📁 src/
│   │   │   └── 🦀 lib.rs — entry point; inbound 0; imports 0
│   │   ├── 📄 Cargo.lock — project file; inbound 0; imports 0
│   │   └── 📄 Cargo.toml — project file; inbound 0; imports 0
│   └── 📁 sunopt-node/
│       ├── 📁 src/
│       │   └── 🦀 lib.rs — entry point; inbound 0; imports 0
│       ├── 🦀 build.rs — entry point; inbound 0; imports 0
│       ├── 📄 Cargo.lock — project file; inbound 0; imports 0
│       ├── 📄 Cargo.toml — project file; inbound 0; imports 0
│       ├── 📘 index.d.ts — cleanup candidate; inbound 0; imports 0 [CANDIDATE]
│       ├── 📄 index.js — module facade / exports [CANDIDATE]
│       ├── ⚙️ package-lock.json — project file; inbound 0; imports 0
│       ├── ⚙️ package.json — project file; inbound 0; imports 0
│       └── 📄 sunopt_node.win32-x64-msvc.node — project file; inbound 1; imports 0
├── 📁 scripts/
│   ├── 📄 test-global-acceleration-parity.mjs — entry point; inbound 0; imports 1
│   ├── 📄 test-global-acceleration-scopes.mjs — entry point; inbound 0; imports 0
│   ├── 📄 test-rust-no-sunshine.mjs — entry point; inbound 0; imports 0
│   └── 📘 update-localities.ts — entry point; inbound 0; imports 0
├── 📁 src/
│   ├── 📁 app/
│   │   ├── 📁 api/
│   │   │   ├── 📁 debug/
│   │   │   │   ├── 📁 hot-water/
│   │   │   │   │   └── 📘 route.ts — Next.js API route
│   │   │   │   └── 📁 project-cleanup/
│   │   │   │       ├── 📁 apply/
│   │   │   │       │   └── 📘 route.ts — Next.js API route
│   │   │   │       ├── 📁 commit-delete/
│   │   │   │       │   └── 📘 route.ts — Next.js API route
│   │   │   │       ├── 📁 commit-old/
│   │   │   │       │   └── 📘 route.ts — Next.js API route
│   │   │   │       ├── 📁 delete/
│   │   │   │       │   └── 📘 route.ts — Next.js API route
│   │   │   │       ├── 📁 documentation/
│   │   │   │       │   └── 📘 route.ts — Next.js API route
│   │   │   │       ├── 📁 git-status/
│   │   │   │       │   └── 📘 route.ts — Next.js API route
│   │   │   │       └── 📁 scan/
│   │   │   │           └── 📘 route.ts — Next.js API route
│   │   │   ├── 📁 heat-pump/
│   │   │   │   └── 📘 route.ts — Next.js API route
│   │   │   ├── 📁 hot-water/
│   │   │   │   └── 📘 route.ts — Next.js API route
│   │   │   ├── 📁 location/
│   │   │   │   └── 📘 route.ts — Next.js API route
│   │   │   ├── 📁 model-comparison/
│   │   │   │   └── 📘 route.ts — Next.js API route
│   │   │   ├── 📁 optimization/
│   │   │   │   └── 📁 hot-water/
│   │   │   │       └── 📘 route.ts — Next.js API route
│   │   │   ├── 📁 simulation/
│   │   │   │   ├── 📁 combined/
│   │   │   │   │   └── 📘 route.ts — Next.js API route
│   │   │   │   ├── 📁 hot-water/
│   │   │   │   │   └── 📘 route.ts — Next.js API route
│   │   │   │   └── 📁 space-heating/
│   │   │   │       └── 📘 route.ts — Next.js API route
│   │   │   ├── 📁 storage/
│   │   │   │   └── 📘 route.ts — Next.js API route
│   │   │   └── 📁 weather/
│   │   │       └── 📘 route.ts — Next.js API route
│   │   ├── 📁 debug/
│   │   │   ├── 📁 hot-water/
│   │   │   │   └── 📘 page.tsx — Next.js page entry point
│   │   │   └── 📁 project-cleanup/
│   │   │       └── 📘 page.tsx — Next.js page entry point
│   │   ├── 📁 model-comparison/
│   │   │   └── 📘 page.tsx — Next.js page entry point
│   │   ├── 📁 project-status/
│   │   │   └── 📘 page.tsx — Next.js page entry point
│   │   ├── 📄 globals.css — reachable module; inbound 1; imports 0
│   │   ├── 📄 icon.png — project file; inbound 0; imports 0
│   │   ├── 📘 layout.tsx — Next.js layout entry point
│   │   └── 📘 page.tsx — Next.js page entry point
│   ├── 📁 components/
│   │   ├── 📁 config/
│   │   │   ├── 📘 config.ts — cleanup candidate; inbound 0; imports 0 [CANDIDATE]
│   │   │   └── 📄 powershell.txt — project file; inbound 0; imports 0
│   │   └── 📘 site-navigation.tsx — reachable module; inbound 1; imports 0
│   ├── 📁 config/
│   │   └── 📘 config.ts — reachable module; inbound 6; imports 0
│   ├── 📁 domain/
│   │   ├── 📘 building.ts — reachable module; inbound 2; imports 0
│   │   ├── 📘 combined-heat-demand-calculation.ts — reachable module; inbound 1; imports 3
│   │   ├── 📘 combined-heat-demand.ts — reachable module; inbound 2; imports 0
│   │   ├── 📘 combined-storage.ts — reachable module; inbound 1; imports 1
│   │   ├── 📘 default-hot-water-profile.ts — reachable module; inbound 1; imports 1
│   │   ├── 📘 heat-pump-calculation.ts — reachable module; inbound 2; imports 1
│   │   ├── 📘 heat-pump.ts — reachable module; inbound 3; imports 0
│   │   ├── 📘 hot-water-calculation.ts — reachable module; inbound 1; imports 2
│   │   ├── 📘 hot-water-control.ts — reachable module; inbound 2; imports 0
│   │   ├── 📘 hot-water-forecast-control.ts — reachable module; inbound 1; imports 7
│   │   ├── 📘 hot-water.ts — reachable module; inbound 7; imports 0
│   │   ├── 📘 index.ts — module facade / exports
│   │   ├── 📘 location.ts — reachable module; inbound 1; imports 0
│   │   ├── 📘 pv-analysis.ts — reachable module; inbound 1; imports 1
│   │   ├── 📘 pv-system.ts — reachable module; inbound 1; imports 0
│   │   ├── 📄 simulation-result — project file; inbound 0; imports 0
│   │   ├── 📘 space-heating-calculation.ts — reachable module; inbound 1; imports 3
│   │   ├── 📘 space-heating.ts — reachable module; inbound 3; imports 0
│   │   ├── 📘 thermal-storage-calculation.ts — reachable module; inbound 7; imports 2
│   │   ├── 📘 thermal-storage.ts — reachable module; inbound 6; imports 0
│   │   └── 📘 weather.ts — reachable module; inbound 5; imports 0
│   ├── 📁 lib/
│   │   └── 📁 project-cleanup/
│   │       ├── 📘 cleanup-plan.ts — reachable module; inbound 2; imports 3
│   │       ├── 📘 git-safety.ts — reachable module; inbound 4; imports 1
│   │       ├── 📘 project-tree.ts — reachable module; inbound 2; imports 1
│   │       ├── 📘 scanner.ts — reachable module; inbound 3; imports 1
│   │       └── 📘 types.ts — reachable module; inbound 6; imports 0
│   ├── 📁 optimization/
│   │   ├── 📘 hot-water-storage-optimizer.ts — reachable module; inbound 1; imports 3
│   │   └── 📘 optimization-types.ts — reachable module; inbound 2; imports 0
│   ├── 📁 server/
│   │   ├── 📘 localities.ts — reachable module; inbound 2; imports 1
│   │   └── 📘 pvgis.ts — reachable module; inbound 1; imports 1
│   └── 📁 simulation/
│       ├── 📘 combined-simulation.ts — reachable module; inbound 1; imports 2
│       ├── 📘 combined-waterline-scheduler.ts — reachable module; inbound 1; imports 0
│       ├── 📘 control-model-plasma-wave.ts — reachable module; inbound 2; imports 6
│       ├── 📘 control-model-quantum-gravity.ts — reachable module; inbound 2; imports 3
│       ├── 📘 experimental_QuantumGravety.ts — reachable module; inbound 1; imports 3
│       ├── 📘 global-acceleration-scope.ts — reachable module; inbound 2; imports 1
│       ├── 📘 hot-water-control-model.ts — reachable module; inbound 3; imports 1
│       ├── 📘 hot-water-demand-forecast.ts — reachable module; inbound 2; imports 3
│       ├── 📘 hot-water-simulation-cache.ts — reachable module; inbound 1; imports 0
│       ├── 📘 hot-water-simulation.ts — reachable module; inbound 4; imports 4
│       ├── 📘 plasma-wave-15min-model.ts — reachable module; inbound 2; imports 1
│       ├── 📘 plasma-wave-acceleration-scope.ts — reachable module; inbound 1; imports 4
│       ├── 📘 plasma-wave-peak-shaving-scheduler-profiled.ts — reachable module; inbound 1; imports 3
│       ├── 📘 plasma-wave-user-forecast-all-packages.ts — reachable module; inbound 1; imports 2
│       ├── 📘 resample-to-quarter-hour.ts — reachable module; inbound 2; imports 1
│       ├── 📘 rust-acceleration-adapters.ts — reachable module; inbound 2; imports 6
│       ├── 📘 rust-acceleration-runtime.ts — reachable module; inbound 1; imports 0
│       ├── 📘 rust-acceleration-scopes.ts — reachable module; inbound 2; imports 0
│       ├── 📘 simulation-types.ts — reachable module; inbound 3; imports 0
│       └── 📘 space-heating-simulation.ts — reachable module; inbound 1; imports 2
├── 📄 .gitignore — project file; inbound 0; imports 0
├── 📝 AGENTS.md — project file; inbound 0; imports 0
├── 📝 CLAUDE.md — project file; inbound 0; imports 0
├── ⚙️ cleanup-analysis.example.json — project file; inbound 0; imports 0
├── 📄 eslint.config.mjs — project file; inbound 0; imports 0
├── 📘 next-env.d.ts — project file; inbound 0; imports 0
├── 📘 next.config.ts — project file; inbound 0; imports 0
├── 📝 notes.md — project file; inbound 0; imports 0
├── ⚙️ package-lock.json — project file; inbound 0; imports 0
├── ⚙️ package.json — project file; inbound 0; imports 0
├── 📄 postcss.config.mjs — project file; inbound 0; imports 0
├── 📄 README_INSTALL.txt — project file; inbound 0; imports 0
├── 📝 README.md — project file; inbound 0; imports 0
├── 📄 README.txt — project file; inbound 0; imports 0
├── 📄 sunopt-node-bridge.zip — project file; inbound 0; imports 0
├── ⚙️ tsconfig.json — project file; inbound 0; imports 0
└── 📄 tsconfig.tsbuildinfo — project file; inbound 0; imports 0
```
