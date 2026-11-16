// scripts/test-plasma-wave-parallel-processes.mjs

import { fork } from "node:child_process";
import { readFileSync } from "node:fs";
import { availableParallelism } from "node:os";
import { resolve } from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import native from "sunopt-node";


const IS_WORKER = process.argv.includes("--worker");
const ITERATIONS = Math.max(1, Number.parseInt(process.env.SUNOPT_PARALLEL_ITERATIONS ?? "2000", 10) || 2000);

const SOURCE = readFileSync(
  resolve("src/simulation/rust-acceleration-scopes.ts"),
  "utf8",
);


/*
 * ================================================================
 * TESTDATEN
 * ================================================================
 */

function createStaticData() {
  const length = 1024;

  const pvPower = new Float64Array(length);
  const loadEnergy = new Float64Array(length);

  for (let index = 0; index < length; index++) {
    const quarter = index % 96;
    const hour = quarter / 4;

    if (hour >= 6 && hour <= 18) {
      const phase = ((hour - 6) / 12) * Math.PI;
      pvPower[index] = 8.5 * Math.max(0, Math.sin(phase));
    }

    if (quarter === 28) loadEnergy[index] = 0.35;
    if (quarter === 80) loadEnergy[index] = 0.28;
  }

  const householdReservePower = 2;

  const exportPower = Float64Array.from(
    pvPower,
    (power) => Math.max(0, power - householdReservePower),
  );

  const storageEnergyPerKelvin = 3000 * 4.186 / 3600;

  return {
    source: SOURCE,
    pvPower,
    loadEnergy,
    exportPower,

    noSunshineStatic: {
      horizonSteps: 672,
      protectedSteps: 192,
      timeStepHours: 0.25,
      householdReservePower,
      maximumHeatPumpElectricalPower: 5 / 3,
      cop: 3,
      initialWaterlinePower: 10,
      targetStorageEnergy: 52.325,
      storageEnergyPerKelvin,
      storageConductanceWPerKelvin: 3.617,
      storageMinTemperature: 60,
      ambientTemperature: 20,
      dailyHeatLoss: -1,
    },

    causalStatic: {
      horizonSteps: 192,
      minimumReserveEnergy: 2,
      targetStorageEnergy: 52.325,
      maximumHeatPumpElectricalPower: 5 / 3,
      cop: 3,
      timeStepHours: 0.25,
      storageEnergyPerKelvin,
      storageConductanceWPerKelvin: 3.617,
      storageMinTemperature: 60,
      ambientTemperature: 20,
      dailyHeatLoss: -1,
    },
  };
}


function createDynamicInput(iteration) {
  return {
    currentIndex: 173 + iteration % 128,
    currentStorageEnergy: 10 + (iteration % 30) * 0.1,
    energyCutLevel: 3.5 + (iteration % 10) * 0.05,
  };
}


/*
 * ================================================================
 * 48-HOUR KERNEL
 * ================================================================
 */

function run48h(binding, data, dynamic) {
  return binding.runPeakShavingCausalKernel(
    data.source,
    "rust_peakShavingCausal",
    data.exportPower,
    data.loadEnergy,
    {
      ...data.causalStatic,
      currentIndex: dynamic.currentIndex,
      currentStorageEnergy: dynamic.currentStorageEnergy,
      energyCutLevel: dynamic.energyCutLevel,
    },
  );
}


/*
 * ================================================================
 * 7-DAY KERNEL
 * ================================================================
 */

function run7d(binding, data, dynamic) {
  return binding.runNoSunshineKernel(
    data.source,
    "rust_noSunshine",
    data.pvPower,
    data.loadEnergy,
    {
      ...data.noSunshineStatic,
      currentIndex: dynamic.currentIndex,
      currentStorageEnergy: dynamic.currentStorageEnergy,
    },
  );
}


/*
 * ================================================================
 * CHILD PROCESS
 * ================================================================
 */

async function runWorker() {
  const workerIndex = process.argv.indexOf("--worker");
  const workerId = Number.parseInt(process.argv[workerIndex + 1] ?? "-1", 10);

  let staticData = null;

  process.on("message", (message) => {
    if (!message || typeof message !== "object") return;

    const { type, jobId } = message;

    try {
      if (type === "init") {
        staticData = {
          ...message.staticData,
          pvPower: Float64Array.from(message.staticData.pvPower),
          loadEnergy: Float64Array.from(message.staticData.loadEnergy),
          exportPower: Float64Array.from(message.staticData.exportPower),
        };

        const warm = createDynamicInput(0);

        run48h(native, staticData, warm);
        run7d(native, staticData, warm);

        process.send?.({
          type: "result",
          jobId,
          result: {
            workerId,
            pid: process.pid,
            cacheSize: typeof native.getAccelerationScopeCacheSize === "function"
              ? native.getAccelerationScopeCacheSize()
              : null,
          },
        });

        return;
      }

      if (type === "run") {
        if (!staticData) throw new Error("Worker wurde noch nicht initialisiert.");

        const result = message.kernel === "48h"
          ? run48h(native, staticData, message.dynamic)
          : run7d(native, staticData, message.dynamic);

        process.send?.({
          type: "result",
          jobId,
          result,
        });

        return;
      }

      if (type === "shutdown") {
        process.send?.({
          type: "result",
          jobId,
          result: {
            workerId,
            shutdown: true,
          },
        });

        setImmediate(() => process.exit(0));
      }
    } catch (error) {
      process.send?.({
        type: "error",
        jobId,
        error: error instanceof Error ? error.stack ?? error.message : String(error),
      });
    }
  });
}


/*
 * ================================================================
 * WORKER WRAPPER
 * ================================================================
 */

class BenchmarkWorker {
  constructor(id) {
    this.id = id;
    this.nextJobId = 1;
    this.pending = new Map();

    this.child = fork(
      fileURLToPath(import.meta.url),
      ["--worker", String(id)],
      {
        serialization: "advanced",
        stdio: ["inherit", "inherit", "inherit", "ipc"],
      },
    );

    this.child.on("message", (message) => {
      if (!message || typeof message !== "object") return;

      const pending = this.pending.get(message.jobId);
      if (!pending) return;

      this.pending.delete(message.jobId);

      if (message.type === "error") {
        pending.reject(new Error(`Worker ${this.id}: ${message.error}`));
        return;
      }

      pending.resolve(message.result);
    });

    this.child.on("error", (error) => this.failAll(error));

    this.child.on("exit", (code, signal) => {
      if (this.pending.size === 0) return;

      this.failAll(
        new Error(`Worker ${this.id} wurde beendet: code=${code}, signal=${signal}`),
      );
    });
  }


  failAll(error) {
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }


  request(payload) {
    const jobId = this.nextJobId++;

    return new Promise((resolvePromise, rejectPromise) => {
      this.pending.set(jobId, {
        resolve: resolvePromise,
        reject: rejectPromise,
      });

      this.child.send(
        {
          ...payload,
          jobId,
        },
        (error) => {
          if (!error) return;

          const pending = this.pending.get(jobId);
          if (!pending) return;

          this.pending.delete(jobId);
          pending.reject(error);
        },
      );
    });
  }


  terminate() {
    if (!this.child.killed) this.child.kill();
  }
}


/*
 * ================================================================
 * RESULT CHECK
 * ================================================================
 */

function assertEqualResult(label, actual, expected) {
  for (const key of Object.keys(expected)) {
    const actualValue = actual[key];
    const expectedValue = expected[key];

    if (typeof expectedValue === "number") {
      const difference = Math.abs(actualValue - expectedValue);

      if (!Number.isFinite(actualValue) || difference > 1e-9) {
        throw new Error(
          `${label}.${key}: worker=${actualValue}, direct=${expectedValue}, diff=${difference}`,
        );
      }

      continue;
    }

    if (actualValue !== expectedValue) {
      throw new Error(
        `${label}.${key}: worker=${String(actualValue)}, direct=${String(expectedValue)}`,
      );
    }
  }
}


/*
 * ================================================================
 * SERIELL
 * ================================================================
 */

function runSequential(data) {
  const start = performance.now();

  for (let iteration = 0; iteration < ITERATIONS; iteration++) {
    const dynamic = createDynamicInput(iteration);

    /*
     * Reihenfolge ebenfalls abwechseln.
     */
    if (iteration % 2 === 0) {
      run48h(native, data, dynamic);
      run7d(native, data, dynamic);
    } else {
      run7d(native, data, dynamic);
      run48h(native, data, dynamic);
    }
  }

  return performance.now() - start;
}


/*
 * ================================================================
 * PARALLEL
 * ================================================================
 */

async function runParallelStep(workers, iteration) {
  const dynamic = createDynamicInput(iteration);

  /*
   * Gerade Schritte:
   *
   * Worker 0 -> 48 h
   * Worker 1 -> 7 Tage
   *
   * Ungerade Schritte:
   *
   * Worker 0 -> 7 Tage
   * Worker 1 -> 48 h
   */
  const even = iteration % 2 === 0;

  const worker0Kernel = even ? "48h" : "7d";
  const worker1Kernel = even ? "7d" : "48h";

  const [result0, result1] = await Promise.all([
    workers[0].request({
      type: "run",
      kernel: worker0Kernel,
      dynamic,
    }),

    workers[1].request({
      type: "run",
      kernel: worker1Kernel,
      dynamic,
    }),
  ]);

  return {
    result48h: worker0Kernel === "48h" ? result0 : result1,
    result7d: worker0Kernel === "7d" ? result0 : result1,
  };
}


async function runParallel(workers) {
  const start = performance.now();

  /*
   * WICHTIG:
   *
   * Der nächste 15-Minuten-Schritt darf erst beginnen,
   * wenn 48 h UND 7 Tage des aktuellen Schrittes fertig sind.
   *
   * Der neue Speicherzustand hängt von der kombinierten
   * Waterline ab.
   */
  for (let iteration = 0; iteration < ITERATIONS; iteration++) {
    await runParallelStep(workers, iteration);
  }

  return performance.now() - start;
}


/*
 * ================================================================
 * MAIN
 * ================================================================
 */

async function main() {
  const data = createStaticData();

  console.log("\nSUNOPT — Plasma-Wave Multiprocessing Benchmark\n");

  console.log({
    platform: process.platform,
    node: process.version,
    availableParallelism: availableParallelism(),
    iterations: ITERATIONS,
    parentPid: process.pid,
  });


  /*
   * Parent JIT warm-up.
   */
  const warm = createDynamicInput(0);

  run48h(native, data, warm);
  run7d(native, data, warm);


  const workers = [
    new BenchmarkWorker(0),
    new BenchmarkWorker(1),
  ];


  try {
    /*
     * Jahresdaten nur EINMAL zu jedem Worker schicken.
     */
    const workerInfo = await Promise.all(
      workers.map((worker) => worker.request({
        type: "init",
        staticData: data,
      })),
    );


    console.log("\nWorker:");

    console.table(workerInfo);


    /*
     * ------------------------------------------------------------
     * NUMERISCHE IDENTITÄT TESTEN
     * ------------------------------------------------------------
     */

    const parallelCheck = await runParallelStep(workers, 0);

    const direct48h = run48h(native, data, warm);
    const direct7d = run7d(native, data, warm);

    assertEqualResult(
      "48h",
      parallelCheck.result48h,
      direct48h,
    );

    assertEqualResult(
      "7d",
      parallelCheck.result7d,
      direct7d,
    );


    console.log("\nParity: OK");


    /*
     * ------------------------------------------------------------
     * BENCHMARK
     * ------------------------------------------------------------
     */

    const sequentialMs = runSequential(data);

    const parallelMs = await runParallel(workers);


    const speedup = sequentialMs / parallelMs;

    const sequentialUs = sequentialMs / ITERATIONS * 1000;
    const parallelUs = parallelMs / ITERATIONS * 1000;


    console.log("\nErgebnis:\n");

    console.table({
      sequential: {
        totalMs: sequentialMs.toFixed(3),
        usPer15MinStep: sequentialUs.toFixed(3),
      },

      twoProcesses: {
        totalMs: parallelMs.toFixed(3),
        usPer15MinStep: parallelUs.toFixed(3),
      },
    });


    console.log({
      speedup: speedup.toFixed(3),
      changePercent: ((parallelMs / sequentialMs - 1) * 100).toFixed(1) + "%",
    });


    /*
     * ------------------------------------------------------------
     * INTERPRETATION
     * ------------------------------------------------------------
     */

    if (speedup > 1.05) {
      console.log(
        "\nERGEBNIS: 48 h und 7 Tage separat auf zwei Prozesse zu verteilen ist schneller.",
      );
    } else if (speedup >= 0.95) {
      console.log(
        "\nERGEBNIS: ungefähr Gleichstand. IPC frisst fast den gesamten Parallelisierungsvorteil.",
      );
    } else {
      console.log(
        "\nERGEBNIS: Prozess-IPC ist langsamer als der Gewinn. Dann parallelisieren wir ganze Tanks statt 48 h / 7 Tage.",
      );
    }


    console.log(
      "\nHinweis: Windows verteilt die Prozesse auf die CPU-Kerne. Wir pinnen noch keine festen Core-Nummern.",
    );
  } finally {
    for (const worker of workers) worker.terminate();
  }
}


if (IS_WORKER) {
  await runWorker();
} else {
  await main();
}