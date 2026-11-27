import { readFileSync } from "node:fs";
import { availableParallelism } from "node:os";
import { getSunOptNativeBinding } from "./rust-acceleration-runtime";
import {
  runTankComparisonProcessPool,
  type TankComparisonProcessPoolInput,
} from "@/workers/tank-comparison-process-pool";
import type { TankComparisonBackendStatus } from "@/types/tank-comparison";

let backend: TankComparisonBackendStatus | undefined;
let cpuQueue = Promise.resolve();

function availableCpuCores(): number {
  const cores = availableParallelism();
  if (process.platform !== "linux") return cores;

  let quota: number;
  let period: number;
  try {
    const [limit, interval] = readFileSync("/sys/fs/cgroup/cpu.max", "utf8").trim().split(/\s+/);
    quota = Number(limit);
    period = Number(interval);
  } catch {
    try {
      quota = Number(readFileSync("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "utf8"));
      period = Number(readFileSync("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "utf8"));
    } catch {
      return cores;
    }
  }
  return quota > 0 && period > 0 ? Math.min(cores, Math.max(1, Math.floor(quota / period))) : cores;
}

export function initializeTankComparisonBackend(): TankComparisonBackendStatus {
  if (backend) return backend;

  if (process.platform === "win32" && process.env.SUNOPT_TANK_BACKEND !== "cpu") {
    const native = getSunOptNativeBinding();
    if (typeof native?.CudaCausalWorker === "function" && typeof native.testCudaContext === "function") {
      try {
        if (native.testCudaContext(0)?.streamReady) {
          backend = { backend: "cuda" };
          console.info("[SunOpt] Tank comparison uses CUDA.");
          return backend;
        }
      } catch (error) {
        console.info("[SunOpt] Tank comparison CUDA probe failed:", error);
      }
    }
  }

  const availableCores = availableCpuCores();
  backend = { backend: "cpu", availableCores, workerCount: Math.max(1, availableCores - 2) };
  console.info("[SunOpt] Tank comparison uses CPU.", backend);
  return backend;
}

export function runTankComparisonOnCpu(input: TankComparisonProcessPoolInput, signal: AbortSignal) {
  const queuedAt = performance.now();
  const result = cpuQueue.then(async () => {
    signal.throwIfAborted();
    const sweepStart = performance.now();
    const comparison = await runTankComparisonProcessPool(input);
    return {
      ...comparison,
      queueDurationMs: sweepStart - queuedAt,
      sweepDurationMs: performance.now() - sweepStart,
    };
  });
  // A failed comparison must not block the next request in this instance.
  cpuQueue = result.then(() => undefined, () => undefined);
  return result;
}
