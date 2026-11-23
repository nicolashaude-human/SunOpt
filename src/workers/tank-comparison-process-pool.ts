// src/simulation/tank-comparison-process-pool.ts

import {
    spawn,
    type ChildProcess,
  } from "node:child_process";

import path from "node:path";

import type {
  HeatPump,
  HotWaterControl,
  HotWaterData,
  WeatherData,
} from "@/domain";

import type {
  TankComparisonPoint,
} from "@/types/tank-comparison";


export interface TankComparisonWorkerInit {
  weatherData: WeatherData;
  hotWaterData: HotWaterData;

  heatPump: HeatPump;
  control: HotWaterControl;

  timeStepHours: number;
  pvProduction: number;

  gridPriceChfPerKWh: number;
  feedInPriceChfPerKWh: number;
}


export type TankComparisonWorkerRequest =
  | {
      type: "init";
      payload: TankComparisonWorkerInit;
    }
  | {
      type: "run";
      storageVolume: number;
    };


export type ApproachANativeScopes = Array<{
  scopeId: string;
  calls: number;
  nativeCalls: number;
  fallbackCalls: number;
  failures: number;
}>;

export interface ApproachAProcessPoolResult {
  points: TankComparisonPoint[];
  workerCount: number;
  nativeScopes: ApproachANativeScopes;
}

export type TankComparisonWorkerResponse =
  | {
      type: "ready";
    }
  | {
      type: "result";
      point: TankComparisonPoint;
      nativeScopes: ApproachANativeScopes;
    }
  | {
      type: "error";
      message: string;
      stack?: string;
    };


export interface TankComparisonProcessPoolInput {
  workerCount: number;

  storageVolumes: number[];

  init: TankComparisonWorkerInit;
}


export async function runTankComparisonProcessPool(
  input: TankComparisonProcessPoolInput,
): Promise<ApproachAProcessPoolResult> {

  const {
    workerCount,
    storageVolumes,
    init,
  } = input;

  if (workerCount < 1) {
    throw new Error(
      "workerCount must be at least 1.",
    );
  }

  if (storageVolumes.length === 0) {
    return { points: [], workerCount: 0, nativeScopes: [] };
  }

  const actualWorkerCount =
    Math.min(
      workerCount,
      storageVolumes.length,
    );

  const queue =
    [...storageVolumes];

  const results:
    TankComparisonPoint[] = [];

  const workers:
    ChildProcess[] = [];

  // Each worker reports cumulative counters; only its LAST snapshot counts.
  const lastNativeScopes = new Map<number, ApproachANativeScopes>();

  const workerPath =
    path.join(
      process.cwd(),
      "src",
      "workers",
      "tank-comparison-worker.ts",
    );

  return new Promise<
    ApproachAProcessPoolResult
  >((resolve, reject) => {

    let completedJobs = 0;
    let settled = false;


    const stopWorkers = () => {

      for (const worker of workers) {

        if (!worker.killed) {
          worker.kill();
        }
      }
    };


    const fail = (
      error: Error,
    ) => {

      if (settled) {
        return;
      }

      settled = true;

      stopWorkers();

      reject(error);
    };


    const finish = () => {

      if (settled) {
        return;
      }

      settled = true;

      results.sort(
        (a, b) =>
          a.storageVolume -
          b.storageVolume,
      );

      stopWorkers();

      const merged = new Map<string, ApproachANativeScopes[number]>();
      for (const scopes of lastNativeScopes.values()) {
        for (const status of scopes) {
          const prior = merged.get(status.scopeId) ?? {
            scopeId: status.scopeId,
            calls: 0,
            nativeCalls: 0,
            fallbackCalls: 0,
            failures: 0,
          };
          prior.calls += status.calls;
          prior.nativeCalls += status.nativeCalls;
          prior.fallbackCalls += status.fallbackCalls;
          prior.failures += status.failures;
          merged.set(status.scopeId, prior);
        }
      }
      resolve({
        points: results,
        workerCount: actualWorkerCount,
        nativeScopes: [...merged.values()],
      });
    };


    const dispatchNext = (
      worker: ChildProcess,
    ) => {

      const storageVolume =
        queue.shift();

      if (
        storageVolume === undefined
      ) {

        if (
          completedJobs ===
          storageVolumes.length
        ) {
          finish();
        }

        return;
      }

      worker.send({
        type: "run",
        storageVolume,
      } satisfies TankComparisonWorkerRequest);
    };


    for (
      let workerIndex = 0;
      workerIndex <
      actualWorkerCount;
      workerIndex += 1
    ) {

        const worker:
        ChildProcess =
        spawn(
          process.execPath,
          [
            "--import",
            "tsx",
            workerPath,
          ],
          {
            stdio: [
              "ignore",
              "inherit",
              "inherit",
              "ipc",
            ],

            serialization:
              "advanced",
          },
        );
        
      workers.push(worker);


      worker.on(
        "message",
        (
          message:
            TankComparisonWorkerResponse,
        ) => {

          if (
            message.type ===
            "ready"
          ) {

            dispatchNext(worker);

            return;
          }


          if (
            message.type ===
            "result"
          ) {

            results.push(
              message.point,
            );
            lastNativeScopes.set(workerIndex, message.nativeScopes);

            completedJobs += 1;

            console.log(
              "[SunOpt] Tank worker result:",
              {
                workerIndex,
                storageVolume:
                  message.point
                    .storageVolume,
                simulationDurationMs:
                  message.point
                    .simulationDurationMs
                    .toFixed(1),
                completedJobs,
                totalJobs:
                  storageVolumes.length,
              },
            );

            if (
              completedJobs ===
              storageVolumes.length
            ) {

              finish();

              return;
            }

            dispatchNext(worker);
          }


          if (
            message.type ===
            "error"
          ) {

            const error =
              new Error(
                message.message,
              );

            if (message.stack) {
              error.stack =
                message.stack;
            }

            fail(error);
          }
        },
      );


      worker.on(
        "error",
        (error) => {
          fail(error);
        },
      );


      worker.on(
        "exit",
        (code) => {

          if (
            !settled
          ) {

            fail(
              new Error(
                `Tank comparison worker exited before completing the assigned work (code ${code}).`,
              ),
            );
          }
        },
      );


      worker.send({
        type: "init",
        payload: init,
      } satisfies TankComparisonWorkerRequest);
    }
  });
}
