// One isolated Windows/NVML GPU profile process for each productive A/B sweep.
// Changes only GPU runtime settings; no solver or A/B selection is modified.
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { join } from "node:path";

export type SunoptGpuProfileReport = {
  status: "done";
  context: "A" | "B";
  target: { smClockMHz: number; fanPercent: number };
  targetAcceptedByDriver: boolean;
  verifiedClockUnderLoad: boolean;
  fanControlSetAccepted: boolean;
  fanTelemetryVerified: boolean;
  compliant: boolean;
  endReason: string;
  elapsedSeconds: number;
  sampleCount: number;
  underLoadSamples: number;
  underLoadSmClockMinMHz: number | null;
  underLoadSmClockMaxMHz: number | null;
  underLoadSmClockMedianMHz: number | null;
  maximumTemperatureC: number | null;
  restoreError: string | null;
  error: string | null;
  reportPath: string;
};

type GuardMessage = {
  status: "ready" | "error" | "done";
  error?: string;
  [key: string]: unknown;
};

type GuardHandle = {
  stop: () => Promise<SunoptGpuProfileReport>;
};

export async function startSunoptGpuProfileGuard(
  causalPath: "A" | "B",
): Promise<GuardHandle> {
  if (process.platform !== "win32") {
    throw new Error("GPU-Profil benötigt Windows/NVML. Kein ungeprüfter A/B-Lauf.");
  }
  const script = join(process.cwd(), "rust", "sunopt-node", "scripts", "sunopt-gpu-profile-guard.py");
  const outbox = join(process.cwd(), "updates", "outbox");
  const python = process.env.SUNOPT_GPU_GUARD_PYTHON || "python";
  const child: ChildProcessWithoutNullStreams = spawn(python, [
    script,
    "--parent-pid", String(process.pid),
    "--context", causalPath,
    "--outbox", outbox,
  ], { stdio: ["pipe", "pipe", "pipe"], windowsHide: true, shell: false });

  let buffer = "";
  let diagnostic = "";
  let ready = false;
  let done = false;
  let startupResolve!: () => void;
  let startupReject!: (reason: Error) => void;
  let reportResolve!: (report: SunoptGpuProfileReport) => void;
  let reportReject!: (reason: Error) => void;

  const started = new Promise<void>((resolve, reject) => {
    startupResolve = resolve;
    startupReject = reject;
  });
  const reported = new Promise<SunoptGpuProfileReport>((resolve, reject) => {
    reportResolve = resolve;
    reportReject = reject;
  });
  // When startup fails the report is never awaited: avoid unhandled rejection.
  void reported.catch(() => undefined);

  function fail(message: string) {
    const error = new Error(`SunOpt GPU 3480 MHz / 70% fan guard: ${message}`);
    if (!ready) startupReject(error);
    if (!done) reportReject(error);
  }

  function processMessage(message: GuardMessage) {
    if (message.status === "ready") {
      if (!ready) {
        ready = true;
        startupResolve();
      }
    } else if (message.status === "error") {
      fail(message.error || "Unbekannter NVML-Fehler");
    } else if (message.status === "done") {
      done = true;
      reportResolve(message as SunoptGpuProfileReport);
    }
  }

  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk: string) => {
    buffer += chunk;
    for (;;) {
      const pos = buffer.indexOf("\n");
      if (pos < 0) break;
      const line = buffer.slice(0, pos).trim();
      buffer = buffer.slice(pos + 1);
      if (!line) continue;
      try {
        processMessage(JSON.parse(line) as GuardMessage);
      } catch {
        fail("Guard lieferte ungültiges JSON: " + line.slice(0, 180));
      }
    }
  });
  child.stderr.setEncoding("utf8");
  child.stderr.on("data", (chunk: string) => {
    diagnostic = (diagnostic + chunk).slice(-16000);
  });
  child.on("error", (error) => fail(error.message));
  child.on("close", (exitCode) => {
    if (!ready || !done) fail(`Guard beendet (Code ${exitCode}): ${diagnostic}`);
  });

  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    await Promise.race([
      started,
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(() => reject(new Error("GPU-Guard Start-Timeout (10 s)")), 10000);
      }),
    ]);
  } catch (error) {
    // Ask the guard to restore the GPU; killing it here could strand manual
    // fan or clock settings after a timeout.
    if (!child.stdin.destroyed) child.stdin.write("STOP\n");
    throw error;
  } finally {
    if (timer) clearTimeout(timer);
  }

  return {
    async stop() {
      // Never terminate the worker immediately: it must reset fan and clock.
      if (!child.stdin.destroyed) child.stdin.write("STOP\n");
      const report = await reported;
      if (!report.compliant) {
        throw new Error(
          `GPU-Profil nicht bestätigt (${report.endReason}, ` +
          `SM ${report.underLoadSmClockMedianMHz ?? "n/a"} MHz, ` +
          `Fan 70% gesetzt: ${report.fanControlSetAccepted}). ` +
          `Details: ${report.reportPath}`,
        );
      }
      return report;
    },
  };
}
