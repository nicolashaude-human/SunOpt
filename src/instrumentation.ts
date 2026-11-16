// src/instrumentation.ts

export async function register(): Promise<void> {
  if (process.env.NEXT_RUNTIME !== "nodejs") {
    return;
  }

  /*
   * Avoid external I/O while next build is assembling the application.
   * The preload runs when the actual Node server instance starts.
   */
  if (process.env.NEXT_PHASE === "phase-production-build") {
    return;
  }

  const {
    initializeReferenceWeatherCache,
  } = await import("./server/reference-weather-cache");

  await initializeReferenceWeatherCache();
}
