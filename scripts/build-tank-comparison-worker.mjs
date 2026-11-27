import { build } from "esbuild";

await build({
  entryPoints: ["src/workers/tank-comparison-worker.ts"],
  outfile: "build/tank-comparison-worker.cjs",
  bundle: true,
  platform: "node",
  format: "cjs",
  target: "node20",
});
