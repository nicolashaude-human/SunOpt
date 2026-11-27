// next.config.ts

import type {
  NextConfig,
} from "next";


const nextConfig:
  NextConfig = {

  allowedDevOrigins: ["172.24.4.107"],
  outputFileTracingIncludes: {
    "/*": ["./data/cache/weather/reference-8001-2023-az0-inc30-loss14.json"],
    "/api/tank-comparison": ["./build/tank-comparison-worker.cjs"],
  },
  ...(process.env.VERCEL === "1" && {
    outputFileTracingExcludes: {
      "/*": [
        "./updates/**",
        "./rust/sunopt-core/target/**",
        "./rust/sunopt-node/target/**",
        "./rust/sunopt-node/benchmarks/**",
        "./rust/sunopt-node/blackwell-*-results/**",
        "./rust/sunopt-node/cuda-disasm/**",
        "./rust/sunopt-node/*.bin",
      ],
    },
  }),

  // ------------------------------------------------------------
  // Native SunOpt N-API binding
  //
  // Must be loaded by Node.js at runtime.
  // Turbopack must not bundle the .node binary.
  // ------------------------------------------------------------

  serverExternalPackages: [
    "sunopt-node",
  ],

};


export default nextConfig;
