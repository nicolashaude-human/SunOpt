// next.config.ts

import type {
  NextConfig,
} from "next";


const nextConfig:
  NextConfig = {

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