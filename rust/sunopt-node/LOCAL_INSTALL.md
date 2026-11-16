# Reproducible local `sunopt-node` installation

This is the first packaging-stabilisation step for the SunOpt N-API module. It replaces the previous manual copy into `node_modules/sunopt-node` with a deterministic installer.

## Install and verify

Stop the Next.js server first, then run from the SunOpt project root:

```powershell
node rust/sunopt-node/install-local-package.mjs
```

The installer checks the source package files, recreates `node_modules/sunopt-node`, copies exactly the required files, compares the SHA-256 of the source and installed `.node` binary, resolves and loads `require("sunopt-node")`, and verifies both native kernel exports.

This deliberately does not modify the root `package.json` yet. After this step is proven stable with Turbopack, it can be wired into an explicit install/postinstall script without changing the native runtime boundary.
