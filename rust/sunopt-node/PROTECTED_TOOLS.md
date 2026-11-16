# Protected SunOpt tool installation

The update watcher intentionally protects `tools/`.

Files that must live there are therefore stored under:

    rust/sunopt-node/protected/

and copied explicitly by:

    powershell -ExecutionPolicy Bypass -File rust/sunopt-node/install-protected-tools.ps1

The script verifies SHA-256 after copying.

Current hot-water simulation reference:
- 8.4 s
