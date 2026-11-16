SunOpt update format 2 bootstrap
================================

One-time migration from ZIP + .zip.sha256 sidecars to one self-contained ZIP.

1. Extract this bootstrap ZIP into the SunOpt project root and overwrite
   tools/update_watcher.py and tools/make_update_package.py.
2. Run:
      python tools/update_watcher.py --once
3. Verify:
      python tools/update_watcher.py --status

Future update packages contain manifest.json + manifest.sha256 internally.
No external .sha256 file is required.

The bundled first format-2 update is:
  updates/inbox/sunopt-update-20261030-presentation-v4-single.zip

It expects the current release parent:
  7e9b25bc0b8ae9566931cc735ca1c353031b7876d811df031981fc3bad98180c

After successful installation its release hash is:
  b921b7a34ae5996cec058cad16936900ee31e585e7f884e2d3bbec6a0169747f
