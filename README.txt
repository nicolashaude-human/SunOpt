SunOpt inbox/outbox watcher bootstrap
======================================

This is a MANUAL bootstrap because tools/update_watcher.py is intentionally
protected from normal format-2 update packages.

1. Extract this ZIP.
2. In PowerShell run:
     .\INSTALL_OUTBOX_WATCHER.ps1

Normal inbox use remains:
  python .\tools\update_watcher.py --once
  python .\tools\update_watcher.py --watch

New outbox capture:
  python .\tools\update_watcher.py --capture "COMMAND" --cwd rust\sunopt-node

Example:
  python .\tools\update_watcher.py --capture "npm run build && node .\scripts\blackwell-digital-lockstep-gpu-fallback.mjs .\blackwell-benchmark.bin 1 0" --cwd rust\sunopt-node

Output is shown live and written to:
  updates\outbox\llm.txt

By default each --capture replaces llm.txt so it stays small and easy to upload.
Use --append-outbox if you deliberately want to append multiple captures.

ZIP update packages never execute commands automatically. --capture only runs
a command explicitly typed by the local user.
