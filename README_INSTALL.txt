SunOpt Project Cleanup — Feedback UI upgrade
=============================================

ERSETZEN / ERGÄNZEN — GESAMTE DATEIEN

Entpacke dieses ZIP direkt in den SunOpt-Projektroot.

Neu / geändert:
- Project Cleanup in der globalen Navigationsleiste.
- Project Structure in der globalen Navigationsleiste.
- /project-structure.html wird aus public/project-structure.html ausgeliefert.
- Update Project Structure schreibt gleichzeitig:
    docs/project-structure.md
    docs/project-structure.html
    public/project-structure.html
- Linke Prozessbox mit Status-LEDs im Chromkragen.
- Erklärung rechts neben jedem Prozessbutton.
- Handlungsanweisungen zwischen den Prozessschritten.
- Kleine Downloadbox rechts.
- LLM Manifest steht separat oben und pulsiert grün nach erfolgreichem Scan.
- Git-Commit des OLD-Zustands ist jetzt ein Button.
- Finaler Git-Commit nach dem Löschen ist ebenfalls ein Button.

Workflow:
1. Scan Project
2. Download LLM Manifest
3. Manifest zu ChatGPT hochladen
4. cleanup-analysis.json importieren
5. OLD-Dateien kontrollieren
6. Rename -> OLD...
7. MANUELLER SICHERHEITSSCHRITT:
     SunOpt testen und npm run build ausführen
8. Commit OLD rename
9. DELETE OLD files
10. Commit deletion
11. Update Project Structure

Automatische Commit-Meldungen:
- Mark obsolete project files
- Remove obsolete project files

Wichtig:
Die Commit-Funktionen stagen nur die Dateien der aktiven Cleanup-Transaktion.
Andere Änderungen werden nicht absichtlich mit `git add .` aufgenommen.

Keine neuen npm-Pakete erforderlich.
