/*
 * Approach A research test harness: resolve the TypeScript compiler API
 * from the SunOpt project's dependencies, not the download directory.
 * No application-specific function knowledge and no production mutation.
 */
import { existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

function* ancestors(directory) {
  let current = path.resolve(directory);
  for (;;) {
    yield current;
    const parent = path.dirname(current);
    if (parent === current) break;
    current = parent;
  }
}

export function findSunOptProjectRoot(moduleUrl = import.meta.url) {
  const startingPoints = [
    process.env.SUNOPT_ROOT,
    process.cwd(),
    path.dirname(fileURLToPath(moduleUrl)),
  ].filter(Boolean);
  for (const start of startingPoints) {
    for (const dir of ancestors(start)) {
      if (existsSync(path.join(dir, 'package.json')) &&
          existsSync(path.join(dir, 'src', 'simulation', 'rust-acceleration-scopes.ts'))) {
        return dir;
      }
    }
  }
  return null;
}

export function loadProjectTypeScript(moduleUrl = import.meta.url) {
  const root = findSunOptProjectRoot(moduleUrl);
  const attempts = [];
  if (root) {
    // Resolve like a JavaScript module IN the project, even if this
    // research script was originally launched from Downloads.
    const requireFromProject = createRequire(path.join(root, 'package.json'));
    try { return requireFromProject('typescript'); }
    catch (error) { attempts.push(`${root}/node_modules: ${error.message}`); }
  }
  const requireFromScript = createRequire(moduleUrl);
  try { return requireFromScript('typescript'); }
  catch (error) { attempts.push(`script location: ${error.message}`); }

  throw new Error([
    'TypeScript compiler API not found.',
    `SunOpt project root: ${root || '(not found)'}`,
    'Run from C:\\NextJS\\SunOpt\\sunopt or set SUNOPT_ROOT.',
    'Verify the project has node_modules/typescript (npm ls typescript).',
    'Checked: ' + attempts.join(' | '),
  ].join('\\n'));
}
