// src/lib/project-cleanup/scanner.ts

import {
  createHash,
} from "node:crypto";

import {
  existsSync,
  readFileSync,
  readdirSync,
  statSync,
} from "node:fs";

import {
  basename,
  dirname,
  extname,
  join,
  normalize,
  relative,
  sep,
} from "node:path";

import type {
  ProjectFileRecord,
  ProjectScan,
} from "./types";

import {
  PROJECT_CLEANUP_VERSION,
} from "./types";


const EXCLUDED_DIRECTORIES =
  new Set([
    ".git",
    ".next",
    ".vs",
    ".idea",
    "node_modules",
    "target",
    "dist",
    "build",
    "coverage",
    ".turbo",
    ".vercel",
    ".cache",
  ]);


const GENERATED_DOCUMENTATION =
  new Set([
    "docs/project-structure.md",
    "docs/project-structure.html",
    "public/project-structure.html",
  ]);


const TEXT_EXTENSIONS =
  new Set([
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".mts",
    ".cts",
    ".json",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".rs",
    ".py",
    ".md",
    ".html",
    ".htm",
    ".toml",
    ".yaml",
    ".yml",
    ".txt",
  ]);


const SOURCE_EXTENSIONS =
  new Set([
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".mts",
    ".cts",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".rs",
    ".py",
  ]);


const MODULE_EXTENSIONS = [
  "",
  ".ts",
  ".tsx",
  ".js",
  ".jsx",
  ".mjs",
  ".cjs",
  ".mts",
  ".cts",
  ".json",
  ".css",
  ".scss",
  ".sass",
  ".less",
  ".py",
  ".rs",
];


const MAX_TEXT_FILE_BYTES =
  2 * 1024 * 1024;


/*
 * Project Cleanup is a development-only static source scanner.
 *
 * These wrappers deliberately mark filesystem access as opaque to
 * Turbopack. Without this, Next.js tries to trace the dynamic scan
 * across the whole repository and can accidentally pull native
 * sunopt-node artifacts and Visual-Studio index files into its module
 * graph.
 */
function scannerReadDirectory(
  directory: string,
) {

  return readdirSync(
    /* turbopackIgnore: true */
    directory,
    {
      withFileTypes: true,
    },
  );
}


function scannerStat(
  filePath: string,
) {

  return statSync(
    /* turbopackIgnore: true */
    filePath,
  );
}


function scannerReadFile(
  filePath: string,
): Buffer {

  return readFileSync(
    /* turbopackIgnore: true */
    filePath,
  );
}


function scannerExists(
  filePath: string,
): boolean {

  return existsSync(
    /* turbopackIgnore: true */
    filePath,
  );
}


function toProjectPath(
  root: string,
  absolutePath: string,
): string {

  return relative(
    root,
    absolutePath,
  )
    .split(sep)
    .join("/");
}


function hashBuffer(
  value: Buffer,
): string {

  return createHash(
    "sha256",
  )
    .update(
      value,
    )
    .digest(
      "hex",
    )
    .slice(
      0,
      16,
    );
}


function collectFiles(
  root: string,
): string[] {

  const result: string[] =
    [];


  function walk(
    directory: string,
  ): void {

    const entries =
      scannerReadDirectory(
        directory,
      );


    for (
      const entry of entries
    ) {

      if (
        entry.name ===
        "." ||
        entry.name ===
        ".."
      ) {
        continue;
      }


      const absolutePath =
        join(
          directory,
          entry.name,
        );


      if (
        entry.isDirectory()
      ) {

        if (
          EXCLUDED_DIRECTORIES.has(
            entry.name,
          )
        ) {
          continue;
        }


        walk(
          absolutePath,
        );

        continue;
      }


      if (
        !entry.isFile()
      ) {
        continue;
      }


      const projectPath =
        toProjectPath(
          root,
          absolutePath,
        );


      if (
        projectPath.startsWith(
          "updates/history/",
        ) ||
        projectPath.startsWith(
          "updates/inbox/",
        ) ||
        projectPath.startsWith(
          "updates/outbox/",
        )
      ) {
        continue;
      }


      if (
        GENERATED_DOCUMENTATION.has(
          projectPath,
        )
      ) {
        continue;
      }


      result.push(
        absolutePath,
      );
    }
  }


  walk(
    root,
  );


  result.sort(
    (
      left,
      right,
    ) =>
      left.localeCompare(
        right,
      ),
  );


  return result;
}


function extractImportSpecifiers(
  source: string,
  extension: string,
): string[] {

  const values =
    new Set<string>();


  function capture(
    expression: RegExp,
  ): void {

    expression.lastIndex =
      0;


    for (
      let match =
        expression.exec(
          source,
        );
      match;
      match =
        expression.exec(
          source,
        )
    ) {

      const value =
        match[1]
          ?.trim();


      if (
        value
      ) {
        values.add(
          value,
        );
      }
    }
  }


  if (
    [
      ".ts",
      ".tsx",
      ".js",
      ".jsx",
      ".mjs",
      ".cjs",
      ".mts",
      ".cts",
    ].includes(
      extension,
    )
  ) {

    /*
     * Static imports / re-exports with "from".
     * Kept deliberately conservative: false negatives become
     * unresolved/review items; the cleanup tool never deletes
     * automatically from this information.
     */
    capture(
      /\b(?:import|export)\s+(?:type\s+)?[\s\S]{0,2000}?\sfrom\s*["'`]([^"'`]+)["'`]/g,
    );


    capture(
      /\bimport\s*["'`]([^"'`]+)["'`]/g,
    );


    capture(
      /\bimport\s*\(\s*["'`]([^"'`]+)["'`]\s*\)/g,
    );


    capture(
      /\brequire\s*\(\s*["'`]([^"'`]+)["'`]\s*\)/g,
    );
  }


  if (
    [
      ".css",
      ".scss",
      ".sass",
      ".less",
    ].includes(
      extension,
    )
  ) {

    capture(
      /@import\s+(?:url\(\s*)?["']([^"']+)["']/g,
    );
  }


  if (
    extension ===
    ".py"
  ) {

    capture(
      /^\s*from\s+([A-Za-z_][A-Za-z0-9_.]*)\s+import\s+/gm,
    );


    capture(
      /^\s*import\s+([A-Za-z_][A-Za-z0-9_.]*)/gm,
    );
  }


  if (
    extension ===
    ".rs"
  ) {

    /*
     * Rust's module system is not import-path based like ESM.
     * We still record explicit `mod foo;` declarations so local
     * .rs files can participate in the dependency graph.
     */
    capture(
      /^\s*mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;/gm,
    );
  }


  return [
    ...values,
  ].sort();
}


function isNextConventionEntryPoint(
  path: string,
): boolean {

  if (
    /^src\/app\/(?:.*\/)?(?:page|layout|route|loading|error|not-found|template|default)\.(?:ts|tsx|js|jsx)$/.test(
      path,
    )
  ) {
    return true;
  }


  if (
    /^(?:src\/)?pages\/.+\.(?:ts|tsx|js|jsx)$/.test(
      path,
    )
  ) {
    return true;
  }


  return false;
}


function isEntryPoint(
  path: string,
): boolean {

  if (
    isNextConventionEntryPoint(
      path,
    )
  ) {
    return true;
  }


  if (
    /^(?:src\/)?(?:middleware|instrumentation)\.(?:ts|js)$/.test(
      path,
    )
  ) {
    return true;
  }


  if (
    /^scripts\/.+\.(?:ts|js|mjs|cjs|py)$/.test(
      path,
    )
  ) {
    return true;
  }


  if (
    /(?:^|\/)(?:[^/]+\.)?(?:test|spec)\.(?:ts|tsx|js|jsx|mjs|cjs)$/.test(
      path,
    ) ||
    path.includes(
      "/__tests__/",
    )
  ) {
    return true;
  }


  if (
    /^rust\/[^/]+\/src\/(?:lib|main)\.rs$/.test(
      path,
    ) ||
    /^rust\/[^/]+\/build\.rs$/.test(
      path,
    )
  ) {
    return true;
  }


  return false;
}


function isProtectedPath(
  path: string,
): boolean {

  if (
    isNextConventionEntryPoint(
      path,
    )
  ) {
    return true;
  }


  if (
    [
      "package.json",
      "package-lock.json",
      "pnpm-lock.yaml",
      "yarn.lock",
      "tsconfig.json",
      "jsconfig.json",
      "next-env.d.ts",
      "Cargo.toml",
      "Cargo.lock",
    ].includes(
      path,
    )
  ) {
    return true;
  }


  if (
    /^(?:next|eslint|postcss|tailwind|vitest|jest|playwright)\.config\./.test(
      path,
    )
  ) {
    return true;
  }


  if (
    /^rust\/[^/]+\/Cargo\.toml$/.test(
      path,
    ) ||
    /^rust\/[^/]+\/src\/(?:lib|main)\.rs$/.test(
      path,
    )
  ) {
    return true;
  }


  return false;
}


function isExternalSpecifier(
  specifier: string,
): boolean {

  return !(
    specifier.startsWith(
      ".",
    ) ||
    specifier.startsWith(
      "@/",
    ) ||
    specifier.startsWith(
      "/",
    )
  );
}


function resolveCandidate(
  candidateBase: string,
  knownByAbsolutePath: Map<
    string,
    string
  >,
): string | null {

  for (
    const extension of MODULE_EXTENSIONS
  ) {

    const candidate =
      normalize(
        `${candidateBase}${extension}`,
      );


    const found =
      knownByAbsolutePath.get(
        candidate,
      );


    if (
      found
    ) {
      return found;
    }
  }


  for (
    const extension of MODULE_EXTENSIONS.slice(
      1,
    )
  ) {

    const candidate =
      normalize(
        join(
          candidateBase,
          `index${extension}`,
        ),
      );


    const found =
      knownByAbsolutePath.get(
        candidate,
      );


    if (
      found
    ) {
      return found;
    }
  }


  return null;
}


function resolveLocalImport(
  root: string,
  importerPath: string,
  specifier: string,
  knownByAbsolutePath: Map<
    string,
    string
  >,
  extension: string,
): string | null {

  if (
    extension ===
    ".rs" &&
    /^[A-Za-z_][A-Za-z0-9_]*$/.test(
      specifier,
    )
  ) {

    const importerAbsolute =
      normalize(
        join(
          root,
          importerPath,
        ),
      );


    const moduleBase =
      normalize(
        join(
          dirname(
            importerAbsolute,
          ),
          specifier,
        ),
      );


    return (
      resolveCandidate(
        moduleBase,
        knownByAbsolutePath,
      ) ??
      resolveCandidate(
        normalize(
          join(
            moduleBase,
            "mod",
          ),
        ),
        knownByAbsolutePath,
      )
    );
  }


  if (
    extension ===
    ".py" &&
    isExternalSpecifier(
      specifier,
    )
  ) {

    const dotted =
      specifier.replaceAll(
        ".",
        "/",
      );


    return (
      resolveCandidate(
        normalize(
          join(
            root,
            dotted,
          ),
        ),
        knownByAbsolutePath,
      ) ??
      resolveCandidate(
        normalize(
          join(
            root,
            "src",
            dotted,
          ),
        ),
        knownByAbsolutePath,
      )
    );
  }


  if (
    isExternalSpecifier(
      specifier,
    )
  ) {
    return null;
  }


  const withoutQuery =
    specifier
      .split(
        "?",
      )[0]
      .split(
        "#",
      )[0];


  let candidateBase:
    string;


  if (
    withoutQuery.startsWith(
      "@/",
    )
  ) {

    candidateBase =
      normalize(
        join(
          root,
          "src",
          withoutQuery.slice(
            2,
          ),
        ),
      );

  } else if (
    withoutQuery.startsWith(
      "/",
    )
  ) {

    candidateBase =
      normalize(
        join(
          root,
          withoutQuery.slice(
            1,
          ),
        ),
      );

  } else {

    candidateBase =
      normalize(
        join(
          dirname(
            join(
              root,
              importerPath,
            ),
          ),
          withoutQuery,
        ),
      );
  }


  return resolveCandidate(
    candidateBase,
    knownByAbsolutePath,
  );
}


function canParseText(
  extension: string,
  sizeBytes: number,
): boolean {

  return (
    TEXT_EXTENSIONS.has(
      extension,
    ) &&
    sizeBytes <=
      MAX_TEXT_FILE_BYTES
  );
}


export function scanProject(
  projectRoot:
    string = process.cwd(),
): ProjectScan {

  const root =
    normalize(
      projectRoot,
    );


  const absoluteFiles =
    collectFiles(
      root,
    );


  const knownByAbsolutePath =
    new Map<
      string,
      string
    >();


  for (
    const absolutePath of absoluteFiles
  ) {

    knownByAbsolutePath.set(
      normalize(
        absolutePath,
      ),
      toProjectPath(
        root,
        absolutePath,
      ),
    );
  }


  type WorkingFile = {
    path: string;
    extension: string;
    sizeBytes: number;
    hash: string;
    imports: string[];
    dependencies: string[];
    externalImports: string[];
    unresolvedImports: string[];
    entryPoint: boolean;
    protected: boolean;
    oldNamed: boolean;
  };


  const workingFiles:
    WorkingFile[] =
    [];


  for (
    const absolutePath of absoluteFiles
  ) {

    const path =
      toProjectPath(
        root,
        absolutePath,
      );


    const stat =
      scannerStat(
        absolutePath,
      );


    const extension =
      extname(
        path,
      ).toLowerCase();


    const buffer =
      scannerReadFile(
        absolutePath,
      );


    const imports =
      canParseText(
        extension,
        stat.size,
      )
        ? extractImportSpecifiers(
            buffer.toString(
              "utf8",
            ),
            extension,
          )
        : [];


    const dependencies:
      string[] =
      [];


    const externalImports:
      string[] =
      [];


    const unresolvedImports:
      string[] =
      [];


    for (
      const specifier of imports
    ) {

      const resolved =
        resolveLocalImport(
          root,
          path,
          specifier,
          knownByAbsolutePath,
          extension,
        );


      if (
        resolved
      ) {

        dependencies.push(
          resolved,
        );

        continue;
      }


      if (
        isExternalSpecifier(
          specifier,
        )
      ) {

        externalImports.push(
          specifier,
        );

      } else {

        unresolvedImports.push(
          specifier,
        );
      }
    }


    workingFiles.push({
      path,
      extension,
      sizeBytes:
        stat.size,
      hash:
        hashBuffer(
          buffer,
        ),
      imports,
      dependencies:
        [
          ...new Set(
            dependencies,
          ),
        ].sort(),
      externalImports:
        [
          ...new Set(
            externalImports,
          ),
        ].sort(),
      unresolvedImports:
        [
          ...new Set(
            unresolvedImports,
          ),
        ].sort(),
      entryPoint:
        isEntryPoint(
          path,
        ),
      protected:
        isProtectedPath(
          path,
        ),
      oldNamed:
        basename(
          path,
        ).startsWith(
          "OLD",
        ),
    });
  }


  workingFiles.sort(
    (
      left,
      right,
    ) =>
      left.path.localeCompare(
        right.path,
      ),
  );


  const idByPath =
    new Map<
      string,
      number
    >();


  workingFiles.forEach(
    (
      file,
      index,
    ) => {

      idByPath.set(
        file.path,
        index + 1,
      );
    },
  );


  const inbound =
    new Map<
      string,
      number
    >();


  for (
    const file of workingFiles
  ) {

    inbound.set(
      file.path,
      0,
    );
  }


  for (
    const file of workingFiles
  ) {

    for (
      const dependency of file.dependencies
    ) {

      inbound.set(
        dependency,
        (
          inbound.get(
            dependency,
          ) ??
          0
        ) +
          1,
      );
    }
  }


  const reachable =
    new Set<string>();


  const queue =
    workingFiles
      .filter(
        (
          file,
        ) =>
          file.entryPoint,
      )
      .map(
        (
          file,
        ) =>
          file.path,
      );


  while (
    queue.length >
    0
  ) {

    const path =
      queue.shift();


    if (
      !path ||
      reachable.has(
        path,
      )
    ) {
      continue;
    }


    reachable.add(
      path,
    );


    const file =
      workingFiles.find(
        (
          candidate,
        ) =>
          candidate.path ===
          path,
      );


    if (
      !file
    ) {
      continue;
    }


    for (
      const dependency of file.dependencies
    ) {

      if (
        !reachable.has(
          dependency,
        )
      ) {
        queue.push(
          dependency,
        );
      }
    }
  }


  const records:
    ProjectFileRecord[] =
    workingFiles.map(
      (
        file,
        index,
      ) => {

        const inboundCount =
          inbound.get(
            file.path,
          ) ??
          0;


        const isReachable =
          reachable.has(
            file.path,
          );


        const candidateReasons:
          string[] =
          [];


        const sourceLike =
          SOURCE_EXTENSIONS.has(
            file.extension,
          );


        if (
          file.oldNamed
        ) {
          candidateReasons.push(
            "filename already starts with OLD",
          );
        }


        if (
          sourceLike &&
          !file.entryPoint &&
          !file.protected &&
          !isReachable
        ) {

          candidateReasons.push(
            "not reachable from a known entry point",
          );
        }


        if (
          sourceLike &&
          !file.entryPoint &&
          !file.protected &&
          inboundCount ===
            0
        ) {

          candidateReasons.push(
            "no resolved inbound dependency",
          );
        }


        const candidate =
          candidateReasons.length >
            0 &&
          !file.protected;


        return {
          id:
            index + 1,
          path:
            file.path,
          extension:
            file.extension,
          sizeBytes:
            file.sizeBytes,
          hash:
            file.hash,
          imports:
            file.imports,
          dependencyIds:
            file.dependencies
              .map(
                (
                  dependency,
                ) =>
                  idByPath.get(
                    dependency,
                  ),
              )
              .filter(
                (
                  value,
                ): value is number =>
                  value !==
                  undefined,
              ),
          externalImports:
            file.externalImports,
          unresolvedImports:
            file.unresolvedImports,
          inboundCount,
          entryPoint:
            file.entryPoint,
          protected:
            file.protected,
          reachable:
            isReachable,
          candidate,
          oldNamed:
            file.oldNamed,
          candidateReasons,
        };
      },
    );


  const scanHasher =
    createHash(
      "sha256",
    );


  for (
    const file of workingFiles
  ) {

    scanHasher.update(
      file.path,
    );

    scanHasher.update(
      "\0",
    );

    scanHasher.update(
      file.hash,
    );

    scanHasher.update(
      "\n",
    );
  }


  const scanId =
    scanHasher
      .digest(
        "hex",
      )
      .slice(
        0,
        20,
      );


  return {
    version:
      PROJECT_CLEANUP_VERSION,
    scanId,
    generatedAt:
      new Date()
        .toISOString(),
    projectRootName:
      basename(
        root,
      ),
    files:
      records,
    summary: {
      files:
        records.length,
      entryPoints:
        records.filter(
          (
            file,
          ) =>
            file.entryPoint,
        ).length,
      reachable:
        records.filter(
          (
            file,
          ) =>
            file.reachable,
        ).length,
      candidates:
        records.filter(
          (
            file,
          ) =>
            file.candidate,
        ).length,
      protectedFiles:
        records.filter(
          (
            file,
          ) =>
            file.protected,
        ).length,
      unresolvedImports:
        records.reduce(
          (
            total,
            file,
          ) =>
            total +
            file
              .unresolvedImports
              .length,
          0,
        ),
    },
  };
}


export function projectPathExists(
  projectRoot: string,
  projectPath: string,
): boolean {

  return scannerExists(
    normalize(
      join(
        projectRoot,
        projectPath,
      ),
    ),
  );
}
