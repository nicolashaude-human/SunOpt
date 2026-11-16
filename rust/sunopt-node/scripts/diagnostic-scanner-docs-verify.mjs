import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here =
  path.dirname(
    fileURLToPath(import.meta.url)
  );

const root =
  path.resolve(
    here,
    "..",
    "..",
    ".."
  );

const scanner =
  fs.readFileSync(
    path.join(
      root,
      "src/lib/project-cleanup/scanner.ts"
    ),
    "utf8"
  );

const nav =
  fs.readFileSync(
    path.join(
      root,
      "src/components/site-navigation.tsx"
    ),
    "utf8"
  );

const checks = {
  excludesVisualStudio:
    scanner.includes(
      '".vs"'
    ),

  excludesUpdateArtifacts:
    scanner.includes(
      '"updates/history/"'
    ) &&
    scanner.includes(
      '"updates/inbox/"'
    ) &&
    scanner.includes(
      '"updates/outbox/"'
    ),

  noPathResolveInScanner:
    !scanner.includes(
      "resolve("
    ),

  turbopackOpaqueFs:
    scanner.includes(
      "/* turbopackIgnore: true */"
    ),

  codePathPage:
    fs.existsSync(
      path.join(
        root,
        "src/app/documentation/code-path/page.tsx"
      )
    ),

  physicsPage:
    fs.existsSync(
      path.join(
        root,
        "src/app/documentation/physics/page.tsx"
      )
    ),

  navCodePath:
    nav.includes(
      '"/documentation/code-path"'
    ),

  navPhysics:
    nav.includes(
      '"/documentation/physics"'
    ),
};

const passed =
  Object.values(
    checks
  ).every(
    Boolean
  );

console.log(
  JSON.stringify(
    {
      stage:
        "DIAGNOSTIC_SCANNER_AND_DOCS_VERIFY",
      passed,
      checks,
    },
    null,
    2
  )
);

if (!passed) {
  process.exitCode = 1;
}
