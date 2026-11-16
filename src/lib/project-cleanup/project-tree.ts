// src/lib/project-cleanup/project-tree.ts

import type {
  ProjectFileRecord,
  ProjectScan,
  ProjectScanResponse,
} from "./types";


function flagsFor(
  file: ProjectFileRecord,
): string {

  const flags =
    [
      file.entryPoint
        ? "E"
        : "",
      file.reachable
        ? "R"
        : "",
      file.protected
        ? "P"
        : "",
      file.candidate
        ? "C"
        : "",
      file.oldNamed
        ? "O"
        : "",
    ]
      .filter(
        Boolean,
      )
      .join(
        "",
      );


  return flags || "-";
}


export function renderLlmManifest(
  scan: ProjectScan,
): string {

  const lines:
    string[] =
    [
      `SUNOPT_PROJECT_SCAN|v=${scan.version}|scan=${scan.scanId}|files=${scan.summary.files}|entries=${scan.summary.entryPoints}|reachable=${scan.summary.reachable}|candidates=${scan.summary.candidates}`,
      "# F|id|path|flags(E entry,R reachable,P protected,C candidate,O OLD)|inbound|deps(ids)|external imports|unresolved local imports",
    ];


  for (
    const file of scan.files
  ) {

    lines.push(
      [
        "F",
        file.id,
        file.path,
        flagsFor(
          file,
        ),
        file.inboundCount,
        file.dependencyIds.join(
          ",",
        ) ||
          "-",
        file.externalImports.join(
          ",",
        ) ||
          "-",
        file.unresolvedImports.join(
          ",",
        ) ||
          "-",
      ].join(
        "|",
      ),
    );
  }


  lines.push(
    "",
    "# Cleanup analysis reply format:",
    `{"version":1,"scanId":"${scan.scanId}","actions":[{"path":"src/example.ts","action":"old","reason":"why it is obsolete"}]}`,
    "# action may be old, keep, or review. The application only renames entries explicitly marked old.",
  );


  return lines.join(
    "\n",
  );
}


export function renderCandidateManifest(
  scan: ProjectScan,
): string {

  const candidates =
    scan.files.filter(
      (
        file,
      ) =>
        file.candidate,
    );


  const lines:
    string[] =
    [
      `SUNOPT_CLEANUP_CANDIDATES|v=${scan.version}|scan=${scan.scanId}|count=${candidates.length}`,
      "# Candidate does NOT mean safe to delete. Review against dynamic/runtime conventions before returning an action.",
      "# C|id|path|inbound|deps(ids)|reasons|unresolved",
    ];


  for (
    const file of candidates
  ) {

    lines.push(
      [
        "C",
        file.id,
        file.path,
        file.inboundCount,
        file.dependencyIds.join(
          ",",
        ) ||
          "-",
        file.candidateReasons.join(
          "; ",
        ) ||
          "-",
        file.unresolvedImports.join(
          ",",
        ) ||
          "-",
      ].join(
        "|",
      ),
    );
  }


  return lines.join(
    "\n",
  );
}


interface TreeNode {
  name: string;
  path: string;
  directory: boolean;
  file: ProjectFileRecord | null;
  children: Map<
    string,
    TreeNode
  >;
}


function makeTree(
  scan: ProjectScan,
): TreeNode {

  const root:
    TreeNode = {
    name:
      scan.projectRootName,
    path:
      "",
    directory:
      true,
    file:
      null,
    children:
      new Map(),
  };


  for (
    const file of scan.files
  ) {

    const parts =
      file.path.split(
        "/",
      );


    let current =
      root;


    parts.forEach(
      (
        part,
        index,
      ) => {

        const directory =
          index <
          parts.length -
            1;


        const childPath =
          parts
            .slice(
              0,
              index + 1,
            )
            .join(
              "/",
            );


        let child =
          current.children.get(
            part,
          );


        if (
          !child
        ) {

          child = {
            name:
              part,
            path:
              childPath,
            directory,
            file:
              directory
                ? null
                : file,
            children:
              new Map(),
          };


          current.children.set(
            part,
            child,
          );
        }


        current =
          child;
      },
    );
  }


  return root;
}


function sortedChildren(
  node: TreeNode,
): TreeNode[] {

  return [
    ...node.children.values(),
  ].sort(
    (
      left,
      right,
    ) => {

      if (
        left.directory !==
        right.directory
      ) {
        return left.directory
          ? -1
          : 1;
      }


      return left.name.localeCompare(
        right.name,
      );
    },
  );
}


function fileDescription(
  file: ProjectFileRecord,
): string {

  const name =
    file.path
      .split(
        "/",
      )
      .at(
        -1,
      ) ??
    file.path;


  if (
    name ===
    "page.tsx" ||
    name ===
    "page.ts"
  ) {
    return "Next.js page entry point";
  }


  if (
    name ===
    "layout.tsx" ||
    name ===
    "layout.ts"
  ) {
    return "Next.js layout entry point";
  }


  if (
    name ===
    "route.ts" ||
    name ===
    "route.js"
  ) {
    return "Next.js API route";
  }


  if (
    name ===
    "index.ts" ||
    name ===
    "index.js"
  ) {
    return "module facade / exports";
  }


  const state =
    file.candidate
      ? "cleanup candidate"
      : file.entryPoint
        ? "entry point"
        : file.reachable
          ? "reachable module"
          : "project file";


  return `${state}; inbound ${file.inboundCount}; imports ${file.dependencyIds.length}`;
}


function iconFor(
  node: TreeNode,
): string {

  if (
    node.directory
  ) {
    return "📁";
  }


  const extension =
    node.file
      ?.extension;


  if (
    extension ===
    ".ts" ||
    extension ===
    ".tsx"
  ) {
    return "📘";
  }


  if (
    extension ===
    ".rs"
  ) {
    return "🦀";
  }


  if (
    extension ===
    ".json"
  ) {
    return "⚙️";
  }


  if (
    extension ===
    ".md"
  ) {
    return "📝";
  }


  return "📄";
}


export function renderMarkdownTree(
  scan: ProjectScan,
): string {

  const tree =
    makeTree(
      scan,
    );


  const lines:
    string[] =
    [
      "# SunOpt Project Structure",
      "",
      `Generated: ${scan.generatedAt}`,
      "",
      `Scan ID: \`${scan.scanId}\``,
      "",
      "Legend: `CANDIDATE` is only a review hint; it is never automatically deleted.",
      "",
      "```text",
      `📁 ${tree.name}/`,
    ];


  function render(
    node: TreeNode,
    prefix: string,
  ): void {

    const children =
      sortedChildren(
        node,
      );


    children.forEach(
      (
        child,
        index,
      ) => {

        const last =
          index ===
          children.length -
            1;


        const branch =
          last
            ? "└── "
            : "├── ";


        const suffix =
          child.directory
            ? "/"
            : ` — ${fileDescription(
                child.file!,
              )}${child.file?.candidate
                ? " [CANDIDATE]"
                : ""}`;


        lines.push(
          `${prefix}${branch}${iconFor(
            child,
          )} ${child.name}${suffix}`,
        );


        if (
          child.directory
        ) {

          render(
            child,
            `${prefix}${last
              ? "    "
              : "│   "}`,
          );
        }
      },
    );
  }


  render(
    tree,
    "",
  );


  lines.push(
    "```",
    "",
  );


  return lines.join(
    "\n",
  );
}


function escapeHtml(
  value: string,
): string {

  return value
    .replaceAll(
      "&",
      "&amp;",
    )
    .replaceAll(
      "<",
      "&lt;",
    )
    .replaceAll(
      ">",
      "&gt;",
    )
    .replaceAll(
      "\"",
      "&quot;",
    );
}


export function renderHtmlTree(
  scan: ProjectScan,
): string {

  const tree =
    makeTree(
      scan,
    );


  function renderNode(
    node: TreeNode,
  ): string {

    const children =
      sortedChildren(
        node,
      );


    if (
      node.directory
    ) {

      return `
        <li>
          <details open>
            <summary>${iconFor(node)} ${escapeHtml(node.name)}</summary>
            <ul>
              ${children.map(renderNode).join("\n")}
            </ul>
          </details>
        </li>
      `;
    }


    const file =
      node.file!;


    return `
      <li class="${file.candidate
        ? "candidate"
        : ""}">
        ${iconFor(node)}
        <code>${escapeHtml(node.name)}</code>
        <span class="description">${escapeHtml(fileDescription(file))}</span>
      </li>
    `;
  }


  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SunOpt Project Structure</title>
<style>
  :root { font-family: system-ui, sans-serif; color-scheme: light; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #f7f7f8; color: #171717; }
  .site-nav {
    position: sticky; top: 0; z-index: 20;
    display: flex; align-items: center; gap: .5rem;
    padding: .8rem 1rem; border-bottom: 1px solid #d4d4d8;
    background: rgba(255,255,255,.96); backdrop-filter: blur(8px);
    overflow-x: auto;
  }
  .site-nav a {
    color: #3f3f46; text-decoration: none; font-size: .9rem;
    font-weight: 650; padding: .5rem .7rem; border-radius: .5rem;
    white-space: nowrap;
  }
  .site-nav a:hover { background: #f1f1f1; color: #111; }
  .site-nav a.active { background: #18181b; color: white; }
  .site-nav .brand { margin-right: .5rem; font-weight: 800; color: #111; }
  .content { max-width: 1200px; margin: 2rem auto; padding: 0 1rem 4rem; }
  ul { list-style: none; padding-left: 1.4rem; }
  li { margin: .2rem 0; }
  summary { cursor: pointer; font-weight: 650; }
  .description { opacity: .7; margin-left: .5rem; }
  .candidate { border-left: 3px solid #b7791f; padding-left: .5rem; }
  .meta { opacity: .7; }
  code { font-size: .95em; }
</style>
</head>
<body>
<nav class="site-nav">
  <a class="brand" href="/">SunOpt</a>
  <a href="/project-status">Projektstatus</a>
  <a href="/model-comparison">Modellvergleich</a>
  <a href="/debug/hot-water">Hot-Water Debug</a>
  <a href="/debug/project-cleanup">Project Cleanup</a>
  <a class="active" href="/project-structure.html">Project Structure</a>
</nav>
<div class="content">
<h1>SunOpt Project Structure</h1>
<p class="meta">Generated ${escapeHtml(scan.generatedAt)} · Scan <code>${escapeHtml(scan.scanId)}</code></p>
<p>${scan.summary.files} files · ${scan.summary.entryPoints} entry points · ${scan.summary.reachable} reachable · ${scan.summary.candidates} cleanup candidates.</p>
<ul>
${renderNode(tree)}
</ul>
</div>
</body>
</html>`;
}


export function createProjectScanResponse(
  scan: ProjectScan,
): ProjectScanResponse {

  return {
    scan,
    llmManifest:
      renderLlmManifest(
        scan,
      ),
    candidateManifest:
      renderCandidateManifest(
        scan,
      ),
    markdownTree:
      renderMarkdownTree(
        scan,
      ),
    htmlTree:
      renderHtmlTree(
        scan,
      ),
  };
}
