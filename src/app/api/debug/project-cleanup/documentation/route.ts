// src/app/api/debug/project-cleanup/documentation/route.ts

import {
  mkdirSync,
  writeFileSync,
} from "node:fs";

import {
  dirname,
  resolve,
} from "node:path";

import {
  NextResponse,
} from "next/server";

import {
  renderHtmlTree,
  renderMarkdownTree,
} from "@/lib/project-cleanup/project-tree";

import {
  scanProject,
} from "@/lib/project-cleanup/scanner";


export const runtime =
  "nodejs";

export const dynamic =
  "force-dynamic";


export async function POST() {

  if (
    process.env.NODE_ENV ===
    "production"
  ) {

    return NextResponse.json(
      {
        error:
          "Project Cleanup is disabled in production.",
      },
      {
        status:
          403,
      },
    );
  }


  try {

    const scan =
      scanProject();


    const markdownPath =
      resolve(
        process.cwd(),
        "docs/project-structure.md",
      );


    const htmlPath =
      resolve(
        process.cwd(),
        "docs/project-structure.html",
      );


    const publicHtmlPath =
      resolve(
        process.cwd(),
        "public/project-structure.html",
      );


    mkdirSync(
      dirname(
        markdownPath,
      ),
      {
        recursive: true,
      },
    );


    mkdirSync(
      dirname(
        publicHtmlPath,
      ),
      {
        recursive: true,
      },
    );


    writeFileSync(
      markdownPath,
      renderMarkdownTree(
        scan,
      ),
      "utf8",
    );


    const html =
      renderHtmlTree(
        scan,
      );


    writeFileSync(
      htmlPath,
      html,
      "utf8",
    );


    writeFileSync(
      publicHtmlPath,
      html,
      "utf8",
    );


    return NextResponse.json({
      ok:
        true,
      scanId:
        scan.scanId,
      files: [
        "docs/project-structure.md",
        "docs/project-structure.html",
        "public/project-structure.html",
      ],
    });

  } catch (error) {

    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : String(
                error,
              ),
      },
      {
        status:
          500,
      },
    );
  }
}
