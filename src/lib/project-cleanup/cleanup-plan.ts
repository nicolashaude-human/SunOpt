// src/lib/project-cleanup/cleanup-plan.ts

import {
  existsSync,
  renameSync,
  rmSync,
} from "node:fs";

import {
  basename,
  dirname,
  resolve,
  sep,
} from "node:path";

import type {
  CleanupPlan,
  CleanupState,
  RenameRecord,
} from "./types";

import {
  PROJECT_CLEANUP_VERSION,
} from "./types";

import {
  getGitCleanupStatus,
  getRepositoryHead,
  writeCleanupState,
} from "./git-safety";

import {
  scanProject,
} from "./scanner";


function assertDevelopmentOnly(): void {

  if (
    process.env.NODE_ENV ===
    "production"
  ) {

    throw new Error(
      "Project Cleanup is disabled in production.",
    );
  }
}


function assertSafeProjectPath(
  projectRoot: string,
  projectPath: string,
): string {

  if (
    !projectPath ||
    projectPath.startsWith(
      "/",
    ) ||
    projectPath.includes(
      "\0",
    )
  ) {

    throw new Error(
      `Unsafe project path: ${projectPath}`,
    );
  }


  const root =
    resolve(
      projectRoot,
    );


  const absolute =
    resolve(
      root,
      projectPath,
    );


  if (
    absolute ===
      root ||
    !absolute.startsWith(
      `${root}${sep}`,
    )
  ) {

    throw new Error(
      `Path escapes project root: ${projectPath}`,
    );
  }


  return absolute;
}


export function renameCleanupTargets(
  plan: CleanupPlan,
  projectRoot:
    string = process.cwd(),
): CleanupState {

  assertDevelopmentOnly();


  if (
    plan.version !==
    PROJECT_CLEANUP_VERSION
  ) {

    throw new Error(
      `Unsupported cleanup plan version: ${plan.version}`,
    );
  }


  const gitStatus =
    getGitCleanupStatus(
      projectRoot,
    );


  if (
    !gitStatus.repository
  ) {

    throw new Error(
      "Project Cleanup requires a Git repository.",
    );
  }


  if (
    !gitStatus.worktreeClean
  ) {

    throw new Error(
      "Git worktree must be clean before files are renamed to OLD*. Commit or stash the current changes first.",
    );
  }


  if (
    gitStatus.state
  ) {

    throw new Error(
      "An OLD-file cleanup is already active. Finish or manually clear it before starting another cleanup.",
    );
  }


  const scan =
    scanProject(
      projectRoot,
    );


  if (
    scan.scanId !==
    plan.scanId
  ) {

    throw new Error(
      `Cleanup plan scanId ${plan.scanId} does not match current project scanId ${scan.scanId}. Re-scan and re-analyse the project.`,
    );
  }


  const selected =
    plan.actions.filter(
      (
        action,
      ) =>
        action.action ===
        "old",
    );


  if (
    selected.length ===
    0
  ) {

    throw new Error(
      "Cleanup plan contains no files marked action=old.",
    );
  }


  const byPath =
    new Map(
      scan.files.map(
        (
          file,
        ) =>
          [
            file.path,
            file,
          ] as const,
      ),
    );


  const renames:
    RenameRecord[] =
    [];


  for (
    const action of selected
  ) {

    const file =
      byPath.get(
        action.path,
      );


    if (
      !file
    ) {

      throw new Error(
        `Cleanup target is not in the current scan: ${action.path}`,
      );
    }


    if (
      file.protected
    ) {

      throw new Error(
        `Refusing to rename protected project entry/configuration file: ${action.path}`,
      );
    }


    if (
      file.oldNamed
    ) {

      throw new Error(
        `File is already OLD-named: ${action.path}`,
      );
    }


    const source =
      assertSafeProjectPath(
        projectRoot,
        action.path,
      );


    if (
      !existsSync(
        source,
      )
    ) {

      throw new Error(
        `Cleanup target does not exist: ${action.path}`,
      );
    }


    const oldName =
      `OLD${basename(
        source,
      )}`;


    const destination =
      resolve(
        dirname(
          source,
        ),
        oldName,
      );


    if (
      existsSync(
        destination,
      )
    ) {

      throw new Error(
        `OLD destination already exists: ${destination}`,
      );
    }


    const destinationProjectPath =
      `${dirname(
        action.path,
      ).replaceAll(
        "\\",
        "/",
      )}/${oldName}`
        .replace(
          /^\.\//,
          "",
        );


    renames.push({
      from:
        action.path,
      to:
        destinationProjectPath,
      reason:
        action.reason,
    });
  }


  /*
   * Validation above is complete before the first filesystem mutation.
   */
  for (
    const rename of renames
  ) {

    renameSync(
      assertSafeProjectPath(
        projectRoot,
        rename.from,
      ),
      assertSafeProjectPath(
        projectRoot,
        rename.to,
      ),
    );
  }


  const state:
    CleanupState = {
    version:
      PROJECT_CLEANUP_VERSION,
    scanId:
      plan.scanId,
    createdAt:
      new Date()
        .toISOString(),
    beforeHead:
      getRepositoryHead(
        projectRoot,
      ),
    renamed:
      renames,
  };


  writeCleanupState(
    projectRoot,
    state,
  );


  return state;
}


export function deleteCommittedOldFiles(
  projectRoot:
    string = process.cwd(),
): {
  deleted: string[];
} {

  assertDevelopmentOnly();


  const status =
    getGitCleanupStatus(
      projectRoot,
    );


  if (
    !status.readyToDelete ||
    !status.state
  ) {

    throw new Error(
      `OLD files are not safe to delete yet. ${status.reasons.join(
        " ",
      )}`,
    );
  }


  const deleted:
    string[] =
    [];


  for (
    const rename of status
      .state
      .renamed
  ) {

    const absolute =
      assertSafeProjectPath(
        projectRoot,
        rename.to,
      );


    if (
      existsSync(
        absolute,
      )
    ) {

      rmSync(
        absolute,
        {
          force: false,
        },
      );


      deleted.push(
        rename.to,
      );
    }
  }


  return {
    deleted,
  };
}
