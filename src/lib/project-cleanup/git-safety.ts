// src/lib/project-cleanup/git-safety.ts

import {
  execFileSync,
} from "node:child_process";

import {
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";

import {
  dirname,
  resolve,
} from "node:path";

import type {
  CleanupState,
  GitCleanupStatus,
} from "./types";


function runGit(
  projectRoot: string,
  args: string[],
  allowFailure:
    boolean = false,
): string {

  try {

    return execFileSync(
      "git",
      args,
      {
        cwd:
          projectRoot,
        encoding:
          "utf8",
        stdio: [
          "ignore",
          "pipe",
          "pipe",
        ],
      },
    ).trim();

  } catch (error) {

    if (
      allowFailure
    ) {
      return "";
    }


    throw error;
  }
}


export function getRepositoryHead(
  projectRoot:
    string = process.cwd(),
): string {

  return runGit(
    projectRoot,
    [
      "rev-parse",
      "HEAD",
    ],
  );
}


export function getCleanupStatePath(
  projectRoot:
    string = process.cwd(),
): string {

  const gitDirectory =
    runGit(
      projectRoot,
      [
        "rev-parse",
        "--git-dir",
      ],
    );


  const absoluteGitDirectory =
    resolve(
      projectRoot,
      gitDirectory,
    );


  return resolve(
    absoluteGitDirectory,
    "sunopt-project-cleanup",
    "state.json",
  );
}


export function writeCleanupState(
  projectRoot: string,
  state: CleanupState,
): void {

  const statePath =
    getCleanupStatePath(
      projectRoot,
    );


  mkdirSync(
    dirname(
      statePath,
    ),
    {
      recursive: true,
    },
  );


  writeFileSync(
    statePath,
    `${JSON.stringify(
      state,
      null,
      2,
    )}\n`,
    "utf8",
  );
}


export function readCleanupState(
  projectRoot:
    string = process.cwd(),
): CleanupState | null {

  let statePath:
    string;


  try {

    statePath =
      getCleanupStatePath(
        projectRoot,
      );

  } catch {

    return null;
  }


  if (
    !existsSync(
      statePath,
    )
  ) {
    return null;
  }


  try {

    return JSON.parse(
      readFileSync(
        statePath,
        "utf8",
      ),
    ) as CleanupState;

  } catch {

    return null;
  }
}


export function clearCleanupState(
  projectRoot:
    string = process.cwd(),
): void {

  let statePath:
    string;


  try {

    statePath =
      getCleanupStatePath(
        projectRoot,
      );

  } catch {

    return;
  }


  if (
    existsSync(
      statePath,
    )
  ) {

    rmSync(
      statePath,
      {
        force: true,
      },
    );
  }
}


function pathExistsInHead(
  projectRoot: string,
  projectPath: string,
): boolean {

  return (
    runGit(
      projectRoot,
      [
        "ls-tree",
        "-r",
        "--name-only",
        "HEAD",
        "--",
        projectPath,
      ],
      true,
    ) ===
    projectPath
  );
}


function pathExistsInWorktree(
  projectRoot: string,
  projectPath: string,
): boolean {

  return existsSync(
    resolve(
      projectRoot,
      projectPath,
    ),
  );
}


function gitAddPaths(
  projectRoot: string,
  paths: string[],
): void {

  if (
    paths.length ===
    0
  ) {
    return;
  }


  runGit(
    projectRoot,
    [
      "add",
      "-A",
      "--",
      ...paths,
    ],
  );
}


function commitStaged(
  projectRoot: string,
  message: string,
): string {

  runGit(
    projectRoot,
    [
      "commit",
      "-m",
      message,
    ],
  );


  return getRepositoryHead(
    projectRoot,
  );
}


export function getGitCleanupStatus(
  projectRoot:
    string = process.cwd(),
): GitCleanupStatus {

  try {

    runGit(
      projectRoot,
      [
        "rev-parse",
        "--is-inside-work-tree",
      ],
    );

  } catch {

    return {
      repository:
        false,
      head:
        null,
      branch:
        null,
      worktreeClean:
        false,
      porcelain:
        [],
      state:
        null,
      renameCommitDetected:
        false,
      renamedFilesCommitted:
        false,
      oldFilesPresentInWorktree:
        false,
      deletionPendingCommit:
        false,
      readyToDelete:
        false,
      reasons: [
        "Project root is not a Git worktree.",
      ],
    };
  }


  const head =
    runGit(
      projectRoot,
      [
        "rev-parse",
        "HEAD",
      ],
    );


  const branch =
    runGit(
      projectRoot,
      [
        "branch",
        "--show-current",
      ],
      true,
    ) ||
    null;


  const porcelainText =
    runGit(
      projectRoot,
      [
        "status",
        "--porcelain",
      ],
      true,
    );


  const porcelain =
    porcelainText
      ? porcelainText.split(
          /\r?\n/,
        )
      : [];


  const worktreeClean =
    porcelain.length ===
    0;


  const state =
    readCleanupState(
      projectRoot,
    );


  const renameCommitDetected =
    state !==
      null &&
    head !==
      state.beforeHead;


  let renamedFilesCommitted =
    state !==
    null;


  if (
    state
  ) {

    for (
      const rename of state.renamed
    ) {

      const oldPathInHead =
        pathExistsInHead(
          projectRoot,
          rename.to,
        );


      const originalPathInHead =
        pathExistsInHead(
          projectRoot,
          rename.from,
        );


      if (
        !oldPathInHead ||
        originalPathInHead
      ) {

        renamedFilesCommitted =
          false;

        break;
      }
    }

  } else {

    renamedFilesCommitted =
      false;
  }


  const oldFilesPresentInWorktree =
    Boolean(
      state &&
      state.renamed.every(
        (
          rename,
        ) =>
          pathExistsInWorktree(
            projectRoot,
            rename.to,
          ),
      ),
    );


  const allOldFilesAbsentInWorktree =
    Boolean(
      state &&
      state.renamed.every(
        (
          rename,
        ) =>
          !pathExistsInWorktree(
            projectRoot,
            rename.to,
          ),
      ),
    );


  const deletionPendingCommit =
    Boolean(
      state &&
      renameCommitDetected &&
      renamedFilesCommitted &&
      allOldFilesAbsentInWorktree,
    );


  const reasons:
    string[] =
    [];


  if (
    !state
  ) {
    reasons.push(
      "No active OLD-file cleanup state.",
    );
  }


  if (
    state &&
    !renameCommitDetected
  ) {
    reasons.push(
      "The OLD-file rename has not been committed yet.",
    );
  }


  if (
    state &&
    renameCommitDetected &&
    !renamedFilesCommitted
  ) {
    reasons.push(
      "The current HEAD does not contain every OLD file while excluding its original path.",
    );
  }


  if (
    deletionPendingCommit
  ) {
    reasons.push(
      "OLD files were deleted and are waiting for the final Git commit.",
    );

  } else if (
    !worktreeClean
  ) {
    reasons.push(
      "Git worktree is not clean.",
    );
  }


  return {
    repository:
      true,
    head,
    branch,
    worktreeClean,
    porcelain,
    state,
    renameCommitDetected,
    renamedFilesCommitted,
    oldFilesPresentInWorktree,
    deletionPendingCommit,
    readyToDelete:
      Boolean(
        state &&
        renameCommitDetected &&
        renamedFilesCommitted &&
        oldFilesPresentInWorktree &&
        worktreeClean,
      ),
    reasons,
  };
}


export function commitOldRename(
  projectRoot:
    string = process.cwd(),
): {
  head: string;
} {

  const status =
    getGitCleanupStatus(
      projectRoot,
    );


  if (
    !status.state
  ) {

    throw new Error(
      "There is no active OLD-file cleanup to commit.",
    );
  }


  if (
    status.renameCommitDetected
  ) {

    throw new Error(
      "The OLD-file rename is already committed.",
    );
  }


  const paths =
    status.state.renamed.flatMap(
      (
        rename,
      ) => [
        rename.from,
        rename.to,
      ],
    );


  gitAddPaths(
    projectRoot,
    paths,
  );


  return {
    head:
      commitStaged(
        projectRoot,
        "Mark obsolete project files",
      ),
  };
}


export function commitOldDeletion(
  projectRoot:
    string = process.cwd(),
): {
  head: string;
} {

  const status =
    getGitCleanupStatus(
      projectRoot,
    );


  if (
    !status.state
  ) {

    throw new Error(
      "There is no active cleanup deletion to commit.",
    );
  }


  if (
    !status.deletionPendingCommit
  ) {

    throw new Error(
      "OLD files have not been deleted yet, or the OLD rename commit is missing.",
    );
  }


  const paths =
    status.state.renamed.map(
      (
        rename,
      ) =>
        rename.to,
    );


  gitAddPaths(
    projectRoot,
    paths,
  );


  const head =
    commitStaged(
      projectRoot,
      "Remove obsolete project files",
    );


  clearCleanupState(
    projectRoot,
  );


  return {
    head,
  };
}
