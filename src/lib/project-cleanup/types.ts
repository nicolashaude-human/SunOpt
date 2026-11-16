// src/lib/project-cleanup/types.ts

export const PROJECT_CLEANUP_VERSION = 1 as const;

export type CleanupAction =
  | "old"
  | "keep"
  | "review";

export interface ProjectFileRecord {
  id: number;
  path: string;
  extension: string;
  sizeBytes: number;
  hash: string;

  imports: string[];
  dependencyIds: number[];
  externalImports: string[];
  unresolvedImports: string[];

  inboundCount: number;
  entryPoint: boolean;
  protected: boolean;
  reachable: boolean;
  candidate: boolean;
  oldNamed: boolean;

  candidateReasons: string[];
}

export interface ProjectScanSummary {
  files: number;
  entryPoints: number;
  reachable: number;
  candidates: number;
  protectedFiles: number;
  unresolvedImports: number;
}

export interface ProjectScan {
  version: typeof PROJECT_CLEANUP_VERSION;
  scanId: string;
  generatedAt: string;
  projectRootName: string;
  files: ProjectFileRecord[];
  summary: ProjectScanSummary;
}

export interface ProjectScanResponse {
  scan: ProjectScan;
  llmManifest: string;
  candidateManifest: string;
  markdownTree: string;
  htmlTree: string;
}

export interface CleanupPlanAction {
  path: string;
  action: CleanupAction;
  reason: string;
}

export interface CleanupPlan {
  version: typeof PROJECT_CLEANUP_VERSION;
  scanId: string;
  actions: CleanupPlanAction[];
}

export interface RenameRecord {
  from: string;
  to: string;
  reason: string;
}

export interface CleanupState {
  version: typeof PROJECT_CLEANUP_VERSION;
  scanId: string;
  createdAt: string;
  beforeHead: string;
  renamed: RenameRecord[];
}

export interface GitCleanupStatus {
  repository: boolean;
  head: string | null;
  branch: string | null;
  worktreeClean: boolean;
  porcelain: string[];
  state: CleanupState | null;
  renameCommitDetected: boolean;
  renamedFilesCommitted: boolean;
  oldFilesPresentInWorktree: boolean;
  deletionPendingCommit: boolean;
  readyToDelete: boolean;
  reasons: string[];
}
