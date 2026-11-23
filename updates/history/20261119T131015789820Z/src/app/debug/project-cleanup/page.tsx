// src/app/debug/project-cleanup/page.tsx

"use client";

import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import type {
  ChangeEvent,
  ReactNode,
} from "react";

import type {
  CleanupPlan,
  GitCleanupStatus,
  ProjectScanResponse,
} from "@/lib/project-cleanup/types";


interface ScanOutboxConfirmation {
  saved: true;
  directory: string;
  llmManifest: string;
  candidateManifest: string;
  jsonReport: string;
}


interface ProcessStepProps {
  done: boolean;
  button: ReactNode;
  title: string;
  description: string;
}


function StatusLed({
  on,
}: {
  on: boolean;
}) {

  return (
    <span
      aria-label={
        on
          ? "completed"
          : "pending"
      }
      title={
        on
          ? "Completed"
          : "Pending"
      }
      className="
        relative inline-flex h-7 w-7 shrink-0
        items-center justify-center rounded-full
        bg-gradient-to-br
        from-gray-100 via-gray-500 to-gray-200
        shadow-[inset_0_1px_1px_rgba(255,255,255,0.9),0_1px_2px_rgba(0,0,0,0.35)]
      "
    >
      <span
        className={[
          "h-3.5 w-3.5 rounded-full border",
          on
            ? [
                "border-green-300",
                "bg-[radial-gradient(circle_at_35%_30%,#ecfdf5_0%,#4ade80_28%,#16a34a_65%,#14532d_100%)]",
                "shadow-[0_0_9px_rgba(34,197,94,0.85)]",
              ].join(
                " ",
              )
            : [
                "border-gray-500",
                "bg-[radial-gradient(circle_at_35%_30%,#d4d4d8_0%,#71717a_38%,#3f3f46_75%,#27272a_100%)]",
                "shadow-inner",
              ].join(
                " ",
              ),
        ].join(
          " ",
        )}
      />
    </span>
  );
}


function ProcessStep({
  done,
  button,
  title,
  description,
}: ProcessStepProps) {

  return (
    <div
      className="
        grid grid-cols-[auto_minmax(180px,230px)_1fr]
        items-center gap-4
        rounded-xl border border-[var(--sunopt-border-soft)]
        bg-[var(--sunopt-card)] p-4
      "
    >
      <StatusLed
        on={
          done
        }
      />

      <div>
        {button}
      </div>

      <div>
        <div
          className="font-semibold text-[var(--sunopt-ink)]"
        >
          {title}
        </div>

        <div
          className="
            mt-1 text-sm leading-6
            text-[var(--sunopt-muted)]
          "
        >
          {description}
        </div>
      </div>
    </div>
  );
}


function ProcessInstruction({
  children,
}: {
  children: ReactNode;
}) {

  return (
    <div
      className="
        ml-3 border-l-2 border-dashed
        border-gray-300 py-3 pl-8
        text-sm leading-6 text-[var(--sunopt-muted)]
      "
    >
      {children}
    </div>
  );
}


function downloadText(
  filename: string,
  content: string,
  type:
    string = "text/plain;charset=utf-8",
): void {

  const blob =
    new Blob(
      [
        content,
      ],
      {
        type,
      },
    );


  const url =
    URL.createObjectURL(
      blob,
    );


  const anchor =
    document.createElement(
      "a",
    );


  anchor.href =
    url;

  anchor.download =
    filename;

  document.body.appendChild(
    anchor,
  );

  anchor.click();
  anchor.remove();

  URL.revokeObjectURL(
    url,
  );
}


async function readJsonResponse<T>(
  response: Response,
): Promise<T> {

  const payload =
    await response.json();


  if (
    !response.ok
  ) {

    const message =
      typeof payload?.error ===
      "string"
        ? payload.error
        : `Request failed with status ${response.status}`;


    throw new Error(
      message,
    );
  }


  return payload as T;
}


const PRIMARY_BUTTON =
  `
    w-full rounded-lg border border-gray-800
    bg-[var(--sunopt-ink)] px-4 py-2.5
    text-sm font-semibold text-white
    shadow-sm transition
    hover:bg-gray-700
    disabled:cursor-not-allowed
    disabled:border-gray-300
    disabled:bg-gray-300
  `;


const SECONDARY_BUTTON =
  `
    w-full rounded-lg border border-gray-300
    bg-[var(--sunopt-card)] px-3 py-2
    text-sm font-semibold text-gray-800
    shadow-sm transition
    hover:bg-[var(--sunopt-page)]
    disabled:cursor-not-allowed
    disabled:text-gray-400
  `;


export default function ProjectCleanupPage() {

  const [
    scanResponse,
    setScanResponse,
  ] =
    useState<
      ProjectScanResponse | null
    >(
      null,
    );


  const [
    scanOutbox,
    setScanOutbox,
  ] = useState<ScanOutboxConfirmation | null>(null);

  const [
    scanRunning,
    setScanRunning,
  ] = useState(false);


  const [
    cleanupPlan,
    setCleanupPlan,
  ] =
    useState<
      CleanupPlan | null
    >(
      null,
    );


  const [
    selectedPaths,
    setSelectedPaths,
  ] =
    useState<
      Set<string>
    >(
      new Set(),
    );


  const [
    gitStatus,
    setGitStatus,
  ] =
    useState<
      GitCleanupStatus | null
    >(
      null,
    );


  const [
    busy,
    setBusy,
  ] =
    useState(
      false,
    );


  const [
    message,
    setMessage,
  ] =
    useState<
      string | null
    >(
      null,
    );


  const [
    error,
    setError,
  ] =
    useState<
      string | null
    >(
      null,
    );


  const [
    manifestDownloaded,
    setManifestDownloaded,
  ] =
    useState(
      false,
    );


  const [
    filesDeleted,
    setFilesDeleted,
  ] =
    useState(
      false,
    );


  const [
    cleanupCommitted,
    setCleanupCommitted,
  ] =
    useState(
      false,
    );


  const fileInputRef =
    useRef<
      HTMLInputElement | null
    >(
      null,
    );


  const oldActions =
    useMemo(
      () =>
        cleanupPlan
          ?.actions
          .filter(
            (
              action,
            ) =>
              action.action ===
              "old",
          ) ??
        [],
      [
        cleanupPlan,
      ],
    );


  const renamed =
    Boolean(
      gitStatus
        ?.state
        ?.renamed
        .length,
    );


  const oldRenameCommitted =
    Boolean(
      gitStatus
        ?.renameCommitDetected &&
      gitStatus
        ?.renamedFilesCommitted,
    );


  const deletionPendingCommit =
    Boolean(
      gitStatus
        ?.deletionPendingCommit,
    );


  useEffect(
    () => {

      void refreshGitStatus(
        false,
      );
    },
    [],
  );


  async function perform(
    work:
      () => Promise<void>,
  ): Promise<void> {

    setBusy(
      true,
    );

    setError(
      null,
    );

    setMessage(
      null,
    );


    try {

      await work();

    } catch (caught) {

      setError(
        caught instanceof Error
          ? caught.message
          : String(
              caught,
            ),
      );

    } finally {

      setBusy(
        false,
      );
    }
  }


  async function fetchGitStatus():
    Promise<GitCleanupStatus> {

    const response =
      await fetch(
        "/api/debug/project-cleanup/git-status",
        {
          cache:
            "no-store",
        },
      );


    return readJsonResponse<
      GitCleanupStatus
    >(
      response,
    );
  }


  async function refreshGitStatus(
    showMessage:
      boolean = true,
  ): Promise<void> {

    try {

      const payload =
        await fetchGitStatus();


      setGitStatus(
        payload,
      );


      if (
        showMessage
      ) {

        setMessage(
          payload.readyToDelete
            ? "Git safety gate is satisfied. The committed OLD files may now be deleted."
            : "Git status refreshed.",
        );
      }

    } catch (caught) {

      if (
        showMessage
      ) {

        setError(
          caught instanceof Error
            ? caught.message
            : String(
                caught,
              ),
        );
      }
    }
  }


  async function scanProject(): Promise<void> {

    // Reset the previous green status when starting another scan.
    setScanRunning(true);
    setScanResponse(null);
    setScanOutbox(null);
    setManifestDownloaded(false);

    try {
      await perform(async () => {
        const response = await fetch(
          "/api/debug/project-cleanup/scan",
          { method: "POST", cache: "no-store" },
        );

        const payload = await readJsonResponse<
          ProjectScanResponse & { outbox?: ScanOutboxConfirmation }
        >(response);

        if (
          payload.outbox?.saved !== true ||
          !payload.outbox.llmManifest ||
          !payload.outbox.candidateManifest ||
          !payload.outbox.jsonReport
        ) {
          throw new Error(
            "Scan abgeschlossen, aber kein vollständiger Bericht in updates/outbox bestätigt.",
          );
        }

        // These updates activate the EXISTING green LED and download rim.
        setScanResponse(payload);
        setScanOutbox(payload.outbox);
        setCleanupPlan(null);
        setSelectedPaths(new Set());
        setFilesDeleted(false);
        setCleanupCommitted(false);

        setMessage(
          `Scan erfolgreich: ${payload.scan.summary.files} Dateien, ` +
          `${payload.scan.summary.candidates} Kandidaten. ` +
          `Bericht gespeichert: ${payload.outbox.llmManifest}`,
        );
      });
    } finally {
      setScanRunning(false);
    }
  }


  async function updateDocumentation(): Promise<void> {

    await perform(
      async () => {

        const response =
          await fetch(
            "/api/debug/project-cleanup/documentation",
            {
              method:
                "POST",
            },
          );


        const payload =
          await readJsonResponse<{
            ok: boolean;
            files: string[];
          }>(
            response,
          );


        setMessage(
          `Project structure updated: ${payload.files.join(
            ", ",
          )}.`,
        );
      },
    );
  }


  async function importCleanupPlan(
    event:
      ChangeEvent<HTMLInputElement>,
  ): Promise<void> {

    const file =
      event.target.files
        ?.item(
          0,
        );


    event.target.value =
      "";


    if (
      !file
    ) {
      return;
    }


    await perform(
      async () => {

        const text =
          await file.text();


        const parsed =
          JSON.parse(
            text,
          ) as
            CleanupPlan;


        if (
          parsed.version !==
          1 ||
          typeof parsed.scanId !==
          "string" ||
          !Array.isArray(
            parsed.actions,
          )
        ) {

          throw new Error(
            "This is not a valid SunOpt cleanup-analysis.json file.",
          );
        }


        if (
          scanResponse &&
          parsed.scanId !==
          scanResponse.scan.scanId
        ) {

          throw new Error(
            `Analysis scanId ${parsed.scanId} does not match the displayed scan ${scanResponse.scan.scanId}.`,
          );
        }


        setCleanupPlan(
          parsed,
        );


        setSelectedPaths(
          new Set(
            parsed.actions
              .filter(
                (
                  action,
                ) =>
                  action.action ===
                  "old",
              )
              .map(
                (
                  action,
                ) =>
                  action.path,
              ),
          ),
        );


        setMessage(
          `Cleanup analysis loaded: ${parsed.actions.length} reviewed actions.`,
        );
      },
    );
  }


  function toggleSelected(
    path: string,
  ): void {

    setSelectedPaths(
      (
        current,
      ) => {

        const next =
          new Set(
            current,
          );


        if (
          next.has(
            path,
          )
        ) {
          next.delete(
            path,
          );

        } else {
          next.add(
            path,
          );
        }


        return next;
      },
    );
  }


  async function renameSelected(): Promise<void> {

    if (
      !cleanupPlan
    ) {
      return;
    }


    await perform(
      async () => {

        const plan:
          CleanupPlan = {
          ...cleanupPlan,
          actions:
            cleanupPlan.actions.map(
              (
                action,
              ) => {

                if (
                  action.action !==
                  "old"
                ) {
                  return action;
                }


                return {
                  ...action,
                  action:
                    selectedPaths.has(
                      action.path,
                    )
                      ? "old"
                      : "keep",
                };
              },
            ),
        };


        const response =
          await fetch(
            "/api/debug/project-cleanup/apply",
            {
              method:
                "POST",
              headers: {
                "content-type":
                  "application/json",
              },
              body:
                JSON.stringify(
                  plan,
                ),
            },
          );


        const payload =
          await readJsonResponse<{
            ok: boolean;
            state: {
              renamed: Array<{
                from: string;
                to: string;
              }>;
            };
          }>(
            response,
          );


        setMessage(
          `Renamed ${payload.state.renamed.length} files to OLD*. Now test SunOpt before committing.`,
        );


        await refreshGitStatus(
          false,
        );
      },
    );
  }


  async function commitOldRename(): Promise<void> {

    await perform(
      async () => {

        const response =
          await fetch(
            "/api/debug/project-cleanup/commit-old",
            {
              method:
                "POST",
            },
          );


        await readJsonResponse<{
          ok: boolean;
          head: string;
        }>(
          response,
        );


        await refreshGitStatus(
          false,
        );


        setMessage(
          "Committed OLD-file rename: Mark obsolete project files.",
        );
      },
    );
  }


  async function deleteOldFiles(): Promise<void> {

    await perform(
      async () => {

        const response =
          await fetch(
            "/api/debug/project-cleanup/delete",
            {
              method:
                "POST",
            },
          );


        const payload =
          await readJsonResponse<{
            ok: boolean;
            deleted: string[];
          }>(
            response,
          );


        setFilesDeleted(
          payload.deleted.length >
            0,
        );


        await refreshGitStatus(
          false,
        );


        setMessage(
          `Deleted ${payload.deleted.length} committed OLD files. Final Git commit is now required.`,
        );
      },
    );
  }


  async function commitDeletion(): Promise<void> {

    await perform(
      async () => {

        const response =
          await fetch(
            "/api/debug/project-cleanup/commit-delete",
            {
              method:
                "POST",
            },
          );


        await readJsonResponse<{
          ok: boolean;
          head: string;
        }>(
          response,
        );


        setCleanupCommitted(
          true,
        );

        setFilesDeleted(
          true,
        );


        await refreshGitStatus(
          false,
        );


        setMessage(
          "Cleanup complete. Deleted obsolete project files were committed.",
        );
      },
    );
  }


  function downloadLlmManifest(): void {

    if (
      !scanResponse
    ) {
      return;
    }


    downloadText(
      "sunopt-project-manifest.llm.txt",
      scanResponse.llmManifest,
    );


    setManifestDownloaded(
      true,
    );
  }


  return (
    <main
      className="
        min-h-screen sunopt-page
        px-4 py-8 sm:px-6
      "
    >
      <div
        className="mx-auto max-w-7xl"
      >
        <header
          className="mb-7"
        >
          <div
            className="
              text-sm font-semibold uppercase
              tracking-[0.18em] text-[var(--sunopt-soft-text)]
            "
          >
            SunOpt developer tools
          </div>

          <h1
            className="
              mt-2 text-3xl font-bold
              tracking-tight text-[var(--sunopt-ink)]
            "
          >
            Project Cleanup
          </h1>

          <p
            className="
              mt-3 max-w-3xl
              leading-7 text-[var(--sunopt-muted)]
            "
          >
            Guided dependency cleanup with
            scan-ID verification, OLD-file
            quarantine and Git safety gates.
          </p>
        </header>


        <div
          className="
            grid items-start gap-6
            lg:grid-cols-[minmax(0,1fr)_310px]
          "
        >
          <section
            className="
              rounded-2xl border border-[var(--sunopt-border-soft)]
              bg-gray-100/70 p-4 shadow-sm
              sm:p-5
            "
          >
            <div
              className="
                mb-4 flex flex-wrap
                items-center justify-between gap-3
              "
            >
              <div>
                <h2
                  className="
                    text-xl font-bold
                    text-[var(--sunopt-ink)]
                  "
                >
                  Cleanup process
                </h2>

                <p
                  className="
                    mt-1 text-sm text-[var(--sunopt-muted)]
                  "
                >
                  Green LEDs show completed
                  process gates.
                </p>
              </div>

              <button
                type="button"
                disabled={
                  busy
                }
                onClick={
                  () =>
                    void refreshGitStatus()
                }
                className="
                  rounded-lg border border-gray-300
                  bg-[var(--sunopt-card)] px-3 py-2
                  text-xs font-semibold text-[var(--sunopt-muted)]
                  shadow-sm hover:bg-[var(--sunopt-page)]
                  disabled:text-gray-400
                "
              >
                Refresh Git Status
              </button>
            </div>


            <ProcessStep
              done={scanOutbox !== null}
              button={
                <button
                  type="button"
                  disabled={busy}
                  onClick={scanProject}
                  className={PRIMARY_BUTTON}
                >
                  {scanRunning
                    ? "Scan läuft ..."
                    : scanOutbox
                      ? "✓ Scan erfolgreich"
                      : "Scan Project"}
                </button>
              }
              title="Create dependency snapshot"
              description={scanOutbox
                ? `Bericht gespeichert: ${scanOutbox.llmManifest}`
                : "Reads the project tree, resolves local imports and produces a scan ID tied to this exact source state."}
            />


            <ProcessInstruction>
              When the scan is complete, use
              the flashing download on the
              right. Upload that manifest to
              ChatGPT and receive
              <code className="mx-1">
                cleanup-analysis.json
              </code>.
            </ProcessInstruction>


            <ProcessStep
              done={
                cleanupPlan !==
                null
              }
              button={
                <>
                  <input
                    ref={
                      fileInputRef
                    }
                    type="file"
                    accept=".json,application/json"
                    hidden
                    onChange={
                      importCleanupPlan
                    }
                  />

                  <button
                    type="button"
                    disabled={
                      busy
                    }
                    onClick={
                      () =>
                        fileInputRef
                          .current
                          ?.click()
                    }
                    className={
                      PRIMARY_BUTTON
                    }
                  >
                    Import analysis
                  </button>
                </>
              }
              title="Load ChatGPT review"
              description="Accepts only cleanup-analysis.json. A mismatched scan ID is rejected before any file is touched."
            />


            {cleanupPlan && (
              <div
                className="
                  my-3 ml-11 rounded-xl
                  border border-[var(--sunopt-border-soft)]
                  bg-[var(--sunopt-card)] p-4
                "
              >
                <div
                  className="
                    mb-3 text-sm font-semibold
                    text-[var(--sunopt-ink)]
                  "
                >
                  Files proposed for OLD quarantine
                </div>

                <div
                  className="
                    max-h-72 space-y-2
                    overflow-y-auto pr-1
                  "
                >
                  {oldActions.map(
                    (
                      action,
                    ) => (
                      <label
                        key={
                          action.path
                        }
                        className="
                          flex cursor-pointer
                          items-start gap-3
                          rounded-lg border
                          border-gray-100 p-2
                          hover:bg-[var(--sunopt-page)]
                        "
                      >
                        <input
                          type="checkbox"
                          className="mt-1"
                          checked={
                            selectedPaths.has(
                              action.path,
                            )
                          }
                          onChange={
                            () =>
                              toggleSelected(
                                action.path,
                              )
                          }
                        />

                        <span>
                          <code
                            className="
                              text-xs font-semibold
                              text-[var(--sunopt-ink)]
                            "
                          >
                            {action.path}
                          </code>

                          <span
                            className="
                              mt-1 block text-xs
                              leading-5 text-[var(--sunopt-soft-text)]
                            "
                          >
                            {action.reason}
                          </span>
                        </span>
                      </label>
                    ),
                  )}
                </div>
              </div>
            )}


            <ProcessInstruction>
              Review the proposed files. Untick
              anything you want to keep. The
              next step only renames selected
              files; it does not delete them.
            </ProcessInstruction>


            <ProcessStep
              done={
                renamed
              }
              button={
                <button
                  type="button"
                  disabled={
                    busy ||
                    !cleanupPlan ||
                    selectedPaths.size ===
                      0 ||
                    renamed
                  }
                  onClick={
                    renameSelected
                  }
                  className={
                    PRIMARY_BUTTON
                  }
                >
                  Rename → OLD…
                </button>
              }
              title="Quarantine obsolete files"
              description="Renames only approved files to OLDfilename.ext. Hidden imports should now fail visibly instead of silently deleting code."
            />


            <ProcessInstruction>
              <strong>
                Manual validation gate:
              </strong>
              {" "}
              open SunOpt, exercise the relevant
              pages and run
              <code className="mx-1">
                npm run build
              </code>
              . Commit only when the application
              still works.
            </ProcessInstruction>


            <ProcessStep
              done={
                oldRenameCommitted
              }
              button={
                <button
                  type="button"
                  disabled={
                    busy ||
                    !renamed ||
                    oldRenameCommitted
                  }
                  onClick={
                    commitOldRename
                  }
                  className={
                    PRIMARY_BUTTON
                  }
                >
                  Commit OLD rename
                </button>
              }
              title="Create the recovery point"
              description='Stages only this cleanup transaction and commits it as "Mark obsolete project files".'
            />


            <ProcessInstruction>
              The OLD versions now exist in Git
              history and in the committed tree.
              Only after this recovery point does
              the delete gate become available.
            </ProcessInstruction>


            <ProcessStep
              done={
                filesDeleted ||
                deletionPendingCommit
              }
              button={
                <button
                  type="button"
                  disabled={
                    busy ||
                    !gitStatus
                      ?.readyToDelete
                  }
                  onClick={
                    deleteOldFiles
                  }
                  className={
                    PRIMARY_BUTTON
                  }
                >
                  DELETE OLD files
                </button>
              }
              title="Remove committed quarantine files"
              description="Deletes only the OLD files recorded in this cleanup transaction. The deletion remains uncommitted until the final step."
            />


            <ProcessInstruction>
              At this point Git shows only the
              cleanup deletions. The final commit
              closes the transaction and clears
              the internal cleanup state.
            </ProcessInstruction>


            <ProcessStep
              done={
                cleanupCommitted
              }
              button={
                <button
                  type="button"
                  disabled={
                    busy ||
                    !deletionPendingCommit
                  }
                  onClick={
                    commitDeletion
                  }
                  className={
                    PRIMARY_BUTTON
                  }
                >
                  Commit deletion
                </button>
              }
              title="Finish cleanup"
              description='Commits the deleted OLD files as "Remove obsolete project files" and closes the active cleanup transaction.'
            />


            {gitStatus && (
              <div
                className="
                  mt-5 rounded-xl border
                  border-[var(--sunopt-border-soft)] bg-gray-950
                  p-4 font-mono text-xs
                  leading-5 text-gray-200
                "
              >
                <div>
                  branch:
                  {" "}
                  {gitStatus.branch ??
                    "-"}
                </div>

                <div>
                  HEAD:
                  {" "}
                  {gitStatus.head ??
                    "-"}
                </div>

                <div>
                  worktree clean:
                  {" "}
                  {String(
                    gitStatus
                      .worktreeClean,
                  )}
                </div>

                <div>
                  rename committed:
                  {" "}
                  {String(
                    oldRenameCommitted,
                  )}
                </div>

                <div>
                  delete gate:
                  {" "}
                  {String(
                    gitStatus
                      .readyToDelete,
                  )}
                </div>

                <div>
                  deletion waiting for commit:
                  {" "}
                  {String(
                    deletionPendingCommit,
                  )}
                </div>
              </div>
            )}
          </section>


          <aside
            className="
              rounded-2xl border border-[var(--sunopt-border-soft)]
              bg-[var(--sunopt-card)] p-4 shadow-sm
              lg:sticky lg:top-24
            "
          >
            <h2
              className="
                text-base font-bold
                text-[var(--sunopt-ink)]
              "
            >
              Downloads
            </h2>

            <p
              className="
                mt-1 text-xs leading-5
                text-[var(--sunopt-soft-text)]
              "
            >
              Compact exports from the current
              scan.
            </p>


            <div
              className={[
                "mt-5 rounded-xl p-[2px]",
                scanOutbox &&
                !manifestDownloaded
                  ? "animate-pulse bg-green-500 shadow-[0_0_18px_rgba(34,197,94,0.55)]"
                  : "bg-gray-200",
              ].join(
                " ",
              )}
            >
              <div
                className="
                  rounded-[10px] bg-[var(--sunopt-card)] p-3
                "
              >
                <div
                  className="
                    mb-2 text-xs font-bold
                    uppercase tracking-wider
                    text-[var(--sunopt-soft-text)]
                  "
                >
                  For ChatGPT
                </div>

                <button
                  type="button"
                  disabled={
                    busy ||
                    !scanResponse
                  }
                  onClick={
                    downloadLlmManifest
                  }
                  className={
                    PRIMARY_BUTTON
                  }
                >
                  Download LLM Manifest
                </button>

                {scanOutbox && (
                  <p className="mt-2 break-all text-xs font-semibold text-green-700" role="status">
                    ✓ Scan erfolgreich – Datei in {scanOutbox.directory} gespeichert.
                  </p>
                )}

                <p
                  className="
                    mt-2 text-xs leading-5
                    text-[var(--sunopt-soft-text)]
                  "
                >
                  Upload this file to ChatGPT for
                  the dependency review.
                </p>
              </div>
            </div>


            <div
              className="mt-5 space-y-2"
            >
              <button
                type="button"
                disabled={
                  busy ||
                  !scanResponse
                }
                onClick={
                  () =>
                    scanResponse &&
                    downloadText(
                      "sunopt-project-candidates.llm.txt",
                      scanResponse.candidateManifest,
                    )
                }
                className={
                  SECONDARY_BUTTON
                }
              >
                Candidates only
              </button>

              <button
                type="button"
                disabled={
                  busy ||
                  !scanResponse
                }
                onClick={
                  () =>
                    scanResponse &&
                    downloadText(
                      "project-structure.md",
                      scanResponse.markdownTree,
                      "text/markdown;charset=utf-8",
                    )
                }
                className={
                  SECONDARY_BUTTON
                }
              >
                Markdown tree
              </button>

              <button
                type="button"
                disabled={
                  busy ||
                  !scanResponse
                }
                onClick={
                  () =>
                    scanResponse &&
                    downloadText(
                      "project-structure.html",
                      scanResponse.htmlTree,
                      "text/html;charset=utf-8",
                    )
                }
                className={
                  SECONDARY_BUTTON
                }
              >
                HTML tree
              </button>

              <button
                type="button"
                disabled={
                  busy
                }
                onClick={
                  updateDocumentation
                }
                className={
                  SECONDARY_BUTTON
                }
              >
                Update Project Structure
              </button>
            </div>


            {scanResponse && (
              <div
                className="
                  mt-5 rounded-lg bg-[var(--sunopt-page)]
                  p-3 text-xs leading-5
                  text-[var(--sunopt-muted)]
                "
              >
                <div
                  className="
                    font-semibold text-[var(--sunopt-ink)]
                  "
                >
                  Scan complete
                </div>

                <div>
                  ID:
                  {" "}
                  <code>
                    {scanResponse.scan.scanId}
                  </code>
                </div>

                <div>
                  Files:
                  {" "}
                  {scanResponse
                    .scan
                    .summary
                    .files}
                </div>

                <div>
                  Candidates:
                  {" "}
                  {scanResponse
                    .scan
                    .summary
                    .candidates}
                </div>
              </div>
            )}
          </aside>
        </div>


        {message && (
          <div
            className="
              mt-6 rounded-xl border
              border-green-300 bg-green-50
              p-4 text-sm text-green-900
            "
          >
            {message}
          </div>
        )}


        {error && (
          <div
            className="
              mt-6 whitespace-pre-wrap
              rounded-xl border border-red-300
              bg-red-50 p-4
              text-sm text-red-900
            "
          >
            {error}
          </div>
        )}
      </div>
    </main>
  );
}
