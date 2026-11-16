// src/simulation/rust-acceleration-runtime.ts

import {
  readFileSync,
} from "node:fs";

import {
  join,
} from "node:path";


/*
 * ================================================================
 * GLOBAL SUNOPT NATIVE RUNTIME
 * ================================================================
 *
 * ONE TypeScript source contains every accelerated hot loop:
 *
 *   src/simulation/rust-acceleration-scopes.ts
 *
 * Every native execution receives:
 *
 *   source + scopeId + numerical arguments
 *
 * The native parser therefore sees the same source file for every
 * acceleration job and selects the requested rust_* scope by ID.
 */


export type SunOptAccelerationScopeId =
  | "rust_noSunshine"
  | "rust_peakShavingCausal";


export type SunOptNativeFunction =
  (...args: any[]) => unknown;


export interface SunOptNativeBinding {
  runNoSunshineKernel?:
    SunOptNativeFunction;

  runPeakShavingCausalKernel?:
    SunOptNativeFunction;

  /*
   * Registered-source production hot path (validated by Approach A1).
   *
   * The original source-based exports stay available as backwards-compatible
   * reference functions.
   */
  registerAccelerationScope?:
    SunOptNativeFunction;

  runNoSunshineKernelRegistered?:
    SunOptNativeFunction;

  runPeakShavingCausalKernelRegistered?:
    SunOptNativeFunction;

  /*
   * Persistent static-data context (validated by Approach A2).
   */
  registerAccelerationDataPair?:
    SunOptNativeFunction;

  runNoSunshineKernelContext?:
    SunOptNativeFunction;

  runPeakShavingCausalKernelContext?:
    SunOptNativeFunction;

  clearAccelerationScopeCache?:
    SunOptNativeFunction;

  getAccelerationScopeCacheSize?:
    SunOptNativeFunction;

  /*
   * Backwards-compatible aliases from the original 7-day bridge.
   */
  clearNoSunshineKernelCache?:
    SunOptNativeFunction;

  getNoSunshineKernelCacheSize?:
    SunOptNativeFunction;

  __napiBindingTarget?:
    string;

  [key: string]:
    unknown;
}


export interface NativeScopeStatus {
  scopeId:
    SunOptAccelerationScopeId;

  available:
    boolean;

  disabled:
    boolean;

  calls:
    number;

  nativeCalls:
    number;

  fallbackCalls:
    number;

  failures:
    number;

  nativeDurationMs:
    number;

  lastError:
    string | null;
}


interface MutableNativeScopeStatus
  extends NativeScopeStatus {
}


type RuntimeRequire =
  (
    specifier:
      string,
  ) => unknown;


let runtimeRequire:
  RuntimeRequire | null =
  null;


function getRuntimeRequire():
  RuntimeRequire {

  if (
    runtimeRequire !==
    null
  ) {

    return runtimeRequire;
  }


  /*
   * Next/Turbopack must not see a syntactic require(dynamicAbsolutePath).
   *
   * Node 24 exposes process.getBuiltinModule(). We obtain createRequire
   * through that runtime API so the native .node load remains a genuine
   * Node runtime operation and is not rewritten to /ROOT/... by Turbopack.
   */
  const processWithBuiltinModule =
    process as typeof process & {
      getBuiltinModule?: (
        name:
          string,
      ) => {
        createRequire?: (
          filename:
            string,
        ) => RuntimeRequire;
      };
    };


  const moduleBuiltin =
    processWithBuiltinModule
      .getBuiltinModule?.(
        "module",
      );


  const createRuntimeRequire =
    moduleBuiltin
      ?.createRequire;


  if (
    typeof createRuntimeRequire !==
    "function"
  ) {

    throw new Error(
      "Node process.getBuiltinModule('module').createRequire is unavailable.",
    );
  }


  runtimeRequire =
    createRuntimeRequire(
      join(
        process.cwd(),
        "package.json",
      ),
    );


  return runtimeRequire;
}


const ACCELERATION_SOURCE_PATH =
  join(
    process.cwd(),
    "src",
    "simulation",
    "rust-acceleration-scopes.ts",
  );


const LOCAL_NATIVE_BINDING_PATH =
  join(
    process.cwd(),
    "rust",
    "sunopt-node",
    "sunopt_node.win32-x64-msvc.node",
  );


let accelerationSource:
  string | null =
  null;


let nativeBinding:
  SunOptNativeBinding | null =
  null;


let nativeBindingLoadAttempted =
  false;


let nativeBindingLoadError:
  Error | null =
  null;


let nativeRuntimeReported =
  false;


const scopeStatus =
  new Map<
    SunOptAccelerationScopeId,
    MutableNativeScopeStatus
  >();

/*
 * Consolidated production path:
 *
 * A scope source is registered once in native code. Subsequent hot-path calls
 * carry only this small handle plus the numerical arguments.
 */
const registeredScopeHandles =
  new Map<
    SunOptAccelerationScopeId,
    number
  >();

let registeredDataHandles =
  new WeakMap<
    object,
    WeakMap<
      object,
      number
    >
  >();


function errorToString(
  error: unknown,
): string {

  if (
    error instanceof Error
  ) {

    return (
      error.stack ??
      error.message
    );
  }


  return String(
    error,
  );
}


export function getAccelerationScopeSource():
  string {

  if (
    accelerationSource !==
    null
  ) {

    return accelerationSource;
  }


  accelerationSource =
    readFileSync(
      ACCELERATION_SOURCE_PATH,
      "utf8",
    );


  return accelerationSource;
}


export function getAccelerationScopeSourcePath():
  string {

  return ACCELERATION_SOURCE_PATH;
}


function getOrCreateScopeStatus(
  scopeId:
    SunOptAccelerationScopeId,
): MutableNativeScopeStatus {

  const existing =
    scopeStatus.get(
      scopeId,
    );


  if (existing) {

    return existing;
  }


  const created:
    MutableNativeScopeStatus = {

    scopeId,

    available:
      false,

    disabled:
      false,

    calls:
      0,

    nativeCalls:
      0,

    fallbackCalls:
      0,

    failures:
      0,

    nativeDurationMs:
      0,

    lastError:
      null,
  };


  scopeStatus.set(
    scopeId,
    created,
  );


  return created;
}


export function getSunOptNativeBinding():
  SunOptNativeBinding | null {

  if (
    nativeBindingLoadAttempted
  ) {

    return nativeBinding;
  }


  nativeBindingLoadAttempted =
    true;


  let localError:
    unknown =
    null;

  let packageError:
    unknown =
    null;


  /*
   * npm "file:" dependencies may leave node_modules/sunopt-node behind the
   * freshly rebuilt local native addon.  The local project .node is therefore
   * the source of truth during SunOpt development and benchmarking.
   */
  try {

    nativeBinding =
      getRuntimeRequire()(
        LOCAL_NATIVE_BINDING_PATH,
      ) as SunOptNativeBinding;

  } catch (error) {

    localError =
      error;

    nativeBinding =
      null;
  }


  if (
    nativeBinding ===
    null
  ) {

    try {

      nativeBinding =
        getRuntimeRequire()(
          "sunopt-node",
        ) as SunOptNativeBinding;

    } catch (error) {

      packageError =
        error;

      nativeBinding =
        null;
    }
  }


  if (
    nativeBinding !==
    null
  ) {

    nativeBindingLoadError =
      null;


    if (
      !nativeRuntimeReported
    ) {

      nativeRuntimeReported =
        true;


      console.log(
        "[SunOpt] Global Oxc/Cranelift acceleration runtime active.",
        {
          target:
            nativeBinding
              .__napiBindingTarget ??
            "unknown",

          bindingSource:
            localError ===
              null
              ? LOCAL_NATIVE_BINDING_PATH
              : "sunopt-node package fallback",

          source:
            ACCELERATION_SOURCE_PATH,
        },
      );
    }


    return nativeBinding;
  }


  const diagnostic =
    [
      "Local native binding failed:",
      errorToString(
        localError,
      ),
      "",
      "Package binding failed:",
      errorToString(
        packageError,
      ),
    ].join(
      "\n",
    );


  nativeBindingLoadError =
    new Error(
      diagnostic,
    );


  console.error(
    "[SunOpt] Native acceleration runtime unavailable. TypeScript scope fallbacks remain active.",
    nativeBindingLoadError,
  );


  return null;
}

function registeredExportName(
  exportName:
    "runNoSunshineKernel" |
    "runPeakShavingCausalKernel",
):
  "runNoSunshineKernelRegistered" |
  "runPeakShavingCausalKernelRegistered" {

  return exportName ===
    "runNoSunshineKernel"
      ? "runNoSunshineKernelRegistered"
      : "runPeakShavingCausalKernelRegistered";
}


function contextExportName(
  exportName:
    "runNoSunshineKernel" |
    "runPeakShavingCausalKernel",
):
  "runNoSunshineKernelContext" |
  "runPeakShavingCausalKernelContext" {

  return exportName ===
    "runNoSunshineKernel"
      ? "runNoSunshineKernelContext"
      : "runPeakShavingCausalKernelContext";
}


function getOrRegisterNativeDataHandle(
  binding:
    SunOptNativeBinding,

  first:
    unknown,

  second:
    unknown,
): number | null {

  if (
    first ===
      null ||
    second ===
      null ||
    typeof first !==
      "object" ||
    typeof second !==
      "object"
  ) {

    return null;
  }


  const register =
    binding
      .registerAccelerationDataPair;


  if (
    typeof register !==
    "function"
  ) {

    return null;
  }


  let bySecond =
    registeredDataHandles.get(
      first,
    );


  if (
    bySecond ===
    undefined
  ) {

    bySecond =
      new WeakMap<
        object,
        number
      >();

    registeredDataHandles.set(
      first,
      bySecond,
    );
  }


  const existing =
    bySecond.get(
      second,
    );


  if (
    existing !==
    undefined
  ) {

    return existing;
  }


  const rawHandle =
    register(
      first,
      second,
    );


  if (
    typeof rawHandle !==
      "number" ||
    !Number.isInteger(
      rawHandle,
    ) ||
    rawHandle <=
      0
  ) {

    throw new Error(
      `Native registerAccelerationDataPair returned invalid handle: ${String(rawHandle)}`,
    );
  }


  bySecond.set(
    second,
    rawHandle,
  );


  return rawHandle;
}


function getOrRegisterNativeScopeHandle(
  binding:
    SunOptNativeBinding,

  scopeId:
    SunOptAccelerationScopeId,
): number | null {

  const existing =
    registeredScopeHandles.get(
      scopeId,
    );


  if (
    existing !==
    undefined
  ) {

    return existing;
  }


  const register =
    binding
      .registerAccelerationScope;


  if (
    typeof register !==
    "function"
  ) {

    return null;
  }


  const rawHandle =
    register(
      getAccelerationScopeSource(),
      scopeId,
    );


  if (
    typeof rawHandle !==
      "number" ||
    !Number.isInteger(
      rawHandle,
    ) ||
    rawHandle <=
      0
  ) {

    throw new Error(
      `Native registerAccelerationScope returned invalid handle for "${scopeId}": ${String(rawHandle)}`,
    );
  }


  registeredScopeHandles.set(
    scopeId,
    rawHandle,
  );


  return rawHandle;
}


export interface TryRunNativeScopeOptions<
  TResult
> {
  scopeId:
    SunOptAccelerationScopeId;

  exportName:
    "runNoSunshineKernel" |
    "runPeakShavingCausalKernel";

  /*
   * Numerical arguments only.
   *
   * The runtime prepends the SINGLE global TypeScript source and
   * the requested rust_* scope ID automatically.
   */
  args:
    unknown[];

  validateResult?: (
    value: unknown,
  ) => TResult;

  retryAfterFailure?:
    boolean;
}


export function tryRunNativeScope<
  TResult
>(
  options:
    TryRunNativeScopeOptions<TResult>,
): TResult | null {

  const status =
    getOrCreateScopeStatus(
      options.scopeId,
    );


  status.calls +=
    1;


  if (
    status.disabled
  ) {

    status.fallbackCalls +=
      1;

    return null;
  }


  const binding =
    getSunOptNativeBinding();


  if (
    binding ===
    null
  ) {

    status.available =
      false;

    status.fallbackCalls +=
      1;

    return null;
  }


  const legacyNativeFunction =
    binding[
      options.exportName
    ];

  const registeredName =
    registeredExportName(
      options.exportName,
    );

  const registeredNativeFunction =
    binding[
      registeredName
    ];


  if (
    typeof legacyNativeFunction !==
      "function" &&
    typeof registeredNativeFunction !==
      "function"
  ) {

    status.available =
      false;

    status.fallbackCalls +=
      1;

    return null;
  }


  status.available =
    true;


  const start =
    performance.now();


  try {

    const handle =
      typeof registeredNativeFunction ===
        "function"
        ? getOrRegisterNativeScopeHandle(
            binding,
            options.scopeId,
          )
        : null;


    const contextName =
      contextExportName(
        options.exportName,
      );

    const contextNativeFunction =
      binding[
        contextName
      ];

    const dataHandle =
      handle !==
        null &&
      typeof contextNativeFunction ===
        "function" &&
      options.args.length >=
        3
        ? getOrRegisterNativeDataHandle(
            binding,
            options.args[0],
            options.args[1],
          )
        : null;


    const rawResult =
      handle !==
        null &&
      dataHandle !==
        null &&
      typeof contextNativeFunction ===
        "function"
        ? contextNativeFunction(
            handle,
            dataHandle,
            ...options.args.slice(
              2,
            ),
          )
        : (
            handle !==
              null &&
            typeof registeredNativeFunction ===
              "function"
              ? registeredNativeFunction(
                  handle,
                  ...options.args,
                )
              : (
                  legacyNativeFunction as
                    SunOptNativeFunction
                )(
                  getAccelerationScopeSource(),
                  options.scopeId,
                  ...options.args,
                )
          );


    const result =
      options.validateResult
        ? options.validateResult(
            rawResult,
          )
        : rawResult as TResult;


    status.nativeCalls +=
      1;

    status.nativeDurationMs +=
      performance.now() -
      start;


    return result;

  } catch (error) {

    status.nativeDurationMs +=
      performance.now() -
      start;

    status.failures +=
      1;

    status.lastError =
      errorToString(
        error,
      );


    if (
      options.retryAfterFailure !==
      true
    ) {

      status.disabled =
        true;
    }


    status.fallbackCalls +=
      1;


    console.error(
      `[SunOpt] Native acceleration scope "${options.scopeId}" failed. Using TypeScript scope fallback.`,
      error,
    );


    return null;
  }
}


export function getNativeScopeStatus(
  scopeId:
    SunOptAccelerationScopeId,
): NativeScopeStatus {

  return {
    ...getOrCreateScopeStatus(
      scopeId,
    ),
  };
}


export function getAllNativeScopeStatus():
  NativeScopeStatus[] {

  return Array.from(
    scopeStatus.values(),
    (status) => ({
      ...status,
    }),
  );
}


export function getNativeAccelerationCacheSize():
  number | null {

  const binding =
    getSunOptNativeBinding();


  if (
    binding ===
    null
  ) {

    return null;
  }


  const functionValue =
    binding
      .getAccelerationScopeCacheSize ??
    binding
      .getNoSunshineKernelCacheSize;


  if (
    typeof functionValue !==
    "function"
  ) {

    return null;
  }


  const result =
    functionValue();


  return typeof result ===
    "number"
      ? result
      : null;
}


export function clearNativeAccelerationCache():
  void {

  registeredScopeHandles.clear();

  registeredDataHandles =
    new WeakMap<
      object,
      WeakMap<
        object,
        number
      >
    >();


  const binding =
    getSunOptNativeBinding();


  if (
    binding ===
    null
  ) {

    return;
  }


  const functionValue =
    binding
      .clearAccelerationScopeCache ??
    binding
      .clearNoSunshineKernelCache;


  if (
    typeof functionValue ===
    "function"
  ) {

    functionValue();
  }
}
