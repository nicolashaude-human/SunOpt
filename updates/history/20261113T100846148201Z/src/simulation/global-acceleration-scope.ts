// src/simulation/global-acceleration-scope.ts

import {
  getAllNativeScopeStatus,
  getNativeAccelerationCacheSize,
  getSunOptNativeBinding,
  tryRunNativeScope,
  type NativeScopeStatus,
  type SunOptAccelerationScopeId,
} from "./rust-acceleration-runtime";


/*
 * ================================================================
 * GLOBAL ACCELERATION SCOPE MANAGER
 * ================================================================
 *
 * The IDs below are the SAME labels that exist in:
 *
 *   src/simulation/rust-acceleration-scopes.ts
 *
 * There is no algorithm-specific TypeScript source and no
 * algorithm-specific Rust module.
 */


interface ScopeDefinition {
  exportName:
    "runNoSunshineKernel" |
    "runPeakShavingCausalKernel";
}


const SCOPES:
  Record<
    SunOptAccelerationScopeId,
    ScopeDefinition
  > = {

  rust_noSunshine: {
    exportName:
      "runNoSunshineKernel",
  },

  rust_peakShavingCausal: {
    exportName:
      "runPeakShavingCausalKernel",
  },
};


export interface GlobalAccelerationScopeStatus {
  nativeRuntimeAvailable:
    boolean;

  nativeCacheSize:
    number | null;

  scopes:
    NativeScopeStatus[];
}


class GlobalAccelerationScope {

  initialise():
    void {

    getSunOptNativeBinding();
  }


  run<
    TResult
  >(
    scopeId:
      SunOptAccelerationScopeId,

    args:
      unknown[],

    validateResult?: (
      value: unknown,
    ) => TResult,
  ): TResult | null {

    const definition =
      SCOPES[
        scopeId
      ];


    return tryRunNativeScope<TResult>({
      scopeId,

      exportName:
        definition
          .exportName,

      args,

      validateResult,
    });
  }


  status():
    GlobalAccelerationScopeStatus {

    return {
      nativeRuntimeAvailable:
        getSunOptNativeBinding() !==
        null,

      nativeCacheSize:
        getNativeAccelerationCacheSize(),

      scopes:
        getAllNativeScopeStatus(),
    };
  }


  printStatus():
    void {

    console.log(
      "[SunOpt] Global acceleration scope:",
      this.status(),
    );
  }
}


/*
 * One process-wide coordinator.
 */
export const SUNOPT_ACCELERATION_SCOPE =
  new GlobalAccelerationScope();
