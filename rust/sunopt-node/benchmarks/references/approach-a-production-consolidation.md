# Approach A production consolidation — 2026-11-13

This file records the production decision after A1–A7. It is intentionally a
consolidation record, not a new optimisation proposal.

## Production default

The normal TypeScript runtime uses the tested A1+A2 chain:

1. register the acceleration source/scope once (A1),
2. register persistent static numeric arrays once (A2),
3. execute one context call with the dynamic JS parameter object,
4. fall back to registered-source and then legacy source-based execution only
   when the faster context path is unavailable.

A1 was retained after exact parity and large isolated gains: realistic
NoSunshine 33.1458 -> 19.9109 us (1.6647x) and Causal 19.7353 -> 6.9449 us
(2.8417x).

A2 was retained selectively after exact parity: realistic Causal
8.0500 -> 7.0787 us (1.1372x); realistic NoSunshine was below the 10% decision
threshold but the same persistent context is useful and already composes with
A1.

## Removed from the production runtime

- A3 JS-object batch exports: no tested workload crossed the 10% threshold.
- A4 packed-single routing: isolated gain was real, but it is superseded by the
  packed-into utility below and is not a production default.
- A5 H1/H2/H3/H4/H6 exports: none won both realistic workloads; only H5 was
  retained as part of the packed-into utility.
- A6 environment-variable routing (`SUNOPT_A6_PACKED_INTO`): A7 showed exact
  production parity but only ~0.5% median end-to-end improvement, below the
  10% acceptance threshold.

## Retained specialised native utilities

- `runNoSunshineKernelContextBatchPacked` and
  `runPeakShavingCausalKernelContextBatchPacked`: retained only for proven
  independent call groups. Feedback-dependent sequential timesteps must not be
  batched through this interface.
- `runNoSunshineKernelContextPackedInto` and
  `runPeakShavingCausalKernelContextPackedInto`: retained as a specialised
  packed input / caller-owned output API. It is not wired into the default
  TypeScript adapter.

## A7 production acceptance result

Five fresh-process runs per variant, balanced order R,C,C,R,R,C,C,R,R,C:

- exact physics parity: PASS,
- optima match: PASS,
- request wall median: 3601.829 ms -> 3583.205 ms (1.0052x),
- API total median: 3590.953 ms -> 3572.283 ms (1.0052x),
- sweep median: 3512.487 ms -> 3495.117 ms (1.0050x).

Decision: keep the packed-into utility, but do not enable it as the production
default.
