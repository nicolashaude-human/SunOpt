// rust/sunopt-node/scripts/blackwell-b12-state-handoff.mjs
//
// B12 Phase 3A runner.
//
// The ordinary cuda_context.rs cannot be include!()'d directly into a second
// Rust module because napi-rs keeps macro bookkeeping for #[napi] structs and
// impls. For this research binary we therefore make a TEMPORARY generated copy
// with only #[napi...] attribute lines removed. No production source is edited.

import {
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";

import {
  spawnSync,
} from "node:child_process";

import path from "node:path";

import {
  fileURLToPath,
} from "node:url";


const here =
  path.dirname(
    fileURLToPath(
      import.meta.url,
    ),
  );

const root =
  path.resolve(
    here,
    "..",
  );

const cudaContextPath =
  path.join(
    root,
    "src",
    "cuda_context.rs",
  );

const extensionPath =
  path.join(
    root,
    "research",
    "b12_state_handoff_extension.rs",
  );

const generatedPath =
  path.join(
    root,
    "src",
    "bin",
    "b12_state_handoff_generated.rs",
  );

const snapshot =
  process.argv[2]
    ? path.resolve(
        process.argv[2],
      )
    : path.join(
        root,
        "blackwell-benchmark.bin",
      );

const physical =
  process.argv[3]
    ? path.resolve(
        process.argv[3],
      )
    : path.join(
        root,
        "blackwell-physical-reference.bin",
      );


for (
  const required of [
    cudaContextPath,
    extensionPath,
    snapshot,
    physical,
  ]
) {
  if (
    !existsSync(
      required,
    )
  ) {
    throw new Error(
      `Required B12 file missing: ${required}`,
    );
  }
}


const currentSource =
  readFileSync(
    cudaContextPath,
    "utf8",
  );


for (
  const token of [
    "struct BlackwellReplaySnapshot",
    "fn read_blackwell_snapshot(",
    "BLACKWELL_DYNAMIC_STRIDE",
    "fn research_tank_from_packed(",
  ]
) {
  if (
    !currentSource.includes(
      token,
    )
  ) {
    throw new Error(
      `Current cuda_context.rs is not the expected B12/B8 generation; missing ${JSON.stringify(token)}`,
    );
  }
}


const lines =
  currentSource.split(
    /\r?\n/,
  );

let removedNapiAttributes =
  0;

const cleanedLines =
  lines.filter(
    (
      line,
    ) => {

      const trimmed =
        line.trimStart();

      if (
        trimmed.startsWith(
          "#[napi",
        )
      ) {
        removedNapiAttributes +=
          1;

        return false;
      }

      return true;
    },
  );


if (
  removedNapiAttributes <
    5
) {
  throw new Error(
    `Refusing to generate B12 source: only ${removedNapiAttributes} #[napi...] attributes were found.`,
  );
}


const extension =
  readFileSync(
    extensionPath,
    "utf8",
  );


const generated =
  [
    "// AUTO-GENERATED B12 RESEARCH BINARY.",
    "// Do not edit; blackwell-b12-state-handoff.mjs regenerates it.",
    "",
    "mod current_cuda_context {",
    cleanedLines.join(
      "\n",
    ),
    "",
    extension,
    "}",
    "",
    fn main() {
    let args =
        std::env::args()
            .collect::<Vec<_>>();

    let snapshot =
        args.get(1)
            .map(String::as_str)
            .unwrap_or(
                "blackwell-benchmark.bin",
            );

    let physical =
        args.get(2)
            .map(String::as_str)
            .unwrap_or(
                "blackwell-physical-reference.bin",
            );

    if let Err(error) =
        current_cuda_context::
            run_b12_state_handoff(
                snapshot,
                physical,
            )
    {
        eprintln!(
            "B12_STATE_HANDOFF_ERROR={error}"
        );

        std::process::exit(
            1,
        );
    }
},
    "",
  ].join(
    "\n",
  );


mkdirSync(
  path.dirname(
    generatedPath,
  ),
  {
    recursive:
      true,
  },
);


writeFileSync(
  generatedPath,
  generated,
  "utf8",
);


console.log(
  `B12_BLACKWELL_SNAPSHOT=${snapshot}`,
);

console.log(
  `B12_PHYSICAL_REFERENCE=${physical}`,
);

console.log(
  `B12_GENERATED_SOURCE=${generatedPath}`,
);

console.log(
  `B12_REMOVED_NAPI_ATTRIBUTES=${removedNapiAttributes}`,
);


const result =
  spawnSync(
    "cargo",
    [
      "run",
      "--release",
      "--bin",
      "b12_state_handoff_generated",
      "--",
      snapshot,
      physical,
    ],
    {
      cwd:
        root,

      stdio:
        "inherit",

      shell:
        false,
    },
  );


/*
 * Keep the project tree clean even after compilation.
 * Cargo's target artifact remains available for diagnostics, but the generated
 * source itself is removed.
 */
try {
  rmSync(
    generatedPath,
    {
      force:
        true,
    },
  );
} catch {
}


if (
  result.error
) {
  throw result.error;
}


if (
  result.status !==
  0
) {
  throw new Error(
    `B12 state-handoff benchmark failed with exit code ${result.status}`,
  );
}

