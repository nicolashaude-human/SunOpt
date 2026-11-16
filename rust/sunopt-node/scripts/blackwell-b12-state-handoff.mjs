// rust/sunopt-node/scripts/blackwell-b12-state-handoff.mjs
//
// B12 Phase 3A runner.
//
// Creates a temporary research Rust binary from the CURRENT cuda_context.rs.
// Only #[napi...] attribute lines are removed in the temporary copy.
// Production source files are never edited.

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

const mainSource =
"fn main() {\n    let args =\n        std::env::args()\n            .collect::<Vec<_>>();\n\n    let snapshot =\n        args.get(1)\n            .map(String::as_str)\n            .unwrap_or(\n                \"blackwell-benchmark.bin\",\n            );\n\n    let physical =\n        args.get(2)\n            .map(String::as_str)\n            .unwrap_or(\n                \"blackwell-physical-reference.bin\",\n            );\n\n    if let Err(error) =\n        current_cuda_context::\n            run_b12_state_handoff(\n                snapshot,\n                physical,\n            )\n    {\n        eprintln!(\n            \"B12_STATE_HANDOFF_ERROR={error}\"\n        );\n\n        std::process::exit(\n            1,\n        );\n    }\n}";



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
    mainSource,
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


