// rust/sunopt-node/scripts/blackwell-b12-phase3c-cross-parity.mjs
//
// Build a temporary research Rust binary from CURRENT cuda_context.rs and run
// the exact installed B8 resident ABI against the synchronized B12 physical
// reference.
//
// Production source is never edited.

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
    "b12_phase3c_cross_parity_extension.rs",
  );

const generatedPath =
  path.join(
    root,
    "src",
    "bin",
    "b12_phase3c_cross_parity_generated.rs",
  );

const snapshot =
  process.argv[2]
    ? path.resolve(
        process.argv[2],
      )
    : path.join(
        root,
        "blackwell-b12-sync-blackwell.bin",
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

const deviceId =
  process.argv[4] ??
  "0";

const chunkSteps =
  process.argv[5] ??
  "4096";


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
      `Required B12 Phase-3C file missing: ${required}`,
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
    "fn sunopt_cuda_run_digital_resident_chunk_replay(",
    "pub fn run_blackwell_digital_resident_chunk(",
    "fn read_blackwell_snapshot(",
    "fn research_tank_from_packed(",
    "BLACKWELL_DYNAMIC_STRIDE",
  ]
) {
  if (
    !currentSource.includes(
      token,
    )
  ) {
    throw new Error(
      `Current cuda_context.rs is not the expected B8 generation; missing ${JSON.stringify(token)}`,
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

      if (
        line
          .trimStart()
          .startsWith(
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
    `Refusing B12 Phase-3C temporary build: found only ${removedNapiAttributes} #[napi...] attributes.`,
  );
}


const extension =
  readFileSync(
    extensionPath,
    "utf8",
  );

const mainSource =
"fn main() {\n    let args =\n        std::env::args()\n            .collect::<Vec<_>>();\n\n    let snapshot =\n        args.get(1)\n            .map(String::as_str)\n            .unwrap_or(\n                \"blackwell-b12-sync-blackwell.bin\",\n            );\n\n    let physical =\n        args.get(2)\n            .map(String::as_str)\n            .unwrap_or(\n                \"blackwell-physical-reference.bin\",\n            );\n\n    let device_id =\n        args.get(3)\n            .and_then(\n                |value| {\n                    value\n                        .parse::<i32>()\n                        .ok()\n                },\n            )\n            .unwrap_or(\n                0,\n            );\n\n    let chunk_steps =\n        args.get(4)\n            .and_then(\n                |value| {\n                    value\n                        .parse::<u32>()\n                        .ok()\n                },\n            )\n            .unwrap_or(\n                4096,\n            );\n\n    if let Err(error) =\n        current_cuda_context::\n            run_b12_phase3c_cross_parity(\n                snapshot,\n                physical,\n                device_id,\n                chunk_steps,\n            )\n    {\n        eprintln!(\n            \"B12_PHASE3C_CROSS_PARITY_ERROR={error}\"\n        );\n\n        std::process::exit(\n            1,\n        );\n    }\n}";


const generated =
  [
    "#![allow(dead_code)]",
    "#![allow(unused_imports)]",
    "#![allow(unused_parens)]",
    "#![allow(unused_variables)]",
    "",
    "// AUTO-GENERATED B12 PHASE-3C RESEARCH BINARY.",
    "// Production cuda_context.rs is copied, never edited.",
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
  `B12_SYNC_BLACKWELL=${snapshot}`,
);

console.log(
  `B12_PHYSICAL_REFERENCE=${physical}`,
);

console.log(
  `B12_DEVICE=${deviceId}`,
);

console.log(
  `B12_CHUNK_STEPS=${chunkSteps}`,
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
      "b12_phase3c_cross_parity_generated",
      "--",
      snapshot,
      physical,
      deviceId,
      chunkSteps,
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
    `B12 Phase-3C cross-parity benchmark failed with exit code ${result.status}`,
  );
}

