// rust/sunopt-node/scripts/blackwell-b12-digital-sync.mjs
//
// B12 Phase 3B:
// 1. convert synchronized B12 coupled reference into the current Blackwell
//    replay format using the CURRENT Rust BlackwellCapture implementation;
// 2. run the already-installed B8 digital resident-chunk benchmark on it.
//
// Production files are never modified.

import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
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
    "b12_sync_blackwell_extension.rs",
  );

const generatedPath =
  path.join(
    root,
    "src",
    "bin",
    "b12_sync_blackwell_generated.rs",
  );

const coupledPath =
  process.argv[2]
    ? path.resolve(
        process.argv[2],
      )
    : path.join(
        root,
        "blackwell-b12-coupled-reference.bin",
      );

const synchronizedSnapshotPath =
  process.argv[3]
    ? path.resolve(
        process.argv[3],
      )
    : path.join(
        root,
        "blackwell-b12-sync-blackwell.bin",
      );

/*
 * Any arguments after the output snapshot are forwarded unchanged to the
 * installed resident-chunk benchmark. With none supplied, that script uses
 * its own known-good defaults from B8.
 */
const residentExtraArgs =
  process.argv.slice(
    4,
  );


for (
  const required of [
    cudaContextPath,
    extensionPath,
    coupledPath,
  ]
) {
  if (
    !existsSync(
      required,
    )
  ) {
    throw new Error(
      `Required B12 Phase-3B file missing: ${required}`,
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
    "struct BlackwellCapture",
    "impl BlackwellCapture",
    "fn blackwell_parameters_from_packed(",
    "BLACKWELL_CONFIG_STRIDE",
    "BLACKWELL_DYNAMIC_STRIDE",
    "pub fn run_blackwell_digital_resident_chunk(",
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


/*
 * Find the installed B8 runner without hard-coding historical package names.
 */
const scriptNames =
  readdirSync(
    here,
  );

const preferredNames = [
  "blackwell-digital-resident-chunk.mjs",
  "blackwell-resident-chunk.mjs",
];

let residentScript =
  null;

for (
  const name of preferredNames
) {
  if (
    scriptNames.includes(
      name,
    )
  ) {
    residentScript =
      path.join(
        here,
        name,
      );

    break;
  }
}

if (
  residentScript ===
  null
) {
  const discovered =
    scriptNames.find(
      (
        name,
      ) =>
        /digital.*resident.*chunk.*\.mjs$/i.test(
          name,
        ) &&
        name !==
          path.basename(
            fileURLToPath(
              import.meta.url,
            ),
          ),
    );

  if (
    discovered
  ) {
    residentScript =
      path.join(
        here,
        discovered,
      );
  }
}


if (
  residentScript ===
  null
) {
  throw new Error(
    "Installed B8 digital resident-chunk runner was not found in rust/sunopt-node/scripts.",
  );
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
    `Refusing B12 temporary build: found only ${removedNapiAttributes} #[napi...] attributes.`,
  );
}


const extension =
  readFileSync(
    extensionPath,
    "utf8",
  );

const mainSource =
"fn main() {\n    let args =\n        std::env::args()\n            .collect::<Vec<_>>();\n\n    let coupled =\n        args.get(1)\n            .map(String::as_str)\n            .unwrap_or(\n                \"blackwell-b12-coupled-reference.bin\",\n            );\n\n    let output =\n        args.get(2)\n            .map(String::as_str)\n            .unwrap_or(\n                \"blackwell-b12-sync-blackwell.bin\",\n            );\n\n    if let Err(error) =\n        current_cuda_context::\n            run_b12_sync_blackwell(\n                coupled,\n                output,\n            )\n    {\n        eprintln!(\n            \"B12_SYNC_BLACKWELL_ERROR={error}\"\n        );\n\n        std::process::exit(\n            1,\n        );\n    }\n}";


const generated =
  [
    "#![allow(dead_code)]",
    "#![allow(unused_imports)]",
    "#![allow(unused_parens)]",
    "#![allow(unused_variables)]",
    "",
    "// AUTO-GENERATED B12 PHASE-3B RESEARCH BINARY.",
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
  `B12_COUPLED_REFERENCE=${coupledPath}`,
);

console.log(
  `B12_SYNC_BLACKWELL=${synchronizedSnapshotPath}`,
);

console.log(
  `B12_CURRENT_CUDA_CONTEXT=${cudaContextPath}`,
);

console.log(
  `B12_REMOVED_NAPI_ATTRIBUTES=${removedNapiAttributes}`,
);

console.log(
  `B12_RESIDENT_RUNNER=${residentScript}`,
);


const buildResult =
  spawnSync(
    "cargo",
    [
      "run",
      "--release",
      "--bin",
      "b12_sync_blackwell_generated",
      "--",
      coupledPath,
      synchronizedSnapshotPath,
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
  buildResult.error
) {
  throw buildResult.error;
}


if (
  buildResult.status !==
  0
) {
  throw new Error(
    `B12 synchronized Blackwell conversion failed with exit code ${buildResult.status}`,
  );
}


console.log();
console.log(
  "--- B12 PHASE 3B: EXISTING B8 RESIDENT SOLVER ON SYNCHRONIZED DATA ---",
);


const residentResult =
  spawnSync(
    process.execPath,
    [
      residentScript,
      synchronizedSnapshotPath,
      ...residentExtraArgs,
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


if (
  residentResult.error
) {
  throw residentResult.error;
}


if (
  residentResult.status !==
  0
) {
  throw new Error(
    `Existing B8 resident-chunk benchmark failed with exit code ${residentResult.status}`,
  );
}

