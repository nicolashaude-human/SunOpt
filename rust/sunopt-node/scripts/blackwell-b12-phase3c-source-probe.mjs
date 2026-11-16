// rust/sunopt-node/scripts/blackwell-b12-phase3c-source-probe.mjs
//
// Read-only Phase 3C source probe.
// It extracts the exact CURRENT B8 resident-chunk Rust/CUDA interfaces from
// the installed project so the next coupling patch can use the actual ABI
// rather than a historical generator.
//
// Production files are not modified.

import {
  readFileSync,
} from "node:fs";

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

const rustPath =
  path.join(
    root,
    "src",
    "cuda_context.rs",
  );

const cudaPath =
  path.join(
    root,
    "cuda",
    "cuda_context.cu",
  );


const rust =
  readFileSync(
    rustPath,
    "utf8",
  );

const cuda =
  readFileSync(
    cudaPath,
    "utf8",
  );


function lineNumberAt(
  source,
  index,
) {
  return (
    source
      .slice(
        0,
        index,
      )
      .split(
        "\n",
      )
      .length
  );
}


function extractBalancedFunction(
  source,
  anchor,
) {
  const start =
    source.indexOf(
      anchor,
    );

  if (
    start <
    0
  ) {
    throw new Error(
      `Required anchor not found: ${anchor}`,
    );
  }

  const open =
    source.indexOf(
      "{",
      start,
    );

  if (
    open <
    0
  ) {
    throw new Error(
      `Opening brace not found after: ${anchor}`,
    );
  }

  let depth =
    0;

  let inString =
    false;

  let stringQuote =
    "";

  let escaped =
    false;

  let inLineComment =
    false;

  let inBlockComment =
    false;

  for (
    let i = open;
    i < source.length;
    i += 1
  ) {
    const ch =
      source[i];

    const next =
      source[i + 1] ??
      "";

    if (
      inLineComment
    ) {
      if (
        ch ===
        "\n"
      ) {
        inLineComment =
          false;
      }

      continue;
    }

    if (
      inBlockComment
    ) {
      if (
        ch ===
          "*" &&
        next ===
          "/"
      ) {
        inBlockComment =
          false;

        i +=
          1;
      }

      continue;
    }

    if (
      inString
    ) {
      if (
        escaped
      ) {
        escaped =
          false;

        continue;
      }

      if (
        ch ===
        "\\"
      ) {
        escaped =
          true;

        continue;
      }

      if (
        ch ===
        stringQuote
      ) {
        inString =
          false;

        stringQuote =
          "";
      }

      continue;
    }

    if (
      ch ===
        "/" &&
      next ===
        "/"
    ) {
      inLineComment =
        true;

      i +=
        1;

      continue;
    }

    if (
      ch ===
        "/" &&
      next ===
        "*"
    ) {
      inBlockComment =
        true;

      i +=
        1;

      continue;
    }

    if (
      ch ===
        "\"" ||
      ch ===
        "'"
    ) {
      inString =
        true;

      stringQuote =
        ch;

      continue;
    }

    if (
      ch ===
      "{"
    ) {
      depth +=
        1;

      continue;
    }

    if (
      ch ===
      "}"
    ) {
      depth -=
        1;

      if (
        depth ===
        0
      ) {
        return {
          startLine:
            lineNumberAt(
              source,
              start,
            ),

          endLine:
            lineNumberAt(
              source,
              i,
            ),

          text:
            source.slice(
              start,
              i +
                1,
            ),
        };
      }
    }
  }

  throw new Error(
    `Unbalanced function after anchor: ${anchor}`,
  );
}


function extractDeclaration(
  source,
  anchor,
  terminator,
) {
  const start =
    source.indexOf(
      anchor,
    );

  if (
    start <
    0
  ) {
    throw new Error(
      `Required declaration anchor not found: ${anchor}`,
    );
  }

  const end =
    source.indexOf(
      terminator,
      start,
    );

  if (
    end <
    0
  ) {
    throw new Error(
      `Declaration terminator ${JSON.stringify(terminator)} not found after ${anchor}`,
    );
  }

  return {
    startLine:
      lineNumberAt(
        source,
        start,
      ),

    endLine:
      lineNumberAt(
        source,
        end +
          terminator.length -
          1,
      ),

    text:
      source.slice(
        start,
        end +
          terminator.length,
      ),
  };
}


function extractAround(
  source,
  needle,
  beforeLines,
  afterLines,
) {
  const lines =
    source.split(
      /\r?\n/,
    );

  const position =
    lines.findIndex(
      (
        line,
      ) =>
        line.includes(
          needle,
        ),
    );

  if (
    position <
    0
  ) {
    throw new Error(
      `Required line not found: ${needle}`,
    );
  }

  const start =
    Math.max(
      0,
      position -
        beforeLines,
    );

  const end =
    Math.min(
      lines.length,
      position +
        afterLines +
        1,
    );

  return {
    startLine:
      start +
      1,

    endLine:
      end,

    text:
      lines
        .slice(
          start,
          end,
        )
        .join(
          "\n",
        ),
  };
}


const rustFfi =
  extractDeclaration(
    rust,
    "fn sunopt_cuda_run_digital_resident_chunk_replay(",
    "-> c_int;",
  );

const rustResultStruct =
  extractBalancedFunction(
    rust,
    "pub struct BlackwellDigitalResidentChunkResult",
  );

const rustVariantStruct =
  extractBalancedFunction(
    rust,
    "pub struct BlackwellDigitalResidentChunkVariant",
  );

const rustRunner =
  extractBalancedFunction(
    rust,
    "pub fn run_blackwell_digital_resident_chunk(",
  );

const cudaEntrypoint =
  extractBalancedFunction(
    cuda,
    'extern "C" int sunopt_cuda_run_digital_resident_chunk_replay(',
  );

const kernelContext =
  extractAround(
    cuda,
    "digital_resident_chunk",
    40,
    220,
  );


const summary = {
  ok:
    true,

  rustPath,
  cudaPath,

  rustFfiLines:
    [
      rustFfi.startLine,
      rustFfi.endLine,
    ],

  rustVariantStructLines:
    [
      rustVariantStruct.startLine,
      rustVariantStruct.endLine,
    ],

  rustResultStructLines:
    [
      rustResultStruct.startLine,
      rustResultStruct.endLine,
    ],

  rustRunnerLines:
    [
      rustRunner.startLine,
      rustRunner.endLine,
    ],

  cudaEntrypointLines:
    [
      cudaEntrypoint.startLine,
      cudaEntrypoint.endLine,
    ],

  kernelContextLines:
    [
      kernelContext.startLine,
      kernelContext.endLine,
    ],
};


console.log(
  "SUNOPT_B12_PHASE3C_SOURCE_PROBE_JSON=" +
    JSON.stringify(
      summary,
    ),
);


function section(
  title,
  result,
) {
  console.log();
  console.log(
    `===== ${title} lines ${result.startLine}-${result.endLine} =====`,
  );

  console.log(
    result.text,
  );

  console.log(
    `===== END ${title} =====`,
  );
}


section(
  "RUST FFI",
  rustFfi,
);

section(
  "RUST VARIANT STRUCT",
  rustVariantStruct,
);

section(
  "RUST RESULT STRUCT",
  rustResultStruct,
);

section(
  "RUST B8 RUNNER",
  rustRunner,
);

section(
  "CUDA RESIDENT ENTRYPOINT",
  cudaEntrypoint,
);

section(
  "CUDA KERNEL CONTEXT",
  kernelContext,
);

