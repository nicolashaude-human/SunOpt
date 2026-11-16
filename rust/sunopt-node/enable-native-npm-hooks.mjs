// rust/sunopt-node/enable-native-npm-hooks.mjs

import {
  copyFileSync,
  readFileSync,
  writeFileSync,
} from "node:fs";

import {
  resolve,
} from "node:path";


const projectRoot =
  process.cwd();

const packageJsonPath =
  resolve(
    projectRoot,
    "package.json",
  );

const backupPath =
  resolve(
    projectRoot,
    "package.json.before-native-hooks.bak",
  );


function prependScript(
  scripts,
  key,
  command,
) {

  const current =
    typeof scripts[key] === "string"
      ? scripts[key].trim()
      : "";

  if (
    current.includes(
      "npm run native:install",
    )
  ) {
    return;
  }

  scripts[key] =
    current.length > 0
      ? `${command} && ${current}`
      : command;
}


const packageJson =
  JSON.parse(
    readFileSync(
      packageJsonPath,
      "utf8",
    ),
  );


if (
  typeof packageJson.scripts !== "object" ||
  packageJson.scripts === null ||
  Array.isArray(packageJson.scripts)
) {
  packageJson.scripts = {};
}


copyFileSync(
  packageJsonPath,
  backupPath,
);


packageJson.scripts["native:install"] =
  "node rust/sunopt-node/install-local-package.mjs";

prependScript(
  packageJson.scripts,
  "predev",
  "npm run native:install",
);

prependScript(
  packageJson.scripts,
  "prebuild",
  "npm run native:install",
);

prependScript(
  packageJson.scripts,
  "prestart",
  "npm run native:install",
);


writeFileSync(
  packageJsonPath,
  `${JSON.stringify(packageJson, null, 2)}\n`,
  "utf8",
);


console.log(
  "[SunOpt] Native npm hooks enabled.",
  {
    packageJson:
      packageJsonPath,

    backup:
      backupPath,

    scripts: {
      "native:install":
        packageJson.scripts["native:install"],

      predev:
        packageJson.scripts.predev,

      prebuild:
        packageJson.scripts.prebuild,

      prestart:
        packageJson.scripts.prestart,
    },
  },
);
