# SunOpt native npm hooks

Run once from the SunOpt project root:

    node rust/sunopt-node/enable-native-npm-hooks.mjs

This preserves the current package.json and only adds/extends:

- native:install
- predev
- prebuild
- prestart

A backup is written to:

    package.json.before-native-hooks.bak
