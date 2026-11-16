// rust/sunopt-core/examples/inspect_rust_payload.rs

use std::{
    env,
    fs,
    path::Path,
};

use sunopt_core::inspect_rust_payloads;


fn main() {

    let file_name =
        env::args()
            .nth(1)
            .expect(
                "Usage: inspect_rust_payload <file.ts>",
            );


    let file_path =
        Path::new(
            &file_name,
        );


    let source =
        fs::read_to_string(
            file_path,
        )
        .expect(
            "Could not read TypeScript file.",
        );


    let scopes =
        inspect_rust_payloads(
            &source,
            file_path,
        )
        .expect(
            "Could not inspect Rust payloads.",
        );


    if scopes.is_empty() {

        println!(
            "No rust_* payload scopes found."
        );

        return;
    }


    for scope in scopes {

        println!(
            "Rust payload found:"
        );

        println!(
            "  label: {}",
            scope.label,
        );

        println!(
            "  body: {}",
            scope.body_kind,
        );

        println!(
            "  statements: {}",
            scope.statement_count,
        );

        println!(
            "  first statement: {}",
            scope
                .first_statement_kind
                .as_deref()
                .unwrap_or("none"),
        );
    }
}