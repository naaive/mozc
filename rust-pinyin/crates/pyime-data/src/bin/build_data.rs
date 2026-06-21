//! `build-data` — download corpora (cached) and build the `data/` artifacts.
//!
//! Usage:
//!   build-data --out <dir> --corpus <dir> [--offline]
//!   build-data verify --out <dir>
//!
//! Defaults: --out ./data  --corpus ./corpus

use anyhow::{bail, Result};
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1).peekable();

    let mut out = PathBuf::from("data");
    let mut corpus = PathBuf::from("corpus");
    let mut offline = false;
    let mut subcommand: Option<String> = None;

    while let Some(a) = args.next() {
        match a.as_str() {
            "verify" | "build" => subcommand = Some(a),
            "--out" | "-o" => {
                out = PathBuf::from(args.next().ok_or_else(|| anyhow::anyhow!("--out needs value"))?)
            }
            "--corpus" | "-c" => {
                corpus =
                    PathBuf::from(args.next().ok_or_else(|| anyhow::anyhow!("--corpus needs value"))?)
            }
            "--offline" => offline = true,
            "-h" | "--help" => {
                eprintln!(
                    "Usage: build-data [build|verify] --out <dir> --corpus <dir> [--offline]"
                );
                return Ok(());
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    match subcommand.as_deref() {
        Some("verify") => {
            pyime_data::verify(&out)?;
        }
        _ => {
            pyime_data::build_all_opts(&out, &corpus, offline)?;
            // Always self-verify after a build.
            pyime_data::verify(&out)?;
        }
    }
    Ok(())
}
