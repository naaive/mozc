//! pyime CLI. CLI agent: implement subcommands:
//!   interactive   — REPL: type pinyin, see ranked candidates
//!   convert       — one-shot convert of args/stdin
//!   predict       — prefix prediction
//!   eval          — run pyime-eval against a gold set, print table + write report.json
//!   build-data    — invoke pyime-data::build_all
//! Default data dir: ./data (overridable with --data).

use clap::Parser;

#[derive(Parser)]
#[command(name = "pyime", about = "Rust Pinyin IME engine (CLI)")]
struct Cli {
    #[arg(long, default_value = "data")]
    data: String,
}

fn main() -> anyhow::Result<()> {
    let _cli = Cli::parse();
    println!("pyime CLI: not yet implemented");
    Ok(())
}
