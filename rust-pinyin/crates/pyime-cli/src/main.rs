//! pyime CLI — the user-facing command line for the Rust Pinyin IME engine.
//!
//! Subcommands:
//!   interactive   — REPL: type pinyin, see ranked candidates (IME-style top-N)
//!   convert       — one-shot convert of args/stdin (text or --json)
//!   predict       — prefix prediction / completion
//!   eval          — run pyime-eval against a gold set, print table (+ report.json)
//!   build-data    — invoke the pyime-data build pipeline
//!
//! Default data dir: ./data (overridable with --data). Feature toggles map onto
//! `EngineConfig` (fuzzy / correction / english / max-candidates).

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use pyime_core::{Candidate, CandidateKind, Engine, EngineConfig, FuzzySet};

// ===========================================================================
// CLI definition (clap derive)
// ===========================================================================

#[derive(Parser)]
#[command(
    name = "pyime",
    version,
    about = "Rust Pinyin IME engine — lattice + word-bigram decoder",
    long_about = "A high-performance Chinese Pinyin input method engine exposed as a CLI.\n\
                  Convert pinyin to Chinese, predict completions, run interactively, \
                  evaluate against a gold set, or (re)build the data artifacts."
)]
struct Cli {
    /// Directory holding the built data artifacts (lexicon.fst, words.bin, ...).
    #[arg(long, global = true, default_value = "data")]
    data: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Interactive REPL: type pinyin, see ranked candidates. `:q` or EOF to quit.
    Interactive(InteractiveArgs),
    /// One-shot conversion of arguments (or stdin lines if no args).
    Convert(ConvertArgs),
    /// Prefix prediction / completion for a partial input.
    Predict(PredictArgs),
    /// Evaluate the engine against a gold set and print a metrics table.
    Eval(EvalArgs),
    /// Build the data artifacts via the pyime-data pipeline.
    BuildData(BuildDataArgs),
}

/// Engine feature toggles shared by interactive/convert/predict.
///
/// All features default to ON (matching `EngineConfig::default`); the `--no-*`
/// flags turn them off.
#[derive(Args, Clone)]
struct FeatureArgs {
    /// Disable fuzzy-syllable matching (zh↔z, an↔ang, ...).
    #[arg(long = "no-fuzzy", action = clap::ArgAction::SetTrue)]
    no_fuzzy: bool,
    /// Disable typo correction (edit-distance ≤ 1).
    #[arg(long = "no-correct", action = clap::ArgAction::SetTrue)]
    no_correct: bool,
    /// Disable English passthrough ranking.
    #[arg(long = "no-english", action = clap::ArgAction::SetTrue)]
    no_english: bool,
}

impl FeatureArgs {
    /// Apply these toggles onto a base config.
    fn apply(&self, mut cfg: EngineConfig) -> EngineConfig {
        if self.no_fuzzy {
            cfg.fuzzy = FuzzySet::none();
        }
        cfg.enable_correction = !self.no_correct;
        cfg.enable_english = !self.no_english;
        cfg
    }
}

#[derive(Args)]
struct InteractiveArgs {
    #[command(flatten)]
    features: FeatureArgs,
    /// Max candidates to display per line (IME-style top-N).
    #[arg(short = 'n', long = "top", default_value_t = 9)]
    top: usize,
}

#[derive(Args)]
struct ConvertArgs {
    /// Inputs to convert. If omitted, read one input per line from stdin.
    inputs: Vec<String>,
    #[command(flatten)]
    features: FeatureArgs,
    /// Number of candidates to show per input.
    #[arg(long, default_value_t = 9)]
    top: usize,
    /// Emit candidates as JSON (one JSON object per input).
    #[arg(long)]
    json: bool,
    /// Verbose: also print each candidate's score and kind.
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Args)]
struct PredictArgs {
    /// Inputs to predict from. If omitted, read one per line from stdin.
    inputs: Vec<String>,
    #[command(flatten)]
    features: FeatureArgs,
    /// Number of predictions to show per input.
    #[arg(long, default_value_t = 9)]
    top: usize,
    /// Emit predictions as JSON.
    #[arg(long)]
    json: bool,
    /// Verbose: also print each candidate's score and kind.
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Args)]
struct EvalArgs {
    /// Gold set path (JSONL). Auto-generated if missing.
    #[arg(long, default_value = "data/gold.jsonl")]
    gold: PathBuf,
    /// Also write the machine-readable report JSON to this path.
    #[arg(long)]
    json: Option<PathBuf>,
    /// RNG seed for gold generation / sampling (reproducibility).
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Args)]
struct BuildDataArgs {
    /// Output directory for built artifacts.
    #[arg(long, default_value = "data")]
    out: PathBuf,
    /// Corpus / download cache directory.
    #[arg(long, default_value = "corpus")]
    corpus: PathBuf,
}

// ===========================================================================
// Entry point
// ===========================================================================

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Interactive(a) => cmd_interactive(&cli.data, &a),
        Command::Convert(a) => cmd_convert(&cli.data, &a),
        Command::Predict(a) => cmd_predict(&cli.data, &a),
        Command::Eval(a) => cmd_eval(&cli.data, &a),
        Command::BuildData(a) => cmd_build_data(&a),
    }
}

// ===========================================================================
// Engine loading
// ===========================================================================

/// Load the engine, mapping a missing data dir to a friendly hint.
fn load_engine(data_dir: &Path) -> Result<Engine> {
    if !data_dir.join("lexicon.fst").exists() {
        anyhow::bail!(
            "no data found in '{}' (lexicon.fst missing). Run `pyime build-data` first.",
            data_dir.display()
        );
    }
    Engine::load(data_dir)
        .with_context(|| format!("loading engine data from '{}'", data_dir.display()))
}

fn kind_str(kind: CandidateKind) -> &'static str {
    match kind {
        CandidateKind::Chinese => "CN",
        CandidateKind::English => "EN",
        CandidateKind::Mixed => "MIX",
    }
}

// ===========================================================================
// Shared rendering helpers (also exercised by the integration test)
// ===========================================================================

/// Render a single candidate as a JSON value: `{ "text", "score", "kind" }`.
fn candidate_json(c: &Candidate) -> serde_json::Value {
    serde_json::json!({
        "text": c.text,
        "score": c.score,
        "kind": kind_str(c.kind),
    })
}

/// Build the JSON object for one input and its candidates.
fn result_json(input: &str, cands: &[Candidate], top: usize) -> serde_json::Value {
    let list: Vec<serde_json::Value> =
        cands.iter().take(top).map(candidate_json).collect();
    serde_json::json!({ "input": input, "candidates": list })
}

/// Print candidates in IME-numbered style to `out`.
fn print_candidates<W: Write>(
    out: &mut W,
    cands: &[Candidate],
    top: usize,
    verbose: bool,
) -> std::io::Result<()> {
    if cands.is_empty() {
        writeln!(out, "  (no candidates)")?;
        return Ok(());
    }
    for (i, c) in cands.iter().take(top).enumerate() {
        if verbose {
            writeln!(
                out,
                "  {}. {}  [score={:.1} {}]",
                i + 1,
                c.text,
                c.score,
                kind_str(c.kind)
            )?;
        } else {
            writeln!(out, "  {}. {}", i + 1, c.text)?;
        }
    }
    Ok(())
}

// ===========================================================================
// convert
// ===========================================================================

fn cmd_convert(data_dir: &Path, args: &ConvertArgs) -> Result<()> {
    let engine = load_engine(data_dir)?;
    let cfg = args.features.apply(EngineConfig::default());

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    let mut emit = |input: &str| -> Result<()> {
        let input = input.trim();
        if input.is_empty() {
            return Ok(());
        }
        let cands = engine.convert(input, &cfg);
        if args.json {
            let v = result_json(input, &cands, args.top);
            writeln!(out, "{}", serde_json::to_string(&v)?)?;
        } else {
            writeln!(out, "{}:", input)?;
            print_candidates(&mut out, &cands, args.top, args.verbose)?;
        }
        Ok(())
    };

    if args.inputs.is_empty() {
        for line in std::io::stdin().lock().lines() {
            emit(&line?)?;
        }
    } else {
        for input in &args.inputs {
            emit(input)?;
        }
    }
    Ok(())
}

// ===========================================================================
// predict
// ===========================================================================

fn cmd_predict(data_dir: &Path, args: &PredictArgs) -> Result<()> {
    let engine = load_engine(data_dir)?;
    let cfg = args.features.apply(EngineConfig::default());

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    let mut emit = |input: &str| -> Result<()> {
        let input = input.trim();
        if input.is_empty() {
            return Ok(());
        }
        let cands = engine.predict(input, &cfg);
        if args.json {
            let v = result_json(input, &cands, args.top);
            writeln!(out, "{}", serde_json::to_string(&v)?)?;
        } else {
            writeln!(out, "{} ...:", input)?;
            print_candidates(&mut out, &cands, args.top, args.verbose)?;
        }
        Ok(())
    };

    if args.inputs.is_empty() {
        for line in std::io::stdin().lock().lines() {
            emit(&line?)?;
        }
    } else {
        for input in &args.inputs {
            emit(input)?;
        }
    }
    Ok(())
}

// ===========================================================================
// interactive REPL
// ===========================================================================

fn cmd_interactive(data_dir: &Path, args: &InteractiveArgs) -> Result<()> {
    let engine = load_engine(data_dir)?;
    let mut cfg = args.features.apply(EngineConfig::default());
    let mut top = args.top.max(1);

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    writeln!(
        out,
        "pyime interactive. Type pinyin and press Enter. Commands:\n  \
         :q                quit\n  \
         :fuzzy on|off     toggle fuzzy syllables\n  \
         :correct on|off   toggle typo correction\n  \
         :english on|off   toggle English passthrough\n  \
         :n <k>            set max candidates shown"
    )?;
    write!(out, "> ")?;
    out.flush()?;

    for line in std::io::stdin().lock().lines() {
        let line = line?;
        let trimmed = line.trim();

        if trimmed == ":q" || trimmed == ":quit" {
            break;
        }

        // Inline command toggles.
        if let Some(rest) = trimmed.strip_prefix(':') {
            match handle_command(rest, &mut cfg, &mut top) {
                Ok(msg) => writeln!(out, "{}", msg)?,
                Err(msg) => writeln!(out, "  ! {}", msg)?,
            }
            write!(out, "> ")?;
            out.flush()?;
            continue;
        }

        if trimmed.is_empty() {
            write!(out, "> ")?;
            out.flush()?;
            continue;
        }

        let cands = engine.convert(trimmed, &cfg);
        print_candidates(&mut out, &cands, top, true)?;
        write!(out, "> ")?;
        out.flush()?;
    }

    writeln!(out, "\nbye.")?;
    Ok(())
}

/// Parse and apply a `:`-prefixed REPL command (without the leading `:`).
/// Returns Ok(status message) or Err(error message).
fn handle_command(
    rest: &str,
    cfg: &mut EngineConfig,
    top: &mut usize,
) -> std::result::Result<String, String> {
    let mut parts = rest.split_whitespace();
    let cmd = parts.next().unwrap_or("");
    let arg = parts.next();

    let parse_on_off = |a: Option<&str>| -> std::result::Result<bool, String> {
        match a {
            Some("on") => Ok(true),
            Some("off") => Ok(false),
            other => Err(format!("expected 'on' or 'off', got {:?}", other)),
        }
    };

    match cmd {
        "fuzzy" => {
            let on = parse_on_off(arg)?;
            cfg.fuzzy = if on { FuzzySet::all() } else { FuzzySet::none() };
            Ok(format!("  fuzzy = {}", if on { "on" } else { "off" }))
        }
        "correct" => {
            cfg.enable_correction = parse_on_off(arg)?;
            Ok(format!(
                "  correct = {}",
                if cfg.enable_correction { "on" } else { "off" }
            ))
        }
        "english" => {
            cfg.enable_english = parse_on_off(arg)?;
            Ok(format!(
                "  english = {}",
                if cfg.enable_english { "on" } else { "off" }
            ))
        }
        "n" => {
            let k: usize = arg
                .ok_or_else(|| "usage: :n <k>".to_string())?
                .parse()
                .map_err(|_| "k must be a positive integer".to_string())?;
            if k == 0 {
                return Err("k must be >= 1".to_string());
            }
            *top = k;
            Ok(format!("  showing top {}", k))
        }
        other => Err(format!("unknown command ':{}'", other)),
    }
}

// ===========================================================================
// eval
// ===========================================================================

fn cmd_eval(data_dir: &Path, args: &EvalArgs) -> Result<()> {
    let engine = load_engine(data_dir)?;
    let cfg = EngineConfig::default();

    // Auto-generate the gold set if missing.
    ensure_gold(&args.gold, data_dir, args.seed)?;

    let report = run_eval_report(&engine, &cfg, &args.gold, args.seed)?;

    // Human-readable table.
    print!("{}", report.table);
    if !report.table.ends_with('\n') {
        println!();
    }

    // Optional machine-readable JSON.
    if let Some(json_path) = &args.json {
        match &report.json {
            Some(json) => {
                std::fs::write(json_path, json)
                    .with_context(|| format!("writing report json to {}", json_path.display()))?;
                eprintln!("wrote report json: {}", json_path.display());
            }
            None => {
                eprintln!(
                    "note: pyime-eval does not yet expose a JSON report; skipping --json {}",
                    json_path.display()
                );
            }
        }
    }
    Ok(())
}

/// A normalized eval result: a printable table and (optionally) a JSON blob.
///
/// This wrapper insulates the CLI from the exact shape of the pyime-eval API
/// while it is still under construction. We adapt to whichever surface exists.
struct EvalOutput {
    table: String,
    json: Option<String>,
}

/// Ensure a gold set exists at `path`, generating it if absent.
fn ensure_gold(path: &Path, data_dir: &Path, seed: u64) -> Result<()> {
    if path.exists() {
        return Ok(());
    }
    eprintln!("gold set {} missing; generating ...", path.display());

    // Sources, per DESIGN/eval task.
    let sentences = data_dir.join("../corpus/heldout_sentences.txt");
    let sentences = if sentences.exists() {
        sentences
    } else {
        // Common case: data/ is a sibling of corpus/ under the workspace root.
        PathBuf::from("corpus/heldout_sentences.txt")
    };
    let hanzi = data_dir.join("hanzi_pinyin.tsv");

    if !sentences.exists() {
        anyhow::bail!(
            "cannot generate gold set: held-out sentences not found at {}. \
             Run `pyime build-data` first.",
            sentences.display()
        );
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    generate_gold_adapter(&sentences, &hanzi, path, seed)
}

// --- Adapters onto the (evolving) pyime-eval API ---------------------------
//
// The eval task specifies `run_eval`, `generate_gold`, `render_table`, and
// `EvalReport`. Today only `run_eval(&Engine, &EngineConfig, &Path) -> Result<String>`
// is published. We code against that stable surface so the CLI always builds,
// and produce the richer output (JSON / generated gold) as those APIs land.

fn generate_gold_adapter(
    sentences: &Path,
    _hanzi: &Path,
    out: &Path,
    _seed: u64,
) -> Result<()> {
    // Preferred path (once available): pyime_eval::generate_gold(sentences, hanzi, seed)
    // writing JSONL gold pairs. Until that exists, fall back to a minimal gold set
    // derived from the held-out sentences so `eval` is runnable end-to-end.
    fallback_generate_gold(sentences, out)
        .with_context(|| format!("generating gold set at {}", out.display()))
}

/// Minimal gold generator: take held-out Chinese sentences and emit
/// `{"input": <pinyin>, "expected": <hanzi>, "bucket": "full"}` JSONL lines,
/// converting each sentence to canonical pinyin via the hanzi table.
fn fallback_generate_gold(sentences: &Path, out: &Path) -> Result<()> {
    // Load the hanzi->pinyin table that sits next to the data dir.
    let hanzi_tsv = out
        .parent()
        .map(|p| p.join("hanzi_pinyin.tsv"))
        .filter(|p| p.exists())
        .or_else(|| {
            let p = PathBuf::from("data/hanzi_pinyin.tsv");
            p.exists().then_some(p)
        });

    let table = match hanzi_tsv {
        Some(p) => load_hanzi_table(&p)?,
        None => {
            anyhow::bail!(
                "hanzi_pinyin.tsv not found; cannot build pinyin for gold set. \
                 Run `pyime build-data` first."
            )
        }
    };

    let text = std::fs::read_to_string(sentences)
        .with_context(|| format!("reading {}", sentences.display()))?;
    let mut buf = String::new();
    let mut n = 0usize;
    for line in text.lines() {
        let s = line.trim();
        if s.chars().count() < 2 || s.chars().count() > 20 {
            continue;
        }
        // Build canonical pinyin (no separators) for all-CJK lines only.
        let mut pinyin = String::new();
        let mut ok = true;
        for ch in s.chars() {
            match table.get(&ch) {
                Some(py) => pinyin.push_str(py),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok || pinyin.is_empty() {
            continue;
        }
        let obj = serde_json::json!({
            "input": pinyin,
            "expected": s,
            "bucket": "full",
        });
        buf.push_str(&serde_json::to_string(&obj)?);
        buf.push('\n');
        n += 1;
        if n >= 2000 {
            break;
        }
    }
    if n == 0 {
        anyhow::bail!("produced an empty gold set from {}", sentences.display());
    }
    std::fs::write(out, buf).with_context(|| format!("writing {}", out.display()))?;
    eprintln!("generated {} gold pairs -> {}", n, out.display());
    Ok(())
}

/// Load a `char \t pinyin` TSV into a map (first reading per char).
fn load_hanzi_table(path: &Path) -> Result<std::collections::HashMap<char, String>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let mut map = std::collections::HashMap::new();
    for line in text.lines() {
        let mut it = line.splitn(2, '\t');
        if let (Some(c), Some(py)) = (it.next(), it.next()) {
            if let Some(ch) = c.chars().next() {
                // pinyin may be comma/space separated alternatives; take the first.
                let first = py
                    .split([',', ' ', '/'])
                    .next()
                    .unwrap_or(py)
                    .trim()
                    .to_string();
                if !first.is_empty() {
                    map.entry(ch).or_insert(first);
                }
            }
        }
    }
    Ok(map)
}

fn run_eval_report(
    engine: &Engine,
    cfg: &EngineConfig,
    gold: &Path,
    _seed: u64,
) -> Result<EvalOutput> {
    // Stable surface today: returns the rendered table as a String.
    let table = pyime_eval::run_eval(engine, cfg, gold)
        .context("running pyime-eval")?;
    // When pyime-eval exposes EvalReport + render_table, we will additionally
    // serialize report.json here. For now the table is the full report.
    Ok(EvalOutput { table, json: None })
}

// ===========================================================================
// build-data
// ===========================================================================

fn cmd_build_data(args: &BuildDataArgs) -> Result<()> {
    eprintln!(
        "building data: out={} corpus={}",
        args.out.display(),
        args.corpus.display()
    );
    pyime_data::build_all(&args.out, &args.corpus).context("building data artifacts")?;
    eprintln!("build-data complete.");
    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn data_dir() -> PathBuf {
        PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data"))
    }

    #[test]
    fn feature_toggles_apply() {
        let base = EngineConfig::default();
        assert!(!base.fuzzy.is_empty());
        assert!(base.enable_correction);
        assert!(base.enable_english);

        let f = FeatureArgs {
            no_fuzzy: true,
            no_correct: true,
            no_english: true,
        };
        let cfg = f.apply(base);
        assert!(cfg.fuzzy.is_empty());
        assert!(!cfg.enable_correction);
        assert!(!cfg.enable_english);
    }

    #[test]
    fn repl_commands_parse() {
        let mut cfg = EngineConfig::default();
        let mut top = 9usize;

        assert!(handle_command("fuzzy off", &mut cfg, &mut top).is_ok());
        assert!(cfg.fuzzy.is_empty());
        assert!(handle_command("fuzzy on", &mut cfg, &mut top).is_ok());
        assert!(!cfg.fuzzy.is_empty());

        assert!(handle_command("correct off", &mut cfg, &mut top).is_ok());
        assert!(!cfg.enable_correction);

        assert!(handle_command("english off", &mut cfg, &mut top).is_ok());
        assert!(!cfg.enable_english);

        assert!(handle_command("n 5", &mut cfg, &mut top).is_ok());
        assert_eq!(top, 5);

        // Errors.
        assert!(handle_command("fuzzy maybe", &mut cfg, &mut top).is_err());
        assert!(handle_command("n 0", &mut cfg, &mut top).is_err());
        assert!(handle_command("bogus", &mut cfg, &mut top).is_err());
    }

    #[test]
    fn result_json_shape() {
        let c = Candidate {
            text: "你好".to_string(),
            score: 12.5,
            segments: vec![],
            kind: CandidateKind::Chinese,
        };
        let v = result_json("nihao", std::slice::from_ref(&c), 9);
        assert_eq!(v["input"], "nihao");
        assert_eq!(v["candidates"][0]["text"], "你好");
        assert_eq!(v["candidates"][0]["kind"], "CN");
    }

    /// Real end-to-end conversion against the built data dir (skips if absent).
    #[test]
    fn convert_produces_chinese_if_data() {
        let dir = data_dir();
        if !dir.join("lexicon.fst").exists() {
            eprintln!("(skipping: no built data at {})", dir.display());
            return;
        }
        let engine = load_engine(&dir).expect("load engine");
        let cfg = EngineConfig::default();
        let cands = engine.convert("nihao", &cfg);
        assert!(!cands.is_empty(), "nihao should produce candidates");
        // Top result should contain Han characters.
        let top = &cands[0].text;
        assert!(
            top.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c)),
            "expected Chinese output for 'nihao', got {top:?}"
        );
    }
}
