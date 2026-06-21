//! pyime CLI — the user-facing command line for the Rust Pinyin IME engine.
//!
//! Subcommands:
//!   interactive   — REPL: type pinyin, see ranked candidates, SELECT to commit/learn
//!   convert       — one-shot convert of args/stdin (text or --json)
//!   predict       — prefix prediction / completion
//!   commit        — batch / scripted learning: record `input → chosen` selections
//!   eval          — run pyime-eval against a gold set, print table (+ report.json)
//!   build-data    — invoke the pyime-data build pipeline
//!
//! Default data dir: ./data (overridable with --data). Feature toggles map onto
//! `EngineConfig` (fuzzy / correction / english / max-candidates).
//!
//! ## User dictionary / online adaptation (`--user`)
//! The global `--user <PATH>` flag attaches a persistent user model (loaded from
//! `PATH` if it exists, created otherwise). With it attached, selections made in
//! `interactive` and rows fed to `commit` are *learned*: re-typing the same pinyin
//! re-surfaces the committed candidate at/near #1, and the history persists across
//! runs (saved to `PATH`). Without `--user`, behavior is byte-identical to before
//! (no user model). `--no-learn` keeps the model loaded for inspection but sets
//! `user_weight = 0`, disabling personalization at decode time.

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

    /// Persistent user-model path for online personalization. Loaded if it exists,
    /// created on save otherwise. When set, `interactive` selections and `commit`
    /// rows are learned and persisted; re-typing the same pinyin re-surfaces the
    /// committed candidate. Omit for classic (no-user-model) behavior.
    #[arg(long, global = true, value_name = "PATH")]
    user: Option<PathBuf>,

    /// Keep the user model loaded but disable personalization (user_weight = 0).
    /// History is still recorded/saved on commit, just not used to re-rank.
    #[arg(long, global = true, action = clap::ArgAction::SetTrue)]
    no_learn: bool,

    #[command(subcommand)]
    command: Command,
}

impl Cli {
    /// Map the global `--no-learn` flag onto a config's `user_weight`.
    fn apply_user_weight(&self, mut cfg: EngineConfig) -> EngineConfig {
        if self.no_learn {
            cfg.user_weight = 0;
        }
        cfg
    }
}

#[derive(Subcommand)]
enum Command {
    /// Interactive REPL: type pinyin, see ranked candidates. `:q` or EOF to quit.
    Interactive(InteractiveArgs),
    /// One-shot conversion of arguments (or stdin lines if no args).
    Convert(ConvertArgs),
    /// Prefix prediction / completion for a partial input.
    Predict(PredictArgs),
    /// Batch / scripted learning: record `input → chosen` commits into the user model.
    Commit(CommitArgs),
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
struct CommitArgs {
    /// The pinyin input buffer that was typed (e.g. `beijing`).
    /// Optional: if omitted, read `input<TAB>chosen` lines from stdin instead.
    input: Option<String>,
    /// The committed/chosen surface (e.g. `背景`). Required when `input` is given.
    chosen: Option<String>,
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
    match &cli.command {
        Command::Interactive(a) => cmd_interactive(&cli, a),
        Command::Convert(a) => cmd_convert(&cli, a),
        Command::Predict(a) => cmd_predict(&cli, a),
        Command::Commit(a) => cmd_commit(&cli, a),
        Command::Eval(a) => cmd_eval(&cli.data, a),
        Command::BuildData(a) => cmd_build_data(a),
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

/// Load the engine and, if `--user <PATH>` was given, attach the persistent user
/// model from that path. Without `--user`, no user model is attached and behavior
/// is identical to the classic engine.
fn load_engine_with_user(cli: &Cli) -> Result<Engine> {
    let engine = load_engine(&cli.data)?;
    Ok(match &cli.user {
        Some(path) => engine.with_user_model(Some(path.clone())),
        None => engine,
    })
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

fn cmd_convert(cli: &Cli, args: &ConvertArgs) -> Result<()> {
    let engine = load_engine_with_user(cli)?;
    let cfg = cli.apply_user_weight(args.features.apply(EngineConfig::default()));

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

fn cmd_predict(cli: &Cli, args: &PredictArgs) -> Result<()> {
    let engine = load_engine_with_user(cli)?;
    let cfg = cli.apply_user_weight(args.features.apply(EngineConfig::default()));

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
// commit (batch / scripted learning)
// ===========================================================================

/// `pyime --user <path> commit <input> <chosen>` records a single committed
/// selection; with no positional args it reads `input<TAB>chosen` lines from stdin
/// (blank lines and lines missing a tab are skipped). All commits are applied to
/// the attached user model and persisted via `save_user`, then a summary is printed.
///
/// Requires `--user` to do anything useful: without it there is no model to persist
/// (we still parse/validate input and report, but warn that nothing was saved).
fn cmd_commit(cli: &Cli, args: &CommitArgs) -> Result<()> {
    let engine = load_engine_with_user(cli)?;

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    if cli.user.is_none() {
        writeln!(
            out,
            "warning: no --user <PATH> set; commits are not persisted (nothing learned)."
        )?;
    }

    let mut n = 0usize;
    let mut apply = |input: &str, chosen: &str, out: &mut dyn Write| -> std::io::Result<()> {
        let input = input.trim();
        let chosen = chosen.trim();
        if input.is_empty() || chosen.is_empty() {
            return Ok(());
        }
        engine.commit(input, chosen);
        n += 1;
        writeln!(out, "  ✓ committed {} -> {}", input, chosen)
    };

    match (&args.input, &args.chosen) {
        (Some(input), Some(chosen)) => {
            apply(input, chosen, &mut out)?;
        }
        (Some(_), None) => {
            anyhow::bail!("`commit <input> <chosen>` needs both arguments (got only <input>)");
        }
        (None, _) => {
            // Read `input<TAB>chosen` lines from stdin.
            for line in std::io::stdin().lock().lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                match line.split_once('\t') {
                    Some((input, chosen)) => apply(input, chosen, &mut out)?,
                    None => writeln!(out, "  ! skipping (no TAB): {:?}", line)?,
                }
            }
        }
    }

    engine.save_user().context("saving user model after commits")?;

    match &cli.user {
        Some(path) => writeln!(out, "committed {} selection(s); saved to {}", n, path.display())?,
        None => writeln!(out, "committed {} selection(s) (not persisted: no --user)", n)?,
    }
    Ok(())
}

// ===========================================================================
// interactive REPL
// ===========================================================================

/// Interactive REPL with a real IME "select-to-learn" loop.
///
/// ## Input protocol (line-oriented, robust to stdin piping)
/// The REPL is a small state machine over input lines:
///   * A **pinyin line** (anything that is not `:`-prefixed, not blank, and not a
///     bare candidate number) decodes the input and prints the numbered candidate
///     list. The decoded input + its candidates become the *pending selection*.
///   * A bare **number** line `1`..`N` (1-based, within the shown top-N) while a
///     selection is pending *commits* that candidate: it calls
///     `engine.commit(input, chosen_text)`, prints `✓ committed <text>`, and (if a
///     user model is attached) learns it. The pending selection is then cleared.
///   * A **blank** line, or a number out of range, clears the pending selection and
///     starts fresh (no commit).
///   * `:`-prefixed lines are inline commands (see `handle_command`), plus:
///       `:save`  persist the user model now (no-op without `--user`)
///       `:q`     quit (persists automatically on exit when `--user` is set)
///
/// Effect: `printf 'beijing\n2\nbeijing\n:q\n'` selects candidate #2 for the first
/// `beijing`, commits/learns it, then shows it promoted toward #1 on the second
/// `beijing`. EOF behaves like `:q`.
fn cmd_interactive(cli: &Cli, args: &InteractiveArgs) -> Result<()> {
    let engine = load_engine_with_user(cli)?;
    let mut cfg = cli.apply_user_weight(args.features.apply(EngineConfig::default()));
    let mut top = args.top.max(1);
    let has_user = cli.user.is_some();

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    writeln!(
        out,
        "pyime interactive. Type pinyin and press Enter, then a number to select.\n\
         Commands:\n  \
         <pinyin>          decode + show numbered candidates\n  \
         <number>          select that candidate -> commit/learn it ('✓ committed ...')\n  \
         <blank>           clear the pending selection (no commit)\n  \
         :save             persist the user model now (needs --user)\n  \
         :q                quit (auto-saves user model on exit when --user is set)\n  \
         :fuzzy on|off     toggle fuzzy syllables\n  \
         :correct on|off   toggle typo correction\n  \
         :english on|off   toggle English passthrough\n  \
         :n <k>            set max candidates shown"
    )?;
    if has_user {
        writeln!(out, "(user model attached: selections are learned and persisted)")?;
    }
    write!(out, "> ")?;
    out.flush()?;

    // Pending selection: the most recently decoded (input, candidates) awaiting a
    // numeric pick. Cleared after a commit or a blank line.
    let mut pending: Option<(String, Vec<Candidate>)> = None;

    for line in std::io::stdin().lock().lines() {
        let line = line?;
        let trimmed = line.trim();

        if trimmed == ":q" || trimmed == ":quit" {
            break;
        }

        // Inline command toggles + :save.
        if let Some(rest) = trimmed.strip_prefix(':') {
            if rest.trim() == "save" {
                match engine.save_user() {
                    Ok(()) if has_user => writeln!(out, "  ✓ saved user model")?,
                    Ok(()) => writeln!(out, "  (no --user set; nothing to save)")?,
                    Err(e) => writeln!(out, "  ! save failed: {e}")?,
                }
                write!(out, "> ")?;
                out.flush()?;
                continue;
            }
            match handle_command(rest, &mut cfg, &mut top) {
                Ok(msg) => writeln!(out, "{}", msg)?,
                Err(msg) => writeln!(out, "  ! {}", msg)?,
            }
            write!(out, "> ")?;
            out.flush()?;
            continue;
        }

        if trimmed.is_empty() {
            // Blank line clears any pending selection.
            pending = None;
            write!(out, "> ")?;
            out.flush()?;
            continue;
        }

        // A bare number selects from the pending candidate list (1-based).
        if let Ok(sel) = trimmed.parse::<usize>() {
            if let Some((input, cands)) = pending.as_ref() {
                let shown = cands.len().min(top);
                if sel >= 1 && sel <= shown {
                    let chosen = cands[sel - 1].text.clone();
                    engine.commit(input, &chosen);
                    writeln!(out, "  ✓ committed {}", chosen)?;
                    pending = None;
                    write!(out, "> ")?;
                    out.flush()?;
                    continue;
                } else {
                    writeln!(out, "  ! selection {} out of range (1..{})", sel, shown)?;
                    pending = None;
                    write!(out, "> ")?;
                    out.flush()?;
                    continue;
                }
            }
            // No pending list: fall through and treat the number as a new input
            // (it will simply decode to whatever the engine makes of it).
        }

        // Otherwise: a new pinyin input. Decode and show candidates.
        let cands = engine.convert(trimmed, &cfg);
        print_candidates(&mut out, &cands, top, true)?;
        pending = Some((trimmed.to_string(), cands));
        write!(out, "> ")?;
        out.flush()?;
    }

    // Persist on clean exit (EOF / :q) when a user model is attached.
    if has_user {
        if let Err(e) = engine.save_user() {
            writeln!(out, "  ! save on exit failed: {e}")?;
        } else {
            writeln!(out, "  ✓ saved user model")?;
        }
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

// --- pyime-eval API adapters -----------------------------------------------

fn generate_gold_adapter(sentences: &Path, hanzi: &Path, out: &Path, seed: u64) -> Result<()> {
    pyime_eval::generate_gold(sentences, hanzi, out, seed)
        .with_context(|| format!("generating gold set at {}", out.display()))
}

fn run_eval_report(
    engine: &Engine,
    cfg: &EngineConfig,
    gold: &Path,
    _seed: u64,
) -> Result<EvalOutput> {
    let report = pyime_eval::run_eval(engine, cfg, gold).context("running pyime-eval")?;
    let table = pyime_eval::render_table(&report);
    let json = serde_json::to_string_pretty(&report).context("serializing eval report")?;
    Ok(EvalOutput {
        table,
        json: Some(json),
    })
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

    /// Helper: rank (1-based) of `want` among candidates for `input`, or None.
    fn rank_of(cands: &[Candidate], want: &str) -> Option<usize> {
        cands.iter().position(|c| c.text == want).map(|i| i + 1)
    }

    /// End-to-end select-to-learn: with a temp `--user` file and real data,
    /// commit `beijing -> 背景` a few times and assert it gets promoted to #1.
    #[test]
    fn commit_promotes_candidate_with_user_model() {
        let dir = data_dir();
        if !dir.join("lexicon.fst").exists() {
            eprintln!("(skipping: no built data at {})", dir.display());
            return;
        }

        let user_path = std::env::temp_dir()
            .join(format!("pyime_cli_user_test_{}.json", std::process::id()));
        let _ = std::fs::remove_file(&user_path);

        let cfg = EngineConfig::default();

        // Baseline: load with a fresh (empty) user model, note 背景's rank.
        let base_rank = {
            let engine = load_engine(&dir)
                .expect("load engine")
                .with_user_model(Some(user_path.clone()));
            let cands = engine.convert("beijing", &cfg);
            let r = rank_of(&cands, "背景").expect("背景 should be a candidate for beijing");
            assert!(r >= 1);
            r
        };

        // Commit beijing -> 背景 several times, then persist.
        {
            let engine = load_engine(&dir)
                .expect("load engine")
                .with_user_model(Some(user_path.clone()));
            for _ in 0..3 {
                engine.commit("beijing", "背景");
            }
            engine.save_user().expect("save user model");
        }
        assert!(user_path.exists(), "user model should persist to {}", user_path.display());

        // Reload (history must survive across runs) and re-convert.
        {
            let engine = load_engine(&dir)
                .expect("load engine")
                .with_user_model(Some(user_path.clone()));
            let cands = engine.convert("beijing", &cfg);
            let new_rank = rank_of(&cands, "背景").expect("背景 still a candidate");
            assert_eq!(
                new_rank, 1,
                "背景 should be promoted to #1 after commits (was #{base_rank}, now #{new_rank})"
            );
        }

        let _ = std::fs::remove_file(&user_path);
    }

    /// With `user_weight = 0` (the `--no-learn` effect), a learned model must NOT
    /// re-rank: behavior is identical to no personalization.
    #[test]
    fn no_learn_disables_personalization() {
        let dir = data_dir();
        if !dir.join("lexicon.fst").exists() {
            eprintln!("(skipping: no built data at {})", dir.display());
            return;
        }
        let user_path = std::env::temp_dir()
            .join(format!("pyime_cli_nolearn_test_{}.json", std::process::id()));
        let _ = std::fs::remove_file(&user_path);

        let engine = load_engine(&dir)
            .expect("load engine")
            .with_user_model(Some(user_path.clone()));
        for _ in 0..3 {
            engine.commit("beijing", "背景");
        }

        let mut cfg = EngineConfig::default();
        cfg.user_weight = 0; // the --no-learn effect
        let cands = engine.convert("beijing", &cfg);
        // With personalization off, 北京 (the clean reading) should remain #1.
        assert_eq!(cands[0].text, "北京", "user_weight=0 must not re-rank");

        let _ = std::fs::remove_file(&user_path);
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
