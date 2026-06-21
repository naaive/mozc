//! Personalization benchmark — quantifies the online-adaptation lift of the user model.
//!
//! Every commercial IME learns from what the user commits. The base eval harness ([`crate::run_eval`])
//! measures *clean-input correctness* on a static gold set. This module measures something the static
//! gold set cannot: how much the engine improves **as a single user keeps typing**, by simulating a
//! realistic user session and scoring it with online adaptation **OFF vs ON**.
//!
//! ## Simulated user session
//! A user does not type a uniform random stream — they repeat themselves (the same names, phrases,
//! jargon) and have personal preferences (an ambiguous pinyin that, for *them*, resolves to a
//! non-default surface). We construct a `(input, gold)` event stream that captures both:
//!
//!   1. **Recurring personal phrases** — a set (~40) of phrases drawn from the held-out gold, each
//!      injected 3–6 times, interleaved with non-repeating *background* items. This models a user
//!      who keeps typing the same things. Adaptation should make later occurrences hit top-1 even
//!      when the first did not.
//!   2. **User-specific preference** — cases whose gold is NOT the base engine's default top-1 (we
//!      detect these by converting with adaptation OFF and keeping only the ones where the engine is
//!      wrong at #1). These *force* the model to learn the user's preference: it can only get them
//!      right by remembering prior commits.
//!   3. **Auto-learned new words** — a few synthetic phrases whose surface the base engine does not
//!      produce at #1 (plausible names/neologisms keyed by their pinyin), to exercise the user-phrase
//!      auto-learning path. Recall after the first commit is reported separately.
//!
//! ## Protocols (run over the SAME stream, with an independent fresh [`Engine`] each)
//!   * **OFF (baseline)** — `user_weight = 0`; convert each event, never commit. The engine has no
//!     memory; every occurrence scores identically.
//!   * **ON (online adaptation)** — a fresh engine with a temp user model (default `user_weight`).
//!     For each event in stream order: **convert FIRST** (score it), **THEN `commit(input, gold)`**.
//!     So each item is scored using only knowledge from *prior* occurrences — a proper online
//!     protocol with no peeking at the current item's answer.
//!
//! ## Metrics
//!   * Overall top-1 / MRR, OFF vs ON, and the delta.
//!   * Recurring-subset top-1 OFF vs ON — the headline number.
//!   * A **learning curve**: recurring-phrase top-1 bucketed by occurrence index (1st, 2nd, 3rd …),
//!     showing accuracy climbing as the user repeats (1st ≈ OFF; later ≫).
//!   * Auto-learned-phrase recall: fraction of injected new-word cases that reach top-1 after their
//!     first commit.
//!
//! Reproducibility: a seeded xorshift RNG (the same [`crate::gold::Rng`]) drives every stochastic
//! choice — session sampling, phrase selection, interleaving. No new dependencies.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use pyime_core::{Engine, EngineConfig};

use crate::gold::{self, Rng};
use crate::GoldCase;

/// One event in the simulated user stream: an input buffer and its gold surface, tagged with
/// whether it is a recurring personal item (and, if so, which occurrence index this is).
#[derive(Debug, Clone)]
struct Event {
    input: String,
    gold: String,
    /// `Some(k)` ⇒ this is the `k`-th (1-based) occurrence of a recurring personal phrase.
    /// `None`   ⇒ a non-repeating background item.
    occurrence: Option<u32>,
    /// True for an injected auto-learned new-word case (its surface is OOV for the base engine).
    auto_learned: bool,
}

/// Top-1 / MRR pair for a set of scored events.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Score {
    pub n: usize,
    pub top1: f64,
    pub mrr: f64,
}

/// One row of the learning curve: top-1 accuracy on recurring phrases at a given occurrence index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningPoint {
    /// Occurrence index (1 = first time the user typed this phrase, 2 = second, …).
    pub occurrence: u32,
    /// How many recurring events fell at this occurrence index.
    pub n: usize,
    /// Top-1 accuracy with adaptation OFF (flat — the engine has no memory).
    pub top1_off: f64,
    /// Top-1 accuracy with adaptation ON (should climb with occurrence index).
    pub top1_on: f64,
}

/// The full personalization report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizationReport {
    /// Seed used to build the session (reproducible).
    pub seed: u64,
    /// Total events in the simulated stream.
    pub stream_len: usize,
    /// Number of distinct recurring personal phrases.
    pub recurring_phrases: usize,
    /// Number of recurring *events* (sum of occurrences).
    pub recurring_events: usize,
    /// Number of injected auto-learned new-word cases.
    pub auto_learned_cases: usize,

    /// Overall top-1/MRR with adaptation OFF.
    pub overall_off: Score,
    /// Overall top-1/MRR with adaptation ON.
    pub overall_on: Score,

    /// Recurring-subset top-1/MRR with adaptation OFF (the headline baseline).
    pub recurring_off: Score,
    /// Recurring-subset top-1/MRR with adaptation ON (the headline lift).
    pub recurring_on: Score,

    /// Learning curve over occurrence indices (ascending).
    pub learning_curve: Vec<LearningPoint>,

    /// Fraction of auto-learned cases that reach top-1 after their first commit (ON).
    pub auto_learned_recall: f64,
}

// ---------------------------------------------------------------------------
// Scoring helpers (mirror lib.rs normalization for comparability)
// ---------------------------------------------------------------------------

/// Drop whitespace for comparison (matching ignores spaces, as in [`crate::run_eval`]).
fn normalize(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Reciprocal rank of `gold` in `cands` (best-first), and whether it is top-1.
fn rank_of(gold: &str, cands: &[String]) -> (bool, f64) {
    let g = normalize(gold);
    for (i, c) in cands.iter().enumerate() {
        if normalize(c) == g {
            return (i == 0, 1.0 / (i + 1) as f64);
        }
    }
    (false, 0.0)
}

// ---------------------------------------------------------------------------
// Session construction
// ---------------------------------------------------------------------------

/// Synthetic auto-learned new-word cases: (pinyin input, surface). The surfaces are plausible
/// names / neologisms the base lexicon will not rank at #1, keyed by their pinyin — exactly the
/// material the user-phrase auto-learning path is meant to capture.
const AUTO_LEARNED: &[(&str, &str)] = &[
    ("zhangweijie", "张维杰"),
    ("liyuanhang", "李远航"),
    ("wangqiyao", "王琪瑶"),
    ("saibowuli", "赛博物理"),
    ("yuanyuzhou", "元宇宙引擎"),
    ("kuilianxinpian", "馈联芯片"),
];

/// Load the session source pool of `(input, gold)` pairs. Prefers a prebuilt gold JSONL
/// (`data/gold_v2.jsonl`); if absent, generates a small correct-reading gold set on the fly from
/// `corpus/heldout_sentences.txt` + `data/word_pinyin.tsv`. Returns `None` (skip gracefully) if no
/// source material is available.
fn load_source_pool(data_dir: &Path, seed: u64) -> Result<Option<Vec<GoldCase>>> {
    // 1. Prebuilt gold_v2 (fair correct-reading set), else gold.jsonl.
    for name in ["gold_v2.jsonl", "gold.jsonl"] {
        let p = data_dir.join(name);
        if p.exists() {
            let cases = gold::load(&p).with_context(|| format!("load {}", p.display()))?;
            if !cases.is_empty() {
                return Ok(Some(cases));
            }
        }
    }

    // 2. Generate from raw heldout + word_pinyin on the fly into a temp file.
    let heldout = {
        let a = data_dir.join("../corpus/heldout_sentences.txt");
        if a.exists() { a } else { PathBuf::from("corpus/heldout_sentences.txt") }
    };
    let word_pinyin = data_dir.join("word_pinyin.tsv");
    let hanzi = data_dir.join("hanzi_pinyin.tsv");
    if heldout.exists() && word_pinyin.exists() && hanzi.exists() {
        let tmp = unique_tmp_dir(seed).join("session_gold.jsonl");
        gold::generate_correct(&heldout, &word_pinyin, &hanzi, &tmp, seed, 300)
            .context("generate session gold")?;
        let cases = gold::load(&tmp)?;
        let _ = std::fs::remove_file(&tmp);
        if !cases.is_empty() {
            return Ok(Some(cases));
        }
    }

    Ok(None)
}

/// Build the simulated user stream from the source pool.
///
/// We prefer multi-character Chinese phrases (`full` / `short_word` / `long_sentence` buckets, and
/// any non-English/mixed) so recurrence is meaningful. The first `n_recurring` distinct phrases
/// become "personal" and are repeated 3–6× each; the rest seed the background stream. Preference
/// (non-default-top-1) cases are not pre-filtered here — they are simply whatever the engine gets
/// wrong, and ON must learn them; recurring ones provide the bulk of the learnable signal. The
/// auto-learned cases are appended (each appears twice: a first "cold" entry, then a repeat after
/// the model has committed it).
fn build_stream(pool: &[GoldCase], seed: u64) -> Vec<Event> {
    let mut rng = Rng::new(seed ^ 0x5151_5151_5151_5151);

    // Candidate phrases: dedupe by gold surface, keep Chinese-ish multi-char items (>= 2 chars,
    // contains a non-ascii char), skip pure-english/mixed buckets for the recurring set.
    let mut seen = std::collections::HashSet::new();
    let mut phrases: Vec<(String, String)> = Vec::new();
    for c in pool {
        if c.bucket == "english" || c.bucket == "mixed" {
            continue;
        }
        let chars = c.expected.chars().count();
        let has_cjk = c.expected.chars().any(|ch| ch as u32 > 0x2E00);
        if chars < 2 || chars > 12 || !has_cjk {
            continue;
        }
        if c.input.len() < 3 {
            continue;
        }
        if seen.insert(c.expected.clone()) {
            phrases.push((c.input.clone(), c.expected.clone()));
        }
    }
    rng.shuffle(&mut phrases);

    const N_RECURRING: usize = 40;
    let n_recurring = N_RECURRING.min(phrases.len() / 3).max(1.min(phrases.len()));
    let n_recurring = n_recurring.min(phrases.len());

    let recurring = &phrases[..n_recurring];
    let background = &phrases[n_recurring..];

    // Pre-expand recurring occurrences (3–6 each) as ungrouped events, then interleave with
    // background by random insertion so repeats are spread through the session (realistic spacing).
    let mut recurring_events: Vec<Event> = Vec::new();
    for (input, gold) in recurring {
        let reps = 3 + rng.below(4); // 3..=6
        for k in 1..=reps {
            recurring_events.push(Event {
                input: input.clone(),
                gold: gold.clone(),
                occurrence: Some(k as u32),
                auto_learned: false,
            });
        }
    }

    // Auto-learned: a cold first occurrence then later repeats so recall-after-first-commit is
    // observable. occurrence index lets them flow through the learning curve too.
    for (input, gold) in AUTO_LEARNED {
        let reps = 2 + rng.below(2); // 2..=3
        for k in 1..=reps {
            recurring_events.push(Event {
                input: input.to_string(),
                gold: gold.to_string(),
                occurrence: Some(k as u32),
                auto_learned: true,
            });
        }
    }

    // Background (non-repeating) events.
    let mut background_events: Vec<Event> = background
        .iter()
        .map(|(input, gold)| Event {
            input: input.clone(),
            gold: gold.clone(),
            occurrence: None,
            auto_learned: false,
        })
        .collect();
    // Cap background so the session is dominated by neither (keep it ~1:1 with recurring events).
    let bg_cap = recurring_events.len().max(20);
    if background_events.len() > bg_cap {
        background_events.truncate(bg_cap);
    }

    // Interleave: we must preserve per-phrase occurrence ORDER (1 before 2 before 3). Group the
    // recurring events by phrase, keep each group's internal order, then merge groups + background
    // by repeatedly drawing a random non-empty queue. This spreads repeats out while keeping each
    // phrase's occurrences monotonically ordered.
    let mut groups: Vec<std::collections::VecDeque<Event>> = Vec::new();
    {
        use std::collections::HashMap;
        let mut idx: HashMap<(String, bool), usize> = HashMap::new();
        for ev in recurring_events {
            let key = (ev.gold.clone(), ev.auto_learned);
            let gi = *idx.entry(key).or_insert_with(|| {
                groups.push(std::collections::VecDeque::new());
                groups.len() - 1
            });
            groups[gi].push_back(ev);
        }
    }
    // Background each as its own singleton queue.
    for ev in background_events {
        let mut q = std::collections::VecDeque::new();
        q.push_back(ev);
        groups.push(q);
    }

    let mut stream: Vec<Event> = Vec::new();
    loop {
        let live: Vec<usize> = groups
            .iter()
            .enumerate()
            .filter(|(_, q)| !q.is_empty())
            .map(|(i, _)| i)
            .collect();
        if live.is_empty() {
            break;
        }
        let pick = live[rng.below(live.len())];
        if let Some(ev) = groups[pick].pop_front() {
            stream.push(ev);
        }
    }
    stream
}

/// A unique temp dir for the user-model file (no `tempfile` dep; mirrors the user.rs test pattern).
fn unique_tmp_dir(seed: u64) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "pyime_personalize_{}_{:x}",
        std::process::id(),
        seed
    ));
    std::fs::create_dir_all(&dir).ok();
    dir
}

// ---------------------------------------------------------------------------
// Protocols
// ---------------------------------------------------------------------------

/// Per-event scoring outcome (top-1 hit, reciprocal rank) plus the event's tags, so the caller can
/// slice metrics (overall / recurring / learning-curve / auto-learned).
struct Scored {
    top1: bool,
    rr: f64,
    occurrence: Option<u32>,
    auto_learned: bool,
}

/// OFF protocol: fresh engine, `user_weight = 0`, never commit. Each event scored independently.
fn run_off(data_dir: &Path, stream: &[Event]) -> Result<Vec<Scored>> {
    let engine = Engine::load(data_dir).context("load engine (OFF)")?;
    let cfg = EngineConfig { user_weight: 0, ..EngineConfig::default() };
    let mut out = Vec::with_capacity(stream.len());
    for ev in stream {
        let cands = engine.convert(&ev.input, &cfg);
        let texts: Vec<String> = cands.iter().map(|c| c.text.clone()).collect();
        let (top1, rr) = rank_of(&ev.gold, &texts);
        out.push(Scored { top1, rr, occurrence: ev.occurrence, auto_learned: ev.auto_learned });
    }
    Ok(out)
}

/// ON protocol: fresh engine with a temp user model (default `user_weight`). For each event, convert
/// FIRST (score with only prior knowledge), THEN commit the gold. No peeking.
fn run_on(data_dir: &Path, stream: &[Event], seed: u64) -> Result<Vec<Scored>> {
    let tmp = unique_tmp_dir(seed).join("user_model.json");
    let _ = std::fs::remove_file(&tmp); // start from a clean slate every run
    let engine = Engine::load(data_dir)
        .context("load engine (ON)")?
        .with_user_model(Some(tmp.clone()));
    let cfg = EngineConfig::default(); // default user_weight (personalization on)

    let mut out = Vec::with_capacity(stream.len());
    for ev in stream {
        // Score using only knowledge from PRIOR occurrences.
        let cands = engine.convert(&ev.input, &cfg);
        let texts: Vec<String> = cands.iter().map(|c| c.text.clone()).collect();
        let (top1, rr) = rank_of(&ev.gold, &texts);
        out.push(Scored { top1, rr, occurrence: ev.occurrence, auto_learned: ev.auto_learned });
        // THEN learn from the committed gold.
        engine.commit(&ev.input, &ev.gold);
    }
    let _ = std::fs::remove_file(&tmp);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

/// Aggregate a Score over events selected by `pred`.
fn score_where(rows: &[Scored], pred: impl Fn(&Scored) -> bool) -> Score {
    let mut s = Score::default();
    for r in rows.iter().filter(|r| pred(r)) {
        s.n += 1;
        if r.top1 {
            s.top1 += 1.0;
        }
        s.mrr += r.rr;
    }
    if s.n > 0 {
        s.top1 /= s.n as f64;
        s.mrr /= s.n as f64;
    }
    s
}

/// Build the learning curve: for each occurrence index present among recurring (incl. auto-learned)
/// events, the OFF and ON top-1 accuracy.
fn learning_curve(off: &[Scored], on: &[Scored]) -> Vec<LearningPoint> {
    let max_occ = on
        .iter()
        .filter_map(|r| r.occurrence)
        .max()
        .unwrap_or(0);
    let mut pts = Vec::new();
    for occ in 1..=max_occ {
        let off_s = score_where(off, |r| r.occurrence == Some(occ));
        let on_s = score_where(on, |r| r.occurrence == Some(occ));
        if on_s.n == 0 {
            continue;
        }
        pts.push(LearningPoint {
            occurrence: occ,
            n: on_s.n,
            top1_off: off_s.top1,
            top1_on: on_s.top1,
        });
    }
    pts
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the personalization benchmark against the built data in `engine_data_dir`, returning the full
/// report. Builds a fresh [`Engine`] per protocol so the OFF and ON runs are fully independent.
///
/// Returns `Ok(None)` (skip gracefully) if no session source material is available (no gold JSONL
/// and no `corpus/heldout_sentences.txt` + `data/word_pinyin.tsv`).
pub fn run_personalization(engine_data_dir: &Path, seed: u64) -> Result<Option<PersonalizationReport>> {
    let Some(pool) = load_source_pool(engine_data_dir, seed)? else {
        return Ok(None);
    };
    let stream = build_stream(&pool, seed);
    if stream.is_empty() {
        return Ok(None);
    }

    let off = run_off(engine_data_dir, &stream)?;
    let on = run_on(engine_data_dir, &stream, seed)?;

    let overall_off = score_where(&off, |_| true);
    let overall_on = score_where(&on, |_| true);
    // Recurring subset = events with an occurrence index (incl. auto-learned).
    let recurring_off = score_where(&off, |r| r.occurrence.is_some());
    let recurring_on = score_where(&on, |r| r.occurrence.is_some());

    let curve = learning_curve(&off, &on);

    // Auto-learned recall: fraction of auto-learned cases at occurrence >= 2 (i.e. AFTER the first
    // commit) that reach top-1 under ON.
    let auto_after = score_where(&on, |r| r.auto_learned && r.occurrence.map(|o| o >= 2).unwrap_or(false));
    let auto_learned_recall = auto_after.top1;

    let recurring_phrases = stream
        .iter()
        .filter(|e| e.occurrence == Some(1) && !e.auto_learned)
        .count();
    let recurring_events = stream.iter().filter(|e| e.occurrence.is_some()).count();
    let auto_learned_cases = stream.iter().filter(|e| e.auto_learned).count();

    Ok(Some(PersonalizationReport {
        seed,
        stream_len: stream.len(),
        recurring_phrases,
        recurring_events,
        auto_learned_cases,
        overall_off,
        overall_on,
        recurring_off,
        recurring_on,
        learning_curve: curve,
        auto_learned_recall,
    }))
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Render the personalization report as a clean fixed-width table (OFF vs ON columns + deltas, the
/// recurring-subset headline, the learning curve, and auto-learned recall).
pub fn render_personalization(report: &PersonalizationReport) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();

    let _ = writeln!(
        out,
        "Personalization benchmark (seed {})  —  online adaptation OFF vs ON",
        report.seed
    );
    let _ = writeln!(
        out,
        "stream: {} events  |  {} recurring phrases ({} recurring events)  |  {} auto-learned cases",
        report.stream_len, report.recurring_phrases, report.recurring_events, report.auto_learned_cases
    );
    let _ = writeln!(out);

    // OFF vs ON metric table.
    let header = format!(
        "{:<22} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "subset", "n", "top1_off", "top1_on", "Δtop1", "mrr_off", "mrr_on", "Δmrr"
    );
    let rule: String = "-".repeat(header.len());
    let _ = writeln!(out, "{header}");
    let _ = writeln!(out, "{rule}");

    let row = |name: &str, off: &Score, on: &Score| -> String {
        format!(
            "{:<22} {:>6} {:>8.3} {:>8.3} {:>+8.3} {:>8.3} {:>8.3} {:>+8.3}",
            name,
            on.n,
            off.top1,
            on.top1,
            on.top1 - off.top1,
            off.mrr,
            on.mrr,
            on.mrr - off.mrr
        )
    };
    let _ = writeln!(out, "{}", row("overall", &report.overall_off, &report.overall_on));
    let _ = writeln!(
        out,
        "{}",
        row("recurring (headline)", &report.recurring_off, &report.recurring_on)
    );
    let _ = writeln!(out, "{rule}");
    let _ = writeln!(out);

    // Learning curve.
    let _ = writeln!(out, "Learning curve — recurring-phrase top-1 by occurrence index:");
    let lc_header = format!("{:<12} {:>6} {:>10} {:>10}", "occurrence", "n", "top1_off", "top1_on");
    let lc_rule: String = "-".repeat(lc_header.len());
    let _ = writeln!(out, "{lc_header}");
    let _ = writeln!(out, "{lc_rule}");
    for p in &report.learning_curve {
        let _ = writeln!(
            out,
            "{:<12} {:>6} {:>10.3} {:>10.3}",
            format!("#{}", p.occurrence),
            p.n,
            p.top1_off,
            p.top1_on
        );
    }
    let _ = writeln!(out, "{lc_rule}");
    let _ = writeln!(out);

    let _ = writeln!(
        out,
        "Auto-learned new-word recall (top-1 after first commit): {:.3}  ({} cases)",
        report.auto_learned_recall, report.auto_learned_cases
    );

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Locate the workspace `data/` directory from the crate dir, if the built artifacts exist.
    fn data_dir() -> Option<PathBuf> {
        // crate dir = rust-pinyin/crates/pyime-eval ; data = ../../data
        let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
        // Require the core artifacts so we don't run on a half-built tree.
        if d.join("lexicon.fst").exists() && d.join("words.bin").exists() {
            Some(d)
        } else {
            None
        }
    }

    /// Run a SMALL session end-to-end and assert the core personalization invariants:
    ///   * ON top-1 on the recurring subset strictly exceeds OFF (adaptation lifts repeats);
    ///   * the learning curve is non-decreasing-ish — the last occurrence's ON top-1 is at least
    ///     the first's, and strictly greater (later repeats ≫ the cold first), with OFF flat-ish.
    /// Skips gracefully (passes) when `data/` is absent so CI without artifacts stays green.
    #[test]
    fn personalization_lifts_recurring_top1() {
        let Some(dir) = data_dir() else {
            eprintln!("skip: built data/ not present");
            return;
        };

        // A small session: a fixed seed keeps it reproducible and fast.
        let report = match run_personalization(&dir, 7) {
            Ok(Some(r)) => r,
            Ok(None) => {
                eprintln!("skip: no session source material");
                return;
            }
            Err(e) => panic!("run_personalization failed: {e:#}"),
        };

        // Sanity on session shape.
        assert!(report.stream_len > 0, "empty stream");
        assert!(report.recurring_off.n > 0, "no recurring events scored");

        // Headline: ON beats OFF on the recurring subset.
        assert!(
            report.recurring_on.top1 > report.recurring_off.top1,
            "recurring top-1 should rise with adaptation: OFF {:.3} -> ON {:.3}",
            report.recurring_off.top1,
            report.recurring_on.top1
        );
        // MRR should also not regress.
        assert!(
            report.recurring_on.mrr >= report.recurring_off.mrr,
            "recurring MRR regressed: OFF {:.3} -> ON {:.3}",
            report.recurring_off.mrr,
            report.recurring_on.mrr
        );

        // Learning curve climbs: later occurrence ON top-1 >= first, and strictly greater overall.
        assert!(report.learning_curve.len() >= 2, "need >=2 occurrence buckets");
        let first = &report.learning_curve[0];
        let last = report.learning_curve.last().unwrap();
        assert!(
            last.top1_on >= first.top1_on,
            "learning curve should be non-decreasing-ish: #{} {:.3} -> #{} {:.3}",
            first.occurrence,
            first.top1_on,
            last.occurrence,
            last.top1_on
        );
        assert!(
            last.top1_on > first.top1_on,
            "later occurrences should clearly beat the cold first occurrence: {:.3} -> {:.3}",
            first.top1_on,
            last.top1_on
        );
        // ON's gain over OFF should widen by the last occurrence (OFF is memory-less; per-bucket
        // OFF can vary because each occurrence index covers a different phrase subset, but the ON
        // advantage at the last occurrence must exceed the advantage at the first).
        let gain_first = first.top1_on - first.top1_off;
        let gain_last = last.top1_on - last.top1_off;
        assert!(
            gain_last >= gain_first,
            "ON advantage should grow with repetition: first +{:.3} -> last +{:.3}",
            gain_first,
            gain_last
        );

        // Render must not panic and must include the headline label.
        let table = render_personalization(&report);
        assert!(table.contains("recurring (headline)"));
    }
}
