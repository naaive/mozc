//! pyime-eval — comprehensive, quantified evaluation harness.
//!
//! Implements:
//!   * gold-set generation (delegated to [`gold`]) from a held-out sentence corpus,
//!   * per-bucket and overall accuracy/ranking metrics (Top-1/5/10, MRR, Char-accuracy,
//!     CER, Coverage),
//!   * latency (mean/p50/p95/p99), throughput, and process-memory (RSS) measurements,
//!   * a human-readable table ([`render_table`]) plus a serializable [`EvalReport`].
//!
//! See DESIGN.md "Evaluation system".

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;

use pyime_core::{Engine, EngineConfig};

pub mod gold;

/// A single gold evaluation case. Serialized one-per-line as JSONL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldCase {
    /// Scenario bucket (see [`gold::BUCKETS`]).
    pub bucket: String,
    /// Raw input fed to the engine.
    pub input: String,
    /// Expected output (gold answer).
    pub expected: String,
}

/// Accuracy/ranking metrics for one bucket (or the overall aggregate).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketMetrics {
    pub bucket: String,
    /// Number of cases scored in this bucket.
    pub n: usize,
    /// Fraction where the top-1 candidate exactly matches the expected output.
    pub top1: f64,
    /// Fraction where the expected output appears in the top-5 candidates.
    pub top5: f64,
    /// Fraction where the expected output appears in the top-10 candidates.
    pub top10: f64,
    /// Mean reciprocal rank of the expected output (0 if not found).
    pub mrr: f64,
    /// Mean character-level accuracy of top-1 (1 - normalized Levenshtein).
    pub char_acc: f64,
    /// Mean character error rate of top-1 (normalized Levenshtein).
    pub cer: f64,
    /// Fraction where the expected output appears anywhere in the candidate list.
    pub coverage: f64,
}

/// Latency statistics over all single-call conversions (milliseconds).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    /// Conversions per second (1000 / mean_ms).
    pub throughput_per_s: f64,
    /// Total conversions timed.
    pub samples: usize,
}

/// Memory / on-disk footprint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Resident set size after loading the engine, in bytes (from /proc/self/status VmRSS).
    pub rss_bytes: u64,
    /// Total size of the `data/` directory on disk, in bytes.
    pub data_disk_bytes: u64,
}

/// The full evaluation report: per-bucket metrics, the overall aggregate, plus
/// latency and memory summaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    /// One entry per bucket, in [`gold::BUCKETS`] order (only buckets with cases).
    pub buckets: Vec<BucketMetrics>,
    /// Aggregate over every scored case.
    pub overall: BucketMetrics,
    pub latency: LatencyStats,
    pub memory: MemoryStats,
}

/// Generate the gold set from the corpus + hanzi-pinyin table, writing JSONL to `out`.
///
/// Delegates to [`gold::generate`] with a default per-bucket cap that keeps eval fast.
pub fn generate_gold(corpus: &Path, hanzi_pinyin: &Path, out: &Path, seed: u64) -> Result<()> {
    gold::generate(corpus, hanzi_pinyin, out, seed, 600)
}

/// Generate the gold set with an explicit per-bucket cap (used by tests for a tiny set).
pub fn generate_gold_capped(
    corpus: &Path,
    hanzi_pinyin: &Path,
    out: &Path,
    seed: u64,
    per_bucket: usize,
) -> Result<()> {
    gold::generate(corpus, hanzi_pinyin, out, seed, per_bucket)
}

/// Generate the FAIR gold set using correct word readings via longest-match tokenization over
/// the curated lexicon (`word_pinyin.tsv`), falling back to the per-char `hanzi_pinyin.tsv`
/// table for chars absent from the lexicon. Writes JSONL (e.g. `data/gold_v2.jsonl`) to `out`.
///
/// Unlike [`generate_gold`], this never mis-reads polyphones / 词组 (e.g. 提高→`tigao`,
/// 系统→`xitong`, 重要→`zhongyao`), so the engine is scored on pinyin a user would actually type.
pub fn generate_gold_correct(
    corpus: &Path,
    word_pinyin: &Path,
    hanzi_pinyin: &Path,
    out: &Path,
    seed: u64,
) -> Result<()> {
    gold::generate_correct(corpus, word_pinyin, hanzi_pinyin, out, seed, 600)
}

/// Like [`generate_gold_correct`] but with an explicit per-bucket cap (used by tests).
pub fn generate_gold_correct_capped(
    corpus: &Path,
    word_pinyin: &Path,
    hanzi_pinyin: &Path,
    out: &Path,
    seed: u64,
    per_bucket: usize,
) -> Result<()> {
    gold::generate_correct(corpus, word_pinyin, hanzi_pinyin, out, seed, per_bucket)
}

// ---------------------------------------------------------------------------
// String comparison helpers
// ---------------------------------------------------------------------------

/// Normalize for comparison: drop ASCII whitespace (spaces, tabs). Matching ignores spaces.
fn normalize(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Levenshtein edit distance over Unicode scalar values.
fn levenshtein(a: &[char], b: &[char]) -> usize {
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur: Vec<usize> = vec![0; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Normalized Levenshtein (0 = identical, 1 = maximally different).
fn norm_levenshtein(a: &str, b: &str) -> f64 {
    let ac: Vec<char> = a.chars().collect();
    let bc: Vec<char> = b.chars().collect();
    let denom = ac.len().max(bc.len());
    if denom == 0 {
        return 0.0;
    }
    levenshtein(&ac, &bc) as f64 / denom as f64
}

// ---------------------------------------------------------------------------
// Per-case scoring
// ---------------------------------------------------------------------------

/// Accumulator of summed metric values for a set of cases.
#[derive(Default, Clone)]
struct Acc {
    n: usize,
    top1: f64,
    top5: f64,
    top10: f64,
    mrr: f64,
    char_acc: f64,
    cer: f64,
    coverage: f64,
}

impl Acc {
    fn add(&mut self, top1_hit: f64, mrr: f64, in5: f64, in10: f64, char_acc: f64, cer: f64, cov: f64) {
        self.n += 1;
        self.top1 += top1_hit;
        self.top5 += in5;
        self.top10 += in10;
        self.mrr += mrr;
        self.char_acc += char_acc;
        self.cer += cer;
        self.coverage += cov;
    }
    fn finish(&self, bucket: &str) -> BucketMetrics {
        let d = if self.n == 0 { 1.0 } else { self.n as f64 };
        BucketMetrics {
            bucket: bucket.to_string(),
            n: self.n,
            top1: self.top1 / d,
            top5: self.top5 / d,
            top10: self.top10 / d,
            mrr: self.mrr / d,
            char_acc: self.char_acc / d,
            cer: self.cer / d,
            coverage: self.coverage / d,
        }
    }
}

/// Score one case against an ordered candidate list, returning the per-case metric tuple.
/// `cand_texts` are the candidate output strings, best-first.
fn score_case(expected: &str, cand_texts: &[String]) -> (f64, f64, f64, f64, f64, f64, f64) {
    let exp = normalize(expected);
    // Rank of the first candidate that matches the expected (1-based), if any.
    let mut rank: Option<usize> = None;
    for (i, c) in cand_texts.iter().enumerate() {
        if normalize(c) == exp {
            rank = Some(i + 1);
            break;
        }
    }
    let top1_hit = if rank == Some(1) { 1.0 } else { 0.0 };
    let in5 = if matches!(rank, Some(r) if r <= 5) { 1.0 } else { 0.0 };
    let in10 = if matches!(rank, Some(r) if r <= 10) { 1.0 } else { 0.0 };
    let mrr = rank.map(|r| 1.0 / r as f64).unwrap_or(0.0);
    let coverage = if rank.is_some() { 1.0 } else { 0.0 };

    // Char-level metrics use the top-1 candidate (empty if none).
    let top1_text = cand_texts.first().map(|s| normalize(s)).unwrap_or_default();
    let cer = norm_levenshtein(&top1_text, &exp);
    let char_acc = 1.0 - cer;
    (top1_hit, mrr, in5, in10, char_acc, cer, coverage)
}

// ---------------------------------------------------------------------------
// System measurements
// ---------------------------------------------------------------------------

/// Read VmRSS (resident set size) from /proc/self/status, in bytes. 0 if unavailable.
fn read_vmrss_bytes() -> u64 {
    let status = match std::fs::read_to_string("/proc/self/status") {
        Ok(s) => s,
        Err(_) => return 0,
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            // Format: "VmRSS:\t   12345 kB"
            let mut it = rest.split_whitespace();
            if let Some(num) = it.next() {
                if let Ok(kb) = num.parse::<u64>() {
                    return kb * 1024;
                }
            }
        }
    }
    0
}

/// Total size of all regular files directly in `dir`, in bytes (non-recursive is fine for data/).
fn dir_size_bytes(dir: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            if let Ok(md) = entry.metadata() {
                if md.is_file() {
                    total += md.len();
                }
            }
        }
    }
    total
}

/// Percentile (nearest-rank) of a sorted slice of f64. p in [0,100].
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (p / 100.0 * sorted.len() as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(sorted.len() - 1);
    sorted[idx]
}

// ---------------------------------------------------------------------------
// Main eval entry point
// ---------------------------------------------------------------------------

/// Run the full evaluation: load the gold set, convert every case (timing each single call),
/// and aggregate per-bucket and overall metrics plus latency/memory.
///
/// `data/` size on disk is inferred from the parent directory of `gold` if it is `data/gold.jsonl`,
/// else from a `data` sibling; falls back to 0.
pub fn run_eval(engine: &Engine, cfg: &EngineConfig, gold: &Path) -> Result<EvalReport> {
    let cases = gold::load(gold).with_context(|| format!("load gold set {}", gold.display()))?;

    // Per-bucket accumulators, keyed in BUCKETS order; plus overall.
    let mut accs: Vec<(String, Acc)> = gold::BUCKETS
        .iter()
        .map(|b| (b.to_string(), Acc::default()))
        .collect();
    let mut overall = Acc::default();
    let mut latencies_ms: Vec<f64> = Vec::with_capacity(cases.len());

    for case in &cases {
        let t0 = Instant::now();
        let cands = engine.convert(&case.input, cfg);
        let dt = t0.elapsed();
        latencies_ms.push(dt.as_secs_f64() * 1000.0);

        let texts: Vec<String> = cands.iter().map(|c| c.text.clone()).collect();
        let (top1, mrr, in5, in10, char_acc, cer, cov) = score_case(&case.expected, &texts);

        if let Some((_, acc)) = accs.iter_mut().find(|(b, _)| b == &case.bucket) {
            acc.add(top1, mrr, in5, in10, char_acc, cer, cov);
        }
        overall.add(top1, mrr, in5, in10, char_acc, cer, cov);
    }

    let buckets: Vec<BucketMetrics> = accs
        .iter()
        .filter(|(_, a)| a.n > 0)
        .map(|(name, a)| a.finish(name))
        .collect();
    let overall_metrics = overall.finish("OVERALL");

    // Latency stats.
    let mut sorted = latencies_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean_ms = if latencies_ms.is_empty() {
        0.0
    } else {
        latencies_ms.iter().sum::<f64>() / latencies_ms.len() as f64
    };
    let latency = LatencyStats {
        mean_ms,
        p50_ms: percentile(&sorted, 50.0),
        p95_ms: percentile(&sorted, 95.0),
        p99_ms: percentile(&sorted, 99.0),
        throughput_per_s: if mean_ms > 0.0 { 1000.0 / mean_ms } else { 0.0 },
        samples: latencies_ms.len(),
    };

    // Memory: RSS now (engine already loaded), data dir size.
    let data_dir = gold.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| Path::new("data").to_path_buf());
    let memory = MemoryStats {
        rss_bytes: read_vmrss_bytes(),
        data_disk_bytes: dir_size_bytes(&data_dir),
    };

    Ok(EvalReport {
        buckets,
        overall: overall_metrics,
        latency,
        memory,
    })
}

// ---------------------------------------------------------------------------
// Human-readable rendering
// ---------------------------------------------------------------------------

/// Render the report as a clean fixed-width table: bucket rows × metric cols, an OVERALL row,
/// and latency/memory summary lines.
pub fn render_table(report: &EvalReport) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();

    let header = format!(
        "{:<14} {:>6} {:>7} {:>7} {:>7} {:>7} {:>8} {:>7} {:>8}",
        "bucket", "n", "top1", "top5", "top10", "mrr", "char_acc", "cer", "coverage"
    );
    let rule: String = "-".repeat(header.len());
    let _ = writeln!(out, "{header}");
    let _ = writeln!(out, "{rule}");

    let row = |m: &BucketMetrics| -> String {
        format!(
            "{:<14} {:>6} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>8.3} {:>7.3} {:>8.3}",
            m.bucket, m.n, m.top1, m.top5, m.top10, m.mrr, m.char_acc, m.cer, m.coverage
        )
    };

    for m in &report.buckets {
        let _ = writeln!(out, "{}", row(m));
    }
    let _ = writeln!(out, "{rule}");
    let _ = writeln!(out, "{}", row(&report.overall));
    let _ = writeln!(out, "{rule}");

    let l = &report.latency;
    let _ = writeln!(
        out,
        "Latency (ms): mean {:.3}  p50 {:.3}  p95 {:.3}  p99 {:.3}   |   Throughput: {:.0} conv/s  (n={})",
        l.mean_ms, l.p50_ms, l.p95_ms, l.p99_ms, l.throughput_per_s, l.samples
    );

    let m = &report.memory;
    let _ = writeln!(
        out,
        "Memory: RSS {:.1} MiB   |   data/ on disk {:.1} MiB",
        m.rss_bytes as f64 / (1024.0 * 1024.0),
        m.data_disk_bytes as f64 / (1024.0 * 1024.0)
    );

    out
}
