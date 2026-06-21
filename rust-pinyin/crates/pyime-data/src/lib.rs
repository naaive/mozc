//! pyime-data — data build pipeline.
//!
//! Downloads freely-available Chinese pinyin dictionary + corpus data and emits the
//! `data/` artifacts consumed by the engine, EXACTLY matching the DESIGN.md
//! "Data format contract (v1)":
//!   meta.json, words.bin (rkyv Vec<WordEntry>), lexicon.fst, postings.bin,
//!   bigram.fst, english.fst, plus helper files hanzi_pinyin.tsv and
//!   heldout_sentences.txt.
//!
//! Public entry points:
//!   - [`build_all`]  — download (with caching) + build everything.
//!   - [`verify`]     — load the produced data back and report counts + bytes.

use anyhow::{Context, Result};
use rustc_hash::FxHashMap;
use std::io::Write as _;
use std::path::Path;

mod download;
mod pinyin;

pub use download::fetch_to_cache;

/// LOG_BASE from DESIGN.md (Mozc-like cost scaling).
pub const LOG_BASE: f64 = 500.0;
/// u16 cost clamp ceiling (DESIGN: clamp 0..=60000).
pub const COST_MAX: u16 = 60000;

/// Convert a probability into an integer cost: `round(-LOG_BASE * ln(prob))`, clamped.
#[inline]
pub fn prob_to_cost(prob: f64) -> u16 {
    if prob <= 0.0 {
        return COST_MAX;
    }
    let c = (-LOG_BASE * prob.ln()).round();
    if c <= 0.0 {
        0
    } else if c >= COST_MAX as f64 {
        COST_MAX
    } else {
        c as u16
    }
}

// ---------------------------------------------------------------------------
// Tuning / pruning knobs (kept well under the size budget).
// ---------------------------------------------------------------------------

/// Keep at most this many dictionary words (after CJK filtering). The jieba dict
/// has ~350k entries; after pruning non-CJK it is far fewer, so this is a safety cap.
const MAX_WORDS: usize = 600_000;
/// Cap on English vocabulary size (most-common / shortest first).
const MAX_ENGLISH: usize = 50_000;
/// Drop word-bigram pairs observed fewer than this many times.
const BIGRAM_MIN_COUNT: u32 = 2;
/// Number of held-out sentences to reserve for the EVAL agent.
const HELDOUT_SENTENCES: usize = 4_000;

/// Build everything: download sources into `corpus_dir` (cache) and write artifacts
/// into `out_dir`. `offline` skips network and requires the cache to be populated.
pub fn build_all(out_dir: &Path, corpus_dir: &Path) -> Result<()> {
    build_all_opts(out_dir, corpus_dir, false)
}

pub fn build_all_opts(out_dir: &Path, corpus_dir: &Path, offline: bool) -> Result<()> {
    std::fs::create_dir_all(out_dir).context("create out dir")?;
    std::fs::create_dir_all(corpus_dir).context("create corpus dir")?;

    let mut notes: Vec<String> = Vec::new();

    // --- 1. Download sources (cached) ---------------------------------------
    eprintln!("[1/7] fetching sources (offline={offline}) ...");
    let read_cached = |name: &str, url: &str| -> Result<Vec<u8>> {
        let p = fetch_to_cache(corpus_dir, name, url, offline)?;
        std::fs::read(&p).with_context(|| format!("read cached {}", p.display()))
    };
    let hanzi_raw = read_cached(
        "pinyin-data.txt",
        "https://raw.githubusercontent.com/mozillazg/pinyin-data/master/pinyin.txt",
    )?;
    let phrase_raw = read_cached(
        "phrase-pinyin-data.txt",
        "https://raw.githubusercontent.com/mozillazg/phrase-pinyin-data/master/pinyin.txt",
    )?;
    let jieba_raw = read_cached(
        "jieba-dict.txt",
        "https://raw.githubusercontent.com/fxsjy/jieba/master/jieba/dict.txt",
    )?;
    let english_raw = read_cached(
        "english-words.txt",
        "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt",
    )?;

    // Sentence corpus (for bigram LM + held-out eval). Best-effort; we fall back
    // to dictionary phrases if it is unreachable.
    let corpus_csv = match fetch_to_cache(
        corpus_dir,
        "online_shopping_10_cats.csv",
        // The repo ships a zip; we cache the *extracted* csv. download.rs handles
        // ".csv from .zip" by extension heuristics below.
        "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/online_shopping_10_cats/online_shopping_10_cats.zip",
        offline,
    ) {
        Ok(p) => {
            notes.push("sentence-corpus=online_shopping_10_cats".into());
            Some(p)
        }
        Err(e) => {
            eprintln!("  WARN sentence corpus unavailable: {e:#}");
            notes.push("sentence-corpus=FALLBACK(phrase-dict)".into());
            None
        }
    };

    // --- 2. Parse hanzi + phrase pinyin tables ------------------------------
    eprintln!("[2/7] parsing pinyin tables ...");
    let hanzi: pinyin::HanziTable = pinyin::parse_hanzi_table(&hanzi_raw)?;
    eprintln!("  hanzi first-readings: {}", hanzi.len());
    let phrases = pinyin::parse_phrase_table(&phrase_raw)?;
    eprintln!("  phrase overrides: {}", phrases.len());

    // helper file: data/hanzi_pinyin.tsv
    write_hanzi_tsv(out_dir, &hanzi).context("write hanzi_pinyin.tsv")?;

    // --- 3. Build word list from jieba dict ---------------------------------
    eprintln!("[3/7] building word list ...");
    let words = build_words(&jieba_raw, &hanzi, &phrases)?;
    eprintln!("  words kept: {}  readings: {}", words.entries.len(), words.readings.len());

    // --- 4. Emit words.bin + lexicon.fst + postings.bin ---------------------
    eprintln!("[4/7] writing words.bin / lexicon.fst / postings.bin ...");
    let num_readings = write_lexicon(out_dir, &words)?;

    // --- 5. Build + emit bigram.fst -----------------------------------------
    eprintln!("[5/7] building bigram LM ...");
    let (num_bigrams, heldout) = build_and_write_bigram(out_dir, corpus_dir, &words, corpus_csv.as_deref())?;
    eprintln!("  bigrams kept: {num_bigrams}  heldout sentences: {}", heldout.len());

    // helper file: corpus/heldout_sentences.txt
    write_heldout(corpus_dir, &heldout, &words, &mut notes)?;

    // --- 6. Emit english.fst ------------------------------------------------
    eprintln!("[6/7] writing english.fst ...");
    let num_english = write_english(out_dir, &english_raw)?;
    eprintln!("  english terms: {num_english}");

    // --- 7. meta.json -------------------------------------------------------
    eprintln!("[7/7] writing meta.json ...");
    notes.push(format!("english={num_english}"));
    let source_notes = format!(
        "jieba-dict + mozillazg/pinyin-data + phrase-pinyin-data + dwyl/english-words; {}",
        notes.join(", ")
    );
    write_meta(
        out_dir,
        words.entries.len() as u64,
        num_readings,
        num_bigrams,
        &source_notes,
    )?;

    eprintln!("done. data dir: {}", out_dir.display());
    Ok(())
}

// ===========================================================================
// Word list construction
// ===========================================================================

/// One reading bucket -> list of (word_id, reading-specific cost).
struct WordSet {
    /// id -> WordEntry
    entries: Vec<pyime_core::format::WordEntry>,
    /// reading key (e.g. "ni'hao") -> Vec<(word_id, cost)>
    readings: FxHashMap<String, Vec<(u32, u16)>>,
    /// surface -> id, for bigram counting
    surface_to_id: FxHashMap<String, u32>,
}

fn build_words(
    jieba_raw: &[u8],
    hanzi: &pinyin::HanziTable,
    phrases: &FxHashMap<String, Vec<String>>,
) -> Result<WordSet> {
    let text = std::str::from_utf8(jieba_raw).context("jieba dict utf8")?;

    // First pass: collect (surface, freq) for valid CJK words.
    struct Raw {
        surface: String,
        freq: u64,
        pos: u8,
        key: String,
    }
    let mut raws: Vec<Raw> = Vec::new();
    let mut total_freq: u64 = 0;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let surface = match it.next() {
            Some(s) => s,
            None => continue,
        };
        let freq: u64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let pos_str = it.next().unwrap_or("");

        // Keep only words whose chars are all CJK (skip ascii / mixed / punctuation).
        if !pinyin::is_all_cjk(surface) {
            continue;
        }
        let key = match pinyin::word_to_key(surface, hanzi, phrases) {
            Some(k) => k,
            None => continue, // every char lacked a reading
        };
        if freq == 0 {
            continue;
        }
        total_freq += freq;
        raws.push(Raw {
            surface: surface.to_string(),
            freq,
            pos: pinyin::pos_tag(pos_str),
            key,
        });
    }

    // Prune to top-N by frequency if needed.
    if raws.len() > MAX_WORDS {
        raws.sort_unstable_by(|a, b| b.freq.cmp(&a.freq));
        raws.truncate(MAX_WORDS);
    }

    let total_freq = total_freq.max(1) as f64;

    let mut entries: Vec<pyime_core::format::WordEntry> = Vec::with_capacity(raws.len());
    let mut readings: FxHashMap<String, Vec<(u32, u16)>> = FxHashMap::default();
    let mut surface_to_id: FxHashMap<String, u32> = FxHashMap::default();

    for r in &raws {
        let prob = r.freq as f64 / total_freq;
        let cost = prob_to_cost(prob);
        let id = entries.len() as u32;
        entries.push(pyime_core::format::WordEntry {
            surface: r.surface.clone(),
            unigram_cost: cost,
            pos: r.pos,
        });
        // surface_to_id: keep the most-frequent id for a surface (raws may have dups)
        surface_to_id.entry(r.surface.clone()).or_insert(id);
        readings.entry(r.key.clone()).or_default().push((id, cost));
    }

    // Sort each posting list by cost ascending (best first) for nicer decoding.
    for v in readings.values_mut() {
        v.sort_unstable_by_key(|&(_, c)| c);
    }

    Ok(WordSet {
        entries,
        readings,
        surface_to_id,
    })
}

// ===========================================================================
// words.bin + lexicon.fst + postings.bin
// ===========================================================================

fn write_lexicon(out_dir: &Path, words: &WordSet) -> Result<u64> {
    // words.bin — rkyv Vec<WordEntry>
    let bytes = rkyv::to_bytes::<_, 1_048_576>(&words.entries)
        .map_err(|e| anyhow::anyhow!("rkyv serialize words: {e}"))?;
    std::fs::write(out_dir.join("words.bin"), &bytes).context("write words.bin")?;

    // postings.bin + lexicon.fst.
    // fst::Map requires keys inserted in lexicographic order.
    let mut keys: Vec<&String> = words.readings.keys().collect();
    keys.sort_unstable();

    let mut postings: Vec<u8> = Vec::new();
    let postings_file = std::fs::File::create(out_dir.join("lexicon.fst")).context("create lexicon.fst")?;
    let wtr = std::io::BufWriter::new(postings_file);
    let mut map_builder = fst::MapBuilder::new(wtr).context("fst MapBuilder")?;

    for key in keys {
        let offset = postings.len() as u64;
        let list = &words.readings[key];
        let n: u16 = list.len().min(u16::MAX as usize) as u16;
        postings.extend_from_slice(&n.to_le_bytes());
        for &(word_id, cost) in list.iter().take(n as usize) {
            postings.extend_from_slice(&word_id.to_le_bytes());
            postings.extend_from_slice(&cost.to_le_bytes());
        }
        map_builder
            .insert(key.as_bytes(), offset)
            .with_context(|| format!("fst insert key {key}"))?;
    }
    map_builder.finish().context("fst finish")?;
    std::fs::write(out_dir.join("postings.bin"), &postings).context("write postings.bin")?;

    Ok(words.readings.len() as u64)
}

// ===========================================================================
// bigram.fst
// ===========================================================================

fn build_and_write_bigram(
    out_dir: &Path,
    corpus_dir: &Path,
    words: &WordSet,
    corpus_csv: Option<&Path>,
) -> Result<(u64, Vec<String>)> {
    // Build a max-munch segmenter trie keyed on surfaces present in the dict.
    let seg = Segmenter::new(&words.surface_to_id);

    // counts: (prev_id, id) -> count, and unigram counts for normalization.
    let mut bi_counts: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    let mut uni_counts: FxHashMap<u32, u32> = FxHashMap::default();
    let mut heldout: Vec<String> = Vec::new();

    let count_sentence = |s: &str,
                          bi: &mut FxHashMap<(u32, u32), u32>,
                          uni: &mut FxHashMap<u32, u32>| {
        let ids = seg.segment(s, &words.surface_to_id);
        for &id in &ids {
            *uni.entry(id).or_insert(0) += 1;
        }
        for w in ids.windows(2) {
            *bi.entry((w[0], w[1])).or_insert(0) += 1;
        }
    };

    let mut used_corpus = false;
    if let Some(csv_path) = corpus_csv {
        eprintln!("  reading sentence corpus {} ...", csv_path.display());
        let sentences = read_corpus_sentences(csv_path)?;
        eprintln!("  corpus sentences: {}", sentences.len());
        // Reserve a held-out tail for eval; train on the rest.
        let n = sentences.len();
        let split = n.saturating_sub(HELDOUT_SENTENCES);
        for (i, s) in sentences.iter().enumerate() {
            if i >= split {
                if heldout.len() < HELDOUT_SENTENCES {
                    heldout.push(s.clone());
                }
                continue;
            }
            count_sentence(s, &mut bi_counts, &mut uni_counts);
        }
        used_corpus = true;
    }

    // Fallback / augmentation: derive bigrams from multi-word phrase readings by
    // segmenting each phrase-pinyin surface as a "sentence". This guarantees a
    // non-trivial bigram.fst even with no sentence corpus.
    if !used_corpus {
        eprintln!("  FALLBACK: building bigrams from dictionary phrases ...");
        for e in &words.entries {
            // Multi-char surfaces only — segment them into sub-words.
            if e.surface.chars().count() >= 2 {
                count_sentence(&e.surface, &mut bi_counts, &mut uni_counts);
            }
        }
    }

    // Prune low-count pairs.
    let mut pairs: Vec<((u32, u32), u32)> = bi_counts
        .into_iter()
        .filter(|&(_, c)| c >= BIGRAM_MIN_COUNT)
        .collect();
    // fst keys must be inserted in sorted order of the 8-byte BE key.
    pairs.sort_unstable_by_key(|&((p, i), _)| ((p as u64) << 32) | i as u64);

    let bigram_file = std::fs::File::create(out_dir.join("bigram.fst")).context("create bigram.fst")?;
    let wtr = std::io::BufWriter::new(bigram_file);
    let mut bb = fst::MapBuilder::new(wtr).context("bigram MapBuilder")?;
    let mut n_bigrams: u64 = 0;
    for ((prev, id), count) in &pairs {
        // P(id | prev) = count(prev,id) / count(prev)
        let denom = *uni_counts.get(prev).unwrap_or(&0) as f64;
        let prob = if denom > 0.0 { *count as f64 / denom } else { 0.0 };
        let cost = prob_to_cost(prob) as u64;
        let key = pyime_core::format::bigram_key(*prev, *id);
        bb.insert(key, cost).context("bigram insert")?;
        n_bigrams += 1;
    }
    bb.finish().context("bigram finish")?;

    let _ = corpus_dir;
    Ok((n_bigrams, heldout))
}

/// Read review sentences from the online_shopping CSV (`cat,label,review`).
/// We split multi-clause reviews on Chinese punctuation into shorter sentences.
fn read_corpus_sentences(csv_path: &Path) -> Result<Vec<String>> {
    let raw = std::fs::read_to_string(csv_path).context("read corpus csv")?;
    let mut out = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        if i == 0 {
            continue; // header
        }
        // review is everything after the 2nd comma (reviews contain no commas in
        // this dataset's escaping; if they do, the tail still forms valid text).
        let mut parts = line.splitn(3, ',');
        let _cat = parts.next();
        let _label = parts.next();
        let review = match parts.next() {
            Some(r) => r.trim().trim_start_matches('\u{feff}'),
            None => continue,
        };
        if review.is_empty() {
            continue;
        }
        // Split into clause-sized sentences on common CJK terminators.
        for clause in review.split(|c| {
            matches!(
                c,
                '。' | '！' | '？' | '；' | '，' | '、' | '\n' | '!' | '?' | ';' | ','
            )
        }) {
            let clause = clause.trim();
            // Keep clauses with at least 2 CJK chars and not absurdly long.
            let cjk = clause.chars().filter(|&c| pinyin::is_cjk(c)).count();
            if cjk >= 2 && clause.chars().count() <= 40 {
                out.push(clause.to_string());
            }
        }
    }
    Ok(out)
}

/// Max-munch segmenter over dictionary surfaces.
struct Segmenter {
    /// max surface char-length, to bound the munch window.
    max_len: usize,
}

impl Segmenter {
    fn new(surface_to_id: &FxHashMap<String, u32>) -> Self {
        let max_len = surface_to_id
            .keys()
            .map(|s| s.chars().count())
            .max()
            .unwrap_or(1)
            .min(8); // bound to keep segmentation cheap
        Segmenter { max_len }
    }

    /// Greedy longest-match segmentation of a CJK run into known word ids.
    /// Unknown single chars are skipped (do not contribute to bigrams).
    fn segment(&self, s: &str, surface_to_id: &FxHashMap<String, u32>) -> Vec<u32> {
        let chars: Vec<char> = s.chars().filter(|&c| pinyin::is_cjk(c)).collect();
        let mut out = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let mut matched = false;
            let upper = (i + self.max_len).min(chars.len());
            let mut j = upper;
            while j > i {
                let cand: String = chars[i..j].iter().collect();
                if let Some(&id) = surface_to_id.get(&cand) {
                    out.push(id);
                    i = j;
                    matched = true;
                    break;
                }
                j -= 1;
            }
            if !matched {
                i += 1; // skip unknown char
            }
        }
        out
    }
}

// ===========================================================================
// english.fst
// ===========================================================================

fn write_english(out_dir: &Path, english_raw: &[u8]) -> Result<u64> {
    let text = std::str::from_utf8(english_raw).context("english words utf8")?;
    // Lowercase, ascii-alpha only. Prefer shorter (more common) words when capping.
    let mut words: Vec<String> = text
        .lines()
        .map(|l| l.trim().to_ascii_lowercase())
        .filter(|w| !w.is_empty() && w.bytes().all(|b| b.is_ascii_lowercase()))
        .collect();
    words.sort_unstable();
    words.dedup();

    if words.len() > MAX_ENGLISH {
        // Keep the shortest words (proxy for "most common"), then re-sort lexically
        // because fst::Set requires sorted insertion.
        let mut by_len = words;
        by_len.sort_unstable_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
        by_len.truncate(MAX_ENGLISH);
        by_len.sort_unstable();
        words = by_len;
    }

    let file = std::fs::File::create(out_dir.join("english.fst")).context("create english.fst")?;
    let wtr = std::io::BufWriter::new(file);
    let mut sb = fst::SetBuilder::new(wtr).context("english SetBuilder")?;
    for w in &words {
        sb.insert(w).context("english insert")?;
    }
    sb.finish().context("english finish")?;
    Ok(words.len() as u64)
}

// ===========================================================================
// meta.json + helper files
// ===========================================================================

fn dir_data_bytes(out_dir: &Path) -> u64 {
    let files = [
        "words.bin",
        "lexicon.fst",
        "postings.bin",
        "bigram.fst",
        "english.fst",
    ];
    files
        .iter()
        .filter_map(|f| std::fs::metadata(out_dir.join(f)).ok())
        .map(|m| m.len())
        .sum()
}

fn write_meta(
    out_dir: &Path,
    num_words: u64,
    num_readings: u64,
    num_bigrams: u64,
    source_notes: &str,
) -> Result<()> {
    // bytes_total counts the core binary artifacts (meta itself excluded).
    let bytes_total = dir_data_bytes(out_dir);
    let meta = pyime_core::format::Meta {
        version: pyime_core::format::FORMAT_VERSION,
        log_base: LOG_BASE as f32,
        num_words,
        num_readings,
        num_bigrams,
        bytes_total,
        source_notes: source_notes.to_string(),
    };
    let json = serde_json::to_string_pretty(&meta).context("serialize meta")?;
    std::fs::write(out_dir.join("meta.json"), json).context("write meta.json")?;
    Ok(())
}

fn write_hanzi_tsv(out_dir: &Path, hanzi: &pinyin::HanziTable) -> Result<()> {
    let mut buf = String::new();
    let mut keys: Vec<(&char, &String)> = hanzi.iter().collect();
    keys.sort_unstable_by_key(|(c, _)| **c as u32);
    for (c, py) in keys {
        buf.push(*c);
        buf.push('\t');
        buf.push_str(py);
        buf.push('\n');
    }
    std::fs::write(out_dir.join("hanzi_pinyin.tsv"), buf).context("write hanzi_pinyin.tsv")?;
    Ok(())
}

fn write_heldout(
    corpus_dir: &Path,
    heldout: &[String],
    words: &WordSet,
    notes: &mut Vec<String>,
) -> Result<()> {
    let path = corpus_dir.join("heldout_sentences.txt");
    let mut file = std::fs::File::create(&path).context("create heldout_sentences.txt")?;
    if !heldout.is_empty() {
        for s in heldout {
            writeln!(file, "{s}")?;
        }
    } else {
        // Fallback: sample multi-char dictionary phrases as pseudo-sentences.
        notes.push("heldout=FALLBACK(dict-phrases)".into());
        let mut count = 0;
        for e in &words.entries {
            if e.surface.chars().count() >= 3 {
                writeln!(file, "{}", e.surface)?;
                count += 1;
                if count >= HELDOUT_SENTENCES {
                    break;
                }
            }
        }
    }
    Ok(())
}

// ===========================================================================
// verify — load produced data back and report counts + bytes
// ===========================================================================

/// Loads the produced data set back and prints counts + total bytes. Returns the
/// loaded [`pyime_core::format::Meta`] for programmatic checks.
pub fn verify(out_dir: &Path) -> Result<pyime_core::format::Meta> {
    use fst::{Map, Set};

    let meta_txt = std::fs::read_to_string(out_dir.join("meta.json")).context("read meta.json")?;
    let meta: pyime_core::format::Meta = serde_json::from_str(&meta_txt).context("parse meta.json")?;

    // words.bin
    let words_bytes = std::fs::read(out_dir.join("words.bin")).context("read words.bin")?;
    let archived = rkyv::check_archived_root::<Vec<pyime_core::format::WordEntry>>(&words_bytes)
        .map_err(|e| anyhow::anyhow!("validate words.bin: {e}"))?;
    let num_words = archived.len();

    // lexicon.fst
    let lex_bytes = std::fs::read(out_dir.join("lexicon.fst")).context("read lexicon.fst")?;
    let lex = Map::new(lex_bytes).context("open lexicon.fst")?;
    let num_readings = lex.len() as u64;

    // postings.bin sanity: read one posting at a sampled key offset.
    let postings = std::fs::read(out_dir.join("postings.bin")).context("read postings.bin")?;

    // bigram.fst
    let bi_bytes = std::fs::read(out_dir.join("bigram.fst")).context("read bigram.fst")?;
    let bi = Map::new(bi_bytes).context("open bigram.fst")?;
    let num_bigrams = bi.len() as u64;

    // english.fst
    let en_bytes = std::fs::read(out_dir.join("english.fst")).context("read english.fst")?;
    let en = Set::new(en_bytes).context("open english.fst")?;
    let num_english = en.len();

    let bytes_total = dir_data_bytes(out_dir);

    eprintln!("=== verify {} ===", out_dir.display());
    eprintln!("  format version : {}", meta.version);
    eprintln!("  log_base       : {}", meta.log_base);
    eprintln!("  words.bin      : {num_words} entries (meta {})", meta.num_words);
    eprintln!("  lexicon.fst    : {num_readings} readings (meta {})", meta.num_readings);
    eprintln!("  postings.bin   : {} bytes", postings.len());
    eprintln!("  bigram.fst     : {num_bigrams} pairs (meta {})", meta.num_bigrams);
    eprintln!("  english.fst    : {num_english} terms");
    eprintln!("  bytes_total    : {bytes_total} ({:.2} MB)", bytes_total as f64 / 1e6);

    anyhow::ensure!(num_words as u64 == meta.num_words, "words count mismatch");
    anyhow::ensure!(num_readings == meta.num_readings, "readings count mismatch");
    anyhow::ensure!(num_bigrams == meta.num_bigrams, "bigram count mismatch");
    anyhow::ensure!(!postings.is_empty(), "postings.bin empty");

    // Spot-check: first key in lexicon resolves to a valid posting.
    if let Some((kbytes, off)) = lex.stream_first() {
        let off = off as usize;
        anyhow::ensure!(off + 2 <= postings.len(), "posting offset OOB");
        let n = u16::from_le_bytes([postings[off], postings[off + 1]]) as usize;
        anyhow::ensure!(off + 2 + n * 6 <= postings.len(), "posting record OOB");
        let key = String::from_utf8_lossy(&kbytes);
        eprintln!("  spot-check key '{key}' -> {n} candidate(s)");
    }

    Ok(meta)
}

// Small fst helper: first key+value of a Map (fst 0.4 has no direct accessor).
trait MapFirst {
    fn stream_first(&self) -> Option<(Vec<u8>, u64)>;
}
impl<D: AsRef<[u8]>> MapFirst for fst::Map<D> {
    fn stream_first(&self) -> Option<(Vec<u8>, u64)> {
        use fst::Streamer;
        let mut s = self.stream();
        s.next().map(|(k, v)| (k.to_vec(), v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prob_cost_monotonic() {
        assert!(prob_to_cost(0.5) < prob_to_cost(0.01));
        assert_eq!(prob_to_cost(0.0), COST_MAX);
        assert_eq!(prob_to_cost(1.0), 0);
    }

    #[test]
    fn syllable_normalization() {
        assert_eq!(pinyin::normalize_syllable("nǐ"), "ni");
        assert_eq!(pinyin::normalize_syllable("lǜ"), "lv");
        assert_eq!(pinyin::normalize_syllable("hǎo"), "hao");
        assert_eq!(pinyin::normalize_syllable("zhong4"), "zhong");
    }

    /// If a built data dir exists, query the lexicon for known readings and confirm
    /// the expected surfaces are present in the posting list.
    #[test]
    fn lexicon_roundtrip_if_built() {
        let dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data"));
        if !dir.join("lexicon.fst").exists() {
            eprintln!("(skipping: no built data at {})", dir.display());
            return;
        }
        let lex = fst::Map::new(std::fs::read(dir.join("lexicon.fst")).unwrap()).unwrap();
        let postings = std::fs::read(dir.join("postings.bin")).unwrap();
        let words_bytes = std::fs::read(dir.join("words.bin")).unwrap();
        let words =
            rkyv::check_archived_root::<Vec<pyime_core::format::WordEntry>>(&words_bytes).unwrap();

        let check = |key: &str, want: &str| {
            let off = lex
                .get(key)
                .unwrap_or_else(|| panic!("key {key} missing")) as usize;
            let n = u16::from_le_bytes([postings[off], postings[off + 1]]) as usize;
            let mut found = false;
            for i in 0..n {
                let base = off + 2 + i * 6;
                let id = u32::from_le_bytes([
                    postings[base],
                    postings[base + 1],
                    postings[base + 2],
                    postings[base + 3],
                ]) as usize;
                if words[id].surface.as_str() == want {
                    found = true;
                    break;
                }
            }
            assert!(found, "expected surface {want} under reading {key}");
        };
        check("ni'hao", "你好");
        check("bei'jing", "北京");
        check("chong'qing", "重庆"); // phrase-override disambiguation (not zhong'qing)
    }
}
