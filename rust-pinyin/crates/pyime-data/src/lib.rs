//! pyime-data — data build pipeline (v3, commercial-grade overhaul).
//!
//! Downloads freely-available Chinese pinyin dictionary + corpus data and emits the
//! `data/` artifacts consumed by the engine, matching the DESIGN.md format contract
//! (v1 on-disk layout + v2 additions):
//!   meta.json, words.bin (rkyv Vec<WordEntry>), lexicon.fst, postings.bin,
//!   bigram.fst, trigram.fst, english.fst, plus helper files hanzi_pinyin.tsv,
//!   word_pinyin.tsv and corpus/heldout_sentences.txt.
//!
//! KEY CHANGE (v3): the lexicon is built PRIMARILY from the hand-curated
//! `iDvel/rime-ice` dictionaries (correct polyphone readings + good weights),
//! supplemented by `rime/rime-essay` and rime-ice `tencent` frequencies, and only
//! then back-filled from jieba for coverage. This fixes mangled polyphone readings
//! (了 le/liao, 行 xing/hang, ...) that the old per-char composition produced.
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

/// SIGNED log-ratio LM cost: `round(-LOG_BASE · ln[ p_cond / p_uni ])`, clamped to
/// `±LM_COST_CLAMP`. This is the bonus/penalty of the conditional over the unigram prior; the
/// decoder adds it to the (separately-paid) unigram word-edge cost to recover `-LOG_BASE·ln p_cond`.
/// Negative when context makes the word MORE likely than its prior (a genuine bonus).
#[inline]
fn log_ratio_cost(p_cond: f64, p_uni: f64) -> i32 {
    if p_cond <= 0.0 || p_uni <= 0.0 || !p_cond.is_finite() || !p_uni.is_finite() {
        // Degenerate: no usable conditional signal -> neutral (defer to the unigram edge cost).
        return 0;
    }
    let c = (-LOG_BASE * (p_cond / p_uni).ln()).round();
    c.clamp(-(LM_COST_CLAMP as f64), LM_COST_CLAMP as f64) as i32
}

// ---------------------------------------------------------------------------
// Tuning / pruning knobs (kept well under the size budget).
// ---------------------------------------------------------------------------

/// Keep at most this many dictionary words. rime-ice base has ~400k+ entries; with
/// jieba back-fill this is a safety cap (we keep the highest-frequency words).
const MAX_WORDS: usize = 700_000;
/// Cap on English vocabulary size (frequency-ranked first).
const MAX_ENGLISH: usize = 60_000;
/// Drop word-bigram pairs observed fewer than this many times. With multiple
/// large corpora, low-count pairs are mostly noise that displaces good candidates.
const BIGRAM_MIN_COUNT: u32 = 5;
/// Drop word-trigram triples observed fewer than this many times. The absolute-discounting
/// smoothing keeps rare triples from overfitting, so we can afford a moderate floor (raising it
/// further only loses recall); the log-ratio formulation is what actually tames the noise.
const TRIGRAM_MIN_COUNT: u32 = 8;
/// Drop word-4-gram quadruples observed fewer than this many times. 4-grams explode
/// combinatorially and are far sparser than trigrams, so we use a noticeably higher floor than the
/// trigram: most 4-grams seen 1–5 times are corpus-specific noise. This keeps `fourgram.fst` small
/// (well within the 80 MB total budget) while retaining the genuinely-repeated long contexts.
/// Tuned to 7, deliberately ONE ABOVE `SHOPPING_WEIGHT` (6): a 4-gram seen exactly once in the
/// in-domain shopping corpus contributes weight 6, so a floor of 7 excludes single-shopping-
/// occurrence 4-grams (the noisiest, most overfit tier — there is a huge count==6 spike of them)
/// while keeping any 4-gram with a genuine *second* observation (or ≥7 general-corpus hits). This
/// mirrors the trigram floor sitting just above the single-occurrence tier. Empirically the count
/// distribution has a cliff at 6: floor 6 → ~466k quads / ~9.4 MB FST, floor 7 → ~19k quads /
/// ~0.46 MB FST (total ~51 MB). 7 captures the repeated long contexts without the single-obs noise.
const FOURGRAM_MIN_COUNT: u32 = 7;

/// Absolute-discounting constant `D` for the **bigram** model `P(w3|w2)`. Subtracted from each
/// observed count; the freed mass is redistributed to the unigram via interpolation. ~0.75 is the
/// standard Kneser-Ney/absolute-discounting value and works well empirically here.
const BIGRAM_DISCOUNT: f64 = 0.75;
/// Absolute-discounting constant `D` for the **trigram** model `P(w3|w1,w2)` (back-off to bigram).
const TRIGRAM_DISCOUNT: f64 = 1.0;
/// Absolute-discounting constant `D` for the **4-gram** model `P(w3|w0,w1,w2)` (back-off to the
/// smoothed trigram). Same value as the trigram order: 4-gram counts are small, so a full-unit
/// discount keeps the higher-order term from overfitting the few contexts that survive pruning.
const FOURGRAM_DISCOUNT: f64 = 1.0;
/// Clamp for the SIGNED log-ratio LM costs (transition = `-500·ln[P(w|ctx)/P(w)]`). The ratio is
/// bounded both ways: a hugely-boosted n-gram cannot drop the path by more than this, and a
/// suppressed one cannot inflate it past this. Keeps stored costs in a sane band and the beam
/// well-behaved. ±12000 ≈ a probability ratio of e^24 ≈ 2.6e10, far beyond any real signal.
const LM_COST_CLAMP: i32 = 12_000;
/// Hard cap on the number of bigram pairs kept (highest-count first).
const MAX_BIGRAMS: usize = 2_500_000;
/// Hard cap on the number of trigram triples kept (highest-count first).
const MAX_TRIGRAMS: usize = 3_000_000;
/// Hard cap on the number of 4-gram quadruples kept (highest-count first). Capped lower than the
/// trigram so the 16-byte-key FST stays small and the total `data/` size remains well under the
/// 80 MB budget. ~1.5M keys at 16 bytes + value is a few MB of FST after compression.
const MAX_FOURGRAMS: usize = 1_500_000;
/// Weight (count multiplier) applied to the in-domain shopping corpus when training
/// the LM (the general corpus is much larger; this keeps the in-domain signal alive).
const SHOPPING_WEIGHT: u32 = 6;
/// Number of held-out sentences to reserve for the EVAL agent.
const HELDOUT_SENTENCES: usize = 4_000;

/// Default rime weight when an entry omits one.
const DEFAULT_RIME_WEIGHT: u64 = 1;

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
    eprintln!("[1/8] fetching sources (offline={offline}) ...");
    let read_cached = |name: &str, url: &str| -> Result<Vec<u8>> {
        let p = fetch_to_cache(corpus_dir, name, url, offline)?;
        std::fs::read(&p).with_context(|| format!("read cached {}", p.display()))
    };
    // best-effort fetch: returns empty Vec (and logs) instead of failing the build.
    let try_cached = |name: &str, url: &str, notes: &mut Vec<String>, tag: &str| -> Vec<u8> {
        match read_cached(name, url) {
            Ok(b) => {
                notes.push(format!("{tag}=ok"));
                b
            }
            Err(e) => {
                eprintln!("  WARN {tag} unavailable: {e:#}");
                notes.push(format!("{tag}=UNAVAILABLE"));
                Vec::new()
            }
        }
    };

    // mozillazg single-char + phrase pinyin (fallback reading sources for coverage).
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

    // rime-ice curated dictionaries (PRIMARY reading + weight source).
    let rime_base_raw = try_cached(
        "rime-ice-base.dict.yaml",
        "https://raw.githubusercontent.com/iDvel/rime-ice/main/cn_dicts/base.dict.yaml",
        &mut notes,
        "rime-ice-base",
    );
    let rime_8105_raw = try_cached(
        "rime-ice-8105.dict.yaml",
        "https://raw.githubusercontent.com/iDvel/rime-ice/main/cn_dicts/8105.dict.yaml",
        &mut notes,
        "rime-ice-8105",
    );
    let rime_41448_raw = try_cached(
        "rime-ice-41448.dict.yaml",
        "https://raw.githubusercontent.com/iDvel/rime-ice/main/cn_dicts/41448.dict.yaml",
        &mut notes,
        "rime-ice-41448",
    );
    let rime_others_raw = try_cached(
        "rime-ice-others.dict.yaml",
        "https://raw.githubusercontent.com/iDvel/rime-ice/main/cn_dicts/others.dict.yaml",
        &mut notes,
        "rime-ice-others",
    );
    // tencent: word<TAB>weight (NO pinyin) — frequency supplement only.
    let rime_tencent_raw = try_cached(
        "rime-ice-tencent.dict.yaml",
        "https://raw.githubusercontent.com/iDvel/rime-ice/main/cn_dicts/tencent.dict.yaml",
        &mut notes,
        "rime-ice-tencent",
    );
    // rime-essay: word<TAB>freq — frequency supplement only.
    let essay_raw = try_cached(
        "rime-essay.txt",
        "https://raw.githubusercontent.com/rime/rime-essay/master/essay.txt",
        &mut notes,
        "rime-essay",
    );

    // OpenCC Traditional→Simplified mappings (PRIMARY normalization source).
    // Phrase map applied before char map at ingestion time so 繁体 surfaces
    // collapse onto their 简体 canonical form. Best-effort: an empty map degrades
    // to identity and the build still succeeds.
    let opencc_chars_raw = try_cached(
        "opencc-TSCharacters.txt",
        "https://raw.githubusercontent.com/BYVoid/OpenCC/master/data/dictionary/TSCharacters.txt",
        &mut notes,
        "opencc-TSCharacters",
    );
    let opencc_phrases_raw = try_cached(
        "opencc-TSPhrases.txt",
        "https://raw.githubusercontent.com/BYVoid/OpenCC/master/data/dictionary/TSPhrases.txt",
        &mut notes,
        "opencc-TSPhrases",
    );
    let opencc = pinyin::OpenCc::from_raw(&opencc_chars_raw, &opencc_phrases_raw);
    eprintln!("  OpenCC Traditional->Simplified mappings: {}", opencc.len());
    notes.push(format!("opencc-mappings={}", opencc.len()));

    let english_raw = read_cached(
        "english-words.txt",
        "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt",
    )?;
    let english_freq_raw = match read_cached(
        "google-10000-english.txt",
        "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt",
    ) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("  WARN english frequency list unavailable: {e:#}");
            Vec::new()
        }
    };

    // --- Sentence corpora (for bigram/trigram LM + held-out eval) ------------
    let corpus_toutiao = match fetch_to_cache(
        corpus_dir,
        "toutiao_cat_data.txt",
        "https://raw.githubusercontent.com/skdjfla/toutiao-text-classfication-dataset/master/toutiao_cat_data.txt.zip",
        offline,
    ) {
        Ok(p) => {
            notes.push("general-corpus=toutiao_news_titles".into());
            Some(p)
        }
        Err(e) => {
            eprintln!("  WARN general (toutiao) corpus unavailable: {e:#}");
            notes.push("general-corpus=UNAVAILABLE".into());
            None
        }
    };
    let corpus_csv = match fetch_to_cache(
        corpus_dir,
        "online_shopping_10_cats.csv",
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

    // --- 2. Build the hanzi (single-char) reading table ---------------------
    // Prefer rime-ice 8105/41448 single-char readings (correct), back-fill with
    // mozillazg pinyin-data for any char they miss.
    eprintln!("[2/8] building hanzi reading table ...");
    let mut hanzi: pinyin::HanziTable = pinyin::parse_hanzi_table(&hanzi_raw)?;
    let moz_chars = hanzi.len();
    let mut rime_char_count = 0usize;
    // Priority order for the per-char COMPOSITION reading (used to back-fill jieba
    // words): 8105 is weight-sorted and authoritative → it OVERRIDES mozillazg.
    // 41448 (weight-less, broader) only fills chars 8105 lacks.
    if !rime_8105_raw.is_empty() {
        for (c, py) in pinyin::parse_rime_char_table(&rime_8105_raw)? {
            hanzi.insert(c, py); // 8105 wins
            rime_char_count += 1;
        }
    }
    if !rime_41448_raw.is_empty() {
        for (c, py) in pinyin::parse_rime_char_table(&rime_41448_raw)? {
            hanzi.entry(c).or_insert(py); // only fill missing
            rime_char_count += 1;
        }
    }
    eprintln!(
        "  hanzi readings: {} (mozillazg {} + rime-ice single-char {} merged)",
        hanzi.len(),
        moz_chars,
        rime_char_count
    );
    let phrases = pinyin::parse_phrase_table(&phrase_raw)?;
    eprintln!("  phrase overrides: {}", phrases.len());

    // helper file: data/hanzi_pinyin.tsv (now rime-ice-corrected single-char readings)
    write_hanzi_tsv(out_dir, &hanzi).context("write hanzi_pinyin.tsv")?;

    // --- 3. Build word list (rime-ice PRIMARY, jieba back-fill) -------------
    eprintln!("[3/8] building lexicon (rime-ice primary) ...");
    // rime-ice reading-bearing word sources, in priority order. base (multi-char,
    // weighted), 8105 (single-char, weighted, all polyphone readings), 41448
    // (single-char, broader coverage, weight 1), others (容错/口语 readings).
    let rime_word_sources: [&[u8]; 4] = [
        &rime_base_raw,
        &rime_8105_raw,
        &rime_41448_raw,
        &rime_others_raw,
    ];
    let words = build_words(
        &rime_word_sources,
        &rime_tencent_raw,
        &essay_raw,
        &jieba_raw,
        &hanzi,
        &phrases,
        &opencc,
        &mut notes,
    )?;
    eprintln!(
        "  words kept: {}  readings: {}",
        words.entries.len(),
        words.readings.len()
    );

    // --- 4. Emit words.bin + lexicon.fst + postings.bin ---------------------
    eprintln!("[4/8] writing words.bin / lexicon.fst / postings.bin ...");
    let num_readings = write_lexicon(out_dir, &words)?;

    // helper file: data/word_pinyin.tsv — every word + canonical reading.
    write_word_pinyin_tsv(out_dir, &words).context("write word_pinyin.tsv")?;

    // --- 5. Build + emit bigram.fst + trigram.fst + fourgram.fst ------------
    eprintln!("[5/8] building bigram + trigram + 4-gram LM ...");
    let (num_bigrams, num_trigrams, num_fourgrams, heldout) = build_and_write_ngrams(
        out_dir,
        &words,
        &opencc,
        corpus_toutiao.as_deref(),
        corpus_csv.as_deref(),
    )?;
    eprintln!(
        "  bigrams kept: {num_bigrams}  trigrams kept: {num_trigrams}  fourgrams kept: {num_fourgrams}  heldout: {}",
        heldout.len()
    );

    // helper file: corpus/heldout_sentences.txt
    write_heldout(corpus_dir, &heldout, &words, &mut notes)?;

    // --- 6. Emit english.fst ------------------------------------------------
    eprintln!("[6/8] writing english.fst ...");
    let num_english = write_english(out_dir, &english_raw, &english_freq_raw)?;
    eprintln!("  english terms: {num_english}");

    // --- 7. meta.json -------------------------------------------------------
    eprintln!("[7/8] writing meta.json ...");
    notes.push(format!("english={num_english}"));
    let source_notes = format!(
        "rime-ice(base/8105/41448/others/tencent) + rime-essay + jieba-dict back-fill \
         + mozillazg/pinyin-data + phrase-pinyin-data + dwyl/english-words \
         + google-10000-english + toutiao-news-titles + online-shopping; \
         LM=word bi/tri/4-gram over toutiao+shopping (longest-match tokenization, T->S normalized), \
         absolute-discounting interpolation (bi D={BIGRAM_DISCOUNT}, tri D={TRIGRAM_DISCOUNT}, \
         4-gram D={FOURGRAM_DISCOUNT}), signed log-ratio costs; \
         min-count bi={BIGRAM_MIN_COUNT}/tri={TRIGRAM_MIN_COUNT}/4-gram={FOURGRAM_MIN_COUNT}; {}",
        notes.join(", ")
    );
    write_meta(
        out_dir,
        words.entries.len() as u64,
        num_readings,
        num_bigrams,
        num_trigrams,
        num_fourgrams,
        &source_notes,
    )?;

    // --- 8. done ------------------------------------------------------------
    eprintln!("[8/8] done. data dir: {}", out_dir.display());
    Ok(())
}

// ===========================================================================
// Word list construction (rime-ice primary)
// ===========================================================================

/// One reading bucket -> list of (word_id, reading-specific cost).
struct WordSet {
    /// id -> WordEntry
    entries: Vec<pyime_core::format::WordEntry>,
    /// reading key (e.g. "ni'hao") -> Vec<(word_id, cost)>
    readings: FxHashMap<String, Vec<(u32, u16)>>,
    /// surface -> id, for n-gram counting (most-frequent id per surface).
    surface_to_id: FxHashMap<String, u32>,
    /// id -> canonical reading key, for word_pinyin.tsv export.
    id_reading: Vec<String>,
}

/// Internal merged word record before id assignment.
struct WordRec {
    surface: String,
    reading: String,
    freq: f64,
    pos: u8,
}

/// Parse rime-ice `word<TAB>pinyin[<TAB>weight]` entries, skipping comments and
/// the YAML header (everything up to and incl. the `...` line, plus `#`/`---`).
/// Calls `f(word, reading_key, weight)` for each valid multi-or-single-char entry.
fn for_each_rime_word_entry(raw: &[u8], mut f: impl FnMut(&str, String, u64)) {
    let text = match std::str::from_utf8(raw) {
        Ok(t) => t,
        Err(_) => return,
    };
    for line in text.lines() {
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() || line.starts_with('#') || line.starts_with("---") || line == "..." {
            continue;
        }
        let mut it = line.split('\t');
        let word = match it.next() {
            Some(w) => w.trim(),
            None => continue,
        };
        if word.is_empty() || !pinyin::is_all_cjk(word) {
            continue;
        }
        let py = match it.next() {
            Some(p) => p.trim(),
            None => continue, // no pinyin field (e.g. tencent) — skip here
        };
        let reading = match pinyin::normalize_reading(py) {
            Some(r) => r,
            None => continue,
        };
        // syllable count should match char count for a sane alignment.
        let nsyl = reading.split('\'').count();
        if nsyl != word.chars().count() {
            continue;
        }
        let weight: u64 = it
            .next()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(DEFAULT_RIME_WEIGHT);
        f(word, reading, weight.max(1));
    }
}

/// Parse a `word<TAB>weight` frequency-only file (rime-ice tencent, rime-essay),
/// calling `f(word, weight)` for each CJK-only entry.
fn for_each_freq_entry(raw: &[u8], mut f: impl FnMut(&str, u64)) {
    let text = match std::str::from_utf8(raw) {
        Ok(t) => t,
        Err(_) => return,
    };
    for line in text.lines() {
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() || line.starts_with('#') || line.starts_with("---") || line == "..." {
            continue;
        }
        let mut it = line.split('\t');
        let word = match it.next() {
            Some(w) => w.trim(),
            None => continue,
        };
        if word.is_empty() || !pinyin::is_all_cjk(word) {
            continue;
        }
        let weight: u64 = it.next().and_then(|s| s.trim().parse().ok()).unwrap_or(1);
        f(word, weight.max(1));
    }
}

#[allow(clippy::too_many_arguments)]
fn build_words(
    rime_word_sources: &[&[u8]],
    rime_tencent_raw: &[u8],
    essay_raw: &[u8],
    jieba_raw: &[u8],
    hanzi: &pinyin::HanziTable,
    phrases: &FxHashMap<String, Vec<String>>,
    opencc: &pinyin::OpenCc,
    notes: &mut Vec<String>,
) -> Result<WordSet> {
    // Normalize a raw lexicon surface to Simplified (phrase-then-char). When a
    // traditional surface collapses onto a simplified one with the SAME reading,
    // it merges into the same (surface, reading) bucket below, summing its weight
    // into the simplified entry; the traditional surface disappears entirely.
    // Count how many surfaces actually changed for the build report.
    let mut t2s_changed = 0u64;
    let normalize_surface = |surface: &str, t2s_changed: &mut u64| -> String {
        if opencc.is_empty() {
            return surface.to_string();
        }
        let simp = opencc.convert(surface);
        if simp != surface {
            *t2s_changed += 1;
        }
        simp
    };
    // Merged store keyed by (surface, reading) → (freq, pos).
    // rime-ice readings are authoritative; freqs are merged additively across
    // rime-ice base, others, tencent (freq-only), essay (freq-only).
    let mut merged: FxHashMap<(String, String), (f64, u8)> = FxHashMap::default();
    // Track which surfaces already have a rime-ice reading (so jieba only back-fills).
    let mut rime_surfaces: FxHashMap<String, ()> = FxHashMap::default();
    // For freq-only sources, we need a reading to attach the freq to: use the
    // best (highest-freq) rime reading for that surface seen so far.
    let mut best_reading_for_surface: FxHashMap<String, (String, f64)> = FxHashMap::default();

    let mut rime_word_lines = 0u64;
    let insert_reading = |merged: &mut FxHashMap<(String, String), (f64, u8)>,
                          rime_surfaces: &mut FxHashMap<String, ()>,
                          best: &mut FxHashMap<String, (String, f64)>,
                          surface: &str,
                          reading: String,
                          freq: f64| {
        rime_surfaces.entry(surface.to_string()).or_insert(());
        let e = merged
            .entry((surface.to_string(), reading.clone()))
            .or_insert((0.0, 0));
        e.0 += freq;
        let total = e.0;
        match best.get_mut(surface) {
            Some(b) if b.1 < total => {
                b.0 = reading;
                b.1 = total;
            }
            Some(_) => {}
            None => {
                best.insert(surface.to_string(), (reading, total));
            }
        }
    };

    // (a) rime-ice reading-bearing sources (base + 8105 + 41448 + others) — the
    //     curated readings & weights, incl. ALL single-char polyphone readings.
    for raw in rime_word_sources {
        for_each_rime_word_entry(raw, |word, reading, weight| {
            rime_word_lines += 1;
            // T→S normalize the surface; the reading is shared by 繁/简 and stays.
            let surface = normalize_surface(word, &mut t2s_changed);
            insert_reading(
                &mut merged,
                &mut rime_surfaces,
                &mut best_reading_for_surface,
                &surface,
                reading,
                weight as f64,
            );
        });
    }
    eprintln!("  rime-ice reading entries: {rime_word_lines}");

    // (b) freq-only supplements: rime-ice tencent + rime-essay. Attach freq to the
    //     surface's best rime reading if known; otherwise compose a reading from the
    //     (rime-corrected) hanzi table so the word still gets a usable entry.
    let mut freq_supp_applied = 0u64;
    let mut freq_supp_composed = 0u64;
    let mut apply_freq = |surface_raw: &str, weight: u64, t2s_changed: &mut u64| {
        let surface = normalize_surface(surface_raw, t2s_changed);
        if let Some((reading, _)) = best_reading_for_surface.get(&surface).cloned() {
            let e = merged.entry((surface, reading)).or_insert((0.0, 0));
            e.0 += weight as f64;
            freq_supp_applied += 1;
        } else if let Some(reading) = pinyin::word_to_key(&surface, hanzi, phrases) {
            // compose a reading (correctness still benefits from rime-corrected hanzi)
            let e = merged.entry((surface, reading)).or_insert((0.0, 0));
            e.0 += weight as f64;
            freq_supp_composed += 1;
        }
    };
    for_each_freq_entry(rime_tencent_raw, |w, wt| apply_freq(w, wt, &mut t2s_changed));
    for_each_freq_entry(essay_raw, |w, wt| apply_freq(w, wt, &mut t2s_changed));
    eprintln!(
        "  freq-supplement entries applied: {freq_supp_applied} (composed-reading: {freq_supp_composed})"
    );

    // (c) jieba back-fill: ONLY for surfaces NOT already present from rime-ice.
    //     Compose readings from the (rime-corrected) hanzi/phrase table.
    let jieba_text = std::str::from_utf8(jieba_raw).context("jieba dict utf8")?;
    // jieba freqs are large counts; rescale to be comparable to rime weights. We keep
    // them as-is (additive into a separate, lower-magnitude pool) — they only matter
    // for words rime doesn't have, so absolute scale vs rime is unimportant.
    let mut jieba_added = 0u64;
    for line in jieba_text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let surface_raw = match it.next() {
            Some(s) => s,
            None => continue,
        };
        let freq: u64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let pos_str = it.next().unwrap_or("");
        if freq == 0 || !pinyin::is_all_cjk(surface_raw) {
            continue;
        }
        // T→S normalize before the rime-conflict check and reading composition so
        // a traditional jieba surface either merges into its existing simplified
        // rime entry or composes a (simplified) reading correctly.
        let surface = normalize_surface(surface_raw, &mut t2s_changed);
        if rime_surfaces.contains_key(&surface) {
            continue; // rime-ice wins on conflict
        }
        let reading = match pinyin::word_to_key(&surface, hanzi, phrases) {
            Some(k) => k,
            None => continue,
        };
        let e = merged
            .entry((surface.clone(), reading))
            .or_insert((0.0, 0));
        e.0 += freq as f64;
        e.1 = pinyin::pos_tag(pos_str);
        jieba_added += 1;
    }
    eprintln!("  jieba back-fill (surfaces not in rime-ice): {jieba_added}");
    eprintln!("  T->S normalized surfaces (繁→简 collapses, merged): {t2s_changed}");
    notes.push(format!(
        "lexicon: rime-words={rime_word_lines}, freq-supp={freq_supp_applied}, jieba-backfill={jieba_added}, t2s-normalized={t2s_changed}"
    ));

    // --- materialize: assign ids, compute unigram cost from merged freq ------
    let mut recs: Vec<WordRec> = merged
        .into_iter()
        .map(|((surface, reading), (freq, pos))| WordRec {
            surface,
            reading,
            freq,
            pos,
        })
        .collect();

    // Prune to top-N by frequency if necessary (keep most frequent).
    if recs.len() > MAX_WORDS {
        recs.sort_unstable_by(|a, b| b.freq.partial_cmp(&a.freq).unwrap_or(std::cmp::Ordering::Equal));
        recs.truncate(MAX_WORDS);
    }

    let total_freq: f64 = recs.iter().map(|r| r.freq).sum::<f64>().max(1.0);

    let mut entries: Vec<pyime_core::format::WordEntry> = Vec::with_capacity(recs.len());
    let mut readings: FxHashMap<String, Vec<(u32, u16)>> = FxHashMap::default();
    let mut surface_to_id: FxHashMap<String, u32> = FxHashMap::default();
    let mut surface_best_freq: FxHashMap<String, f64> = FxHashMap::default();
    let mut id_reading: Vec<String> = Vec::with_capacity(recs.len());

    for r in &recs {
        let prob = r.freq / total_freq;
        let cost = prob_to_cost(prob);
        let id = entries.len() as u32;
        entries.push(pyime_core::format::WordEntry {
            surface: r.surface.clone(),
            unigram_cost: cost,
            pos: r.pos,
        });
        id_reading.push(r.reading.clone());
        // surface_to_id: keep the id of the most-frequent reading for a surface.
        match surface_best_freq.get(&r.surface) {
            Some(&f) if f >= r.freq => {}
            _ => {
                surface_best_freq.insert(r.surface.clone(), r.freq);
                surface_to_id.insert(r.surface.clone(), id);
            }
        }
        readings.entry(r.reading.clone()).or_default().push((id, cost));
    }

    // Sort each posting list by cost ascending (best first).
    for v in readings.values_mut() {
        v.sort_unstable_by_key(|&(_, c)| c);
    }

    Ok(WordSet {
        entries,
        readings,
        surface_to_id,
        id_reading,
    })
}

// ===========================================================================
// words.bin + lexicon.fst + postings.bin
// ===========================================================================

fn write_lexicon(out_dir: &Path, words: &WordSet) -> Result<u64> {
    let bytes = rkyv::to_bytes::<_, 1_048_576>(&words.entries)
        .map_err(|e| anyhow::anyhow!("rkyv serialize words: {e}"))?;
    std::fs::write(out_dir.join("words.bin"), &bytes).context("write words.bin")?;

    let mut keys: Vec<&String> = words.readings.keys().collect();
    keys.sort_unstable();

    let mut postings: Vec<u8> = Vec::new();
    let postings_file =
        std::fs::File::create(out_dir.join("lexicon.fst")).context("create lexicon.fst")?;
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

/// data/word_pinyin.tsv: `word<TAB>canonical_reading` for EVERY lexicon word.
fn write_word_pinyin_tsv(out_dir: &Path, words: &WordSet) -> Result<()> {
    let mut buf = String::with_capacity(words.entries.len() * 16);
    for (id, e) in words.entries.iter().enumerate() {
        buf.push_str(&e.surface);
        buf.push('\t');
        buf.push_str(&words.id_reading[id]);
        buf.push('\n');
    }
    std::fs::write(out_dir.join("word_pinyin.tsv"), buf).context("write word_pinyin.tsv")?;
    Ok(())
}

// ===========================================================================
// bigram.fst + trigram.fst
// ===========================================================================

#[allow(clippy::type_complexity)]
fn build_and_write_ngrams(
    out_dir: &Path,
    words: &WordSet,
    opencc: &pinyin::OpenCc,
    corpus_toutiao: Option<&Path>,
    corpus_csv: Option<&Path>,
) -> Result<(u64, u64, u64, Vec<String>)> {
    let seg = Segmenter::new(&words.surface_to_id);
    // Normalize corpus text to Simplified before segmentation so any traditional
    // text in the corpora maps onto the (simplified) vocabulary, instead of failing
    // to segment. Identity when OpenCC is unavailable.
    let t2s = |s: &str| -> String {
        if opencc.is_empty() {
            s.to_string()
        } else {
            opencc.convert(s)
        }
    };

    let mut uni_counts: FxHashMap<u32, u32> = FxHashMap::default();
    let mut bi_counts: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    let mut tri_counts: FxHashMap<(u32, u32, u32), u32> = FxHashMap::default();
    let mut four_counts: FxHashMap<(u32, u32, u32, u32), u32> = FxHashMap::default();
    let mut heldout: Vec<String> = Vec::new();

    let count_sentence =
        |s: &str,
         weight: u32,
         uni: &mut FxHashMap<u32, u32>,
         bi: &mut FxHashMap<(u32, u32), u32>,
         tri: &mut FxHashMap<(u32, u32, u32), u32>,
         four: &mut FxHashMap<(u32, u32, u32, u32), u32>| {
            let ids = seg.segment(s, &words.surface_to_id);
            for &id in &ids {
                *uni.entry(id).or_insert(0) += weight;
            }
            for w in ids.windows(2) {
                *bi.entry((w[0], w[1])).or_insert(0) += weight;
            }
            for w in ids.windows(3) {
                *tri.entry((w[0], w[1], w[2])).or_insert(0) += weight;
            }
            for w in ids.windows(4) {
                *four.entry((w[0], w[1], w[2], w[3])).or_insert(0) += weight;
            }
        };

    let mut used_corpus = false;

    if let Some(tt_path) = corpus_toutiao {
        eprintln!("  reading general corpus {} ...", tt_path.display());
        let sentences = read_toutiao_sentences(tt_path)?;
        eprintln!("  general (toutiao) sentences: {}", sentences.len());
        for s in &sentences {
            let s = t2s(s);
            count_sentence(
                &s,
                1,
                &mut uni_counts,
                &mut bi_counts,
                &mut tri_counts,
                &mut four_counts,
            );
        }
        used_corpus = true;
    }

    if let Some(csv_path) = corpus_csv {
        eprintln!("  reading sentence corpus {} ...", csv_path.display());
        let sentences = read_corpus_sentences(csv_path)?;
        eprintln!("  corpus sentences: {}", sentences.len());
        let n = sentences.len();
        let split = n.saturating_sub(HELDOUT_SENTENCES);
        for (i, s) in sentences.iter().enumerate() {
            // Normalize to Simplified so both the LM counts and the held-out eval
            // sentences are over the (simplified) vocabulary.
            let s = t2s(s);
            if i >= split {
                if heldout.len() < HELDOUT_SENTENCES {
                    heldout.push(s);
                }
                continue;
            }
            count_sentence(
                &s,
                SHOPPING_WEIGHT,
                &mut uni_counts,
                &mut bi_counts,
                &mut tri_counts,
                &mut four_counts,
            );
        }
        used_corpus = true;
    }

    if !used_corpus {
        eprintln!("  FALLBACK: building n-grams from dictionary phrases ...");
        for e in &words.entries {
            if e.surface.chars().count() >= 2 {
                count_sentence(
                    &e.surface,
                    1,
                    &mut uni_counts,
                    &mut bi_counts,
                    &mut tri_counts,
                    &mut four_counts,
                );
            }
        }
    }

    // --- smoothing statistics ----------------------------------------------
    // Absolute discounting + interpolation needs, besides the raw counts:
    //   * total token count           -> unigram P(w) = c1(w)/total
    //   * N1+(w2•)  = #distinct words following w2     (bigram continuation diversity)
    //   * N1+(w1,w2•) = #distinct words following (w1,w2) (trigram continuation diversity)
    // We derive the continuation-diversity maps from the *full* count tables (BEFORE min-count
    // pruning) so the interpolation weights λ reflect the true distribution, not the pruned subset.
    let total_tokens: f64 = uni_counts.values().map(|&c| c as f64).sum::<f64>().max(1.0);
    let mut bi_distinct: FxHashMap<u32, u32> = FxHashMap::default(); // w2 -> #distinct w3
    for &(w2, _w3) in bi_counts.keys() {
        *bi_distinct.entry(w2).or_insert(0) += 1;
    }
    let mut tri_distinct: FxHashMap<(u32, u32), u32> = FxHashMap::default(); // (w1,w2) -> #distinct w3
    for &(w1, w2, _w3) in tri_counts.keys() {
        *tri_distinct.entry((w1, w2)).or_insert(0) += 1;
    }
    // N1+(w0,w1,w2•) = #distinct words following the context (w0,w1,w2) (4-gram continuation
    // diversity), derived from the FULL 4-gram table before min-count pruning.
    let mut four_distinct: FxHashMap<(u32, u32, u32), u32> = FxHashMap::default();
    for &(w0, w1, w2, _w3) in four_counts.keys() {
        *four_distinct.entry((w0, w1, w2)).or_insert(0) += 1;
    }

    // Smoothed unigram probability P(w) = c1(w)/total.
    let uni_p = |w: u32| -> f64 {
        (*uni_counts.get(&w).unwrap_or(&0) as f64) / total_tokens
    };
    // Smoothed bigram P(w3|w2) = max(c2-D,0)/c1(w2) + λ(w2)·P(w3),
    // with λ(w2) = D·N1+(w2•)/c1(w2). Falls back to the unigram when w2 is unseen.
    let bi_p = |w2: u32, w3: u32, c2: f64| -> f64 {
        let c_w2 = *uni_counts.get(&w2).unwrap_or(&0) as f64;
        let p_uni = uni_p(w3);
        if c_w2 <= 0.0 {
            return p_uni;
        }
        let n1 = *bi_distinct.get(&w2).unwrap_or(&0) as f64;
        let lambda = BIGRAM_DISCOUNT * n1 / c_w2;
        ((c2 - BIGRAM_DISCOUNT).max(0.0)) / c_w2 + lambda * p_uni
    };
    // Smoothed trigram P(w3|w1,w2) = max(c3-D,0)/c2(w1,w2) + λ(w1,w2)·P_bigram(w3|w2),
    // with λ(w1,w2) = D·N1+(w1,w2•)/c2(w1,w2). `c3` is the raw count of (w1,w2,w3). Falls back to
    // the smoothed bigram when the context (w1,w2) is unseen. Shared by the trigram emission below
    // AND the 4-gram interpolation (which backs off onto this exact lower-order estimate).
    let tri_p = |w1: u32, w2: u32, w3: u32, c3: f64| -> f64 {
        let c2 = *bi_counts.get(&(w1, w2)).unwrap_or(&0) as f64;
        let p_bi = bi_p(w2, w3, *bi_counts.get(&(w2, w3)).unwrap_or(&0) as f64);
        if c2 <= 0.0 {
            return p_bi;
        }
        let n1 = *tri_distinct.get(&(w1, w2)).unwrap_or(&0) as f64;
        let lambda = TRIGRAM_DISCOUNT * n1 / c2;
        ((c3 - TRIGRAM_DISCOUNT).max(0.0)) / c2 + lambda * p_bi
    };

    // --- bigram.fst (signed log-ratio: -500·ln[P(w3|w2)/P(w3)]) -------------
    let mut pairs: Vec<((u32, u32), u32)> = bi_counts
        .iter()
        .filter(|&(_, &c)| c >= BIGRAM_MIN_COUNT)
        .map(|(&k, &c)| (k, c))
        .collect();
    if pairs.len() > MAX_BIGRAMS {
        eprintln!("  pruning bigrams {} -> {}", pairs.len(), MAX_BIGRAMS);
        pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        pairs.truncate(MAX_BIGRAMS);
    }
    pairs.sort_unstable_by_key(|&((p, i), _)| ((p as u64) << 32) | i as u64);

    let bigram_file =
        std::fs::File::create(out_dir.join("bigram.fst")).context("create bigram.fst")?;
    let mut bb =
        fst::MapBuilder::new(std::io::BufWriter::new(bigram_file)).context("bigram MapBuilder")?;
    let mut n_bigrams: u64 = 0;
    for &((prev, id), count) in &pairs {
        let p_cond = bi_p(prev, id, count as f64);
        let p_uni = uni_p(id);
        let cost = log_ratio_cost(p_cond, p_uni);
        bb.insert(pyime_core::format::bigram_key(prev, id), pyime_core::lm::encode_cost(cost))
            .context("bigram insert")?;
        n_bigrams += 1;
    }
    bb.finish().context("bigram finish")?;

    // --- trigram.fst (signed log-ratio: -500·ln[P(w3|w1,w2)/P(w3)]) --------
    // P(w3|w1,w2) = max(c3-D,0)/c2(w1,w2) + λ(w1,w2)·P_bigram(w3|w2),
    // with λ(w1,w2) = D·N1+(w1,w2•)/c2(w1,w2). The trigram thus *refines* the (already smoothed)
    // bigram on the SAME log-ratio scale, instead of dwarfing it as the old raw-MLE cost did.
    let mut triples: Vec<((u32, u32, u32), u32)> = tri_counts
        .iter()
        .filter(|&(_, &c)| c >= TRIGRAM_MIN_COUNT)
        .map(|(&k, &c)| (k, c))
        .collect();
    if triples.len() > MAX_TRIGRAMS {
        eprintln!("  pruning trigrams {} -> {}", triples.len(), MAX_TRIGRAMS);
        triples.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        triples.truncate(MAX_TRIGRAMS);
    }
    triples.sort_unstable_by_key(|&((a, b, c), _)| {
        // 12-byte BE order == (a,b,c) lexicographic; encode into u128 for the sort key.
        ((a as u128) << 64) | ((b as u128) << 32) | (c as u128)
    });

    let trigram_file =
        std::fs::File::create(out_dir.join("trigram.fst")).context("create trigram.fst")?;
    let mut tb =
        fst::MapBuilder::new(std::io::BufWriter::new(trigram_file)).context("trigram MapBuilder")?;
    let mut n_trigrams: u64 = 0;
    for &((w1, w2, w3), count) in &triples {
        let p_cond = tri_p(w1, w2, w3, count as f64);
        let p_uni = uni_p(w3);
        let cost = log_ratio_cost(p_cond, p_uni);
        tb.insert(pyime_core::format::trigram_key(w1, w2, w3), pyime_core::lm::encode_cost(cost))
            .context("trigram insert")?;
        n_trigrams += 1;
    }
    tb.finish().context("trigram finish")?;

    // --- fourgram.fst (signed log-ratio: -500·ln[P(w3|w0,w1,w2)/P(w3)]) ----
    // P(w3|w0,w1,w2) = max(c4-D,0)/c3(w0,w1,w2) + λ(w0,w1,w2)·P_trigram(w3|w1,w2),
    // with λ(w0,w1,w2) = D·N1+(w0,w1,w2•)/c3(w0,w1,w2). The 4-gram REFINES the (already smoothed)
    // trigram on the SAME log-ratio scale — exactly one order above the trigram, mirroring how the
    // trigram refines the bigram. `c3(w0,w1,w2)` is the raw trigram count of the context.
    let mut quads: Vec<((u32, u32, u32, u32), u32)> = four_counts
        .iter()
        .filter(|&(_, &c)| c >= FOURGRAM_MIN_COUNT)
        .map(|(&k, &c)| (k, c))
        .collect();
    if quads.len() > MAX_FOURGRAMS {
        eprintln!("  pruning fourgrams {} -> {}", quads.len(), MAX_FOURGRAMS);
        quads.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        quads.truncate(MAX_FOURGRAMS);
    }
    quads.sort_unstable_by_key(|&((a, b, c, d), _)| {
        // 16-byte BE order == (a,b,c,d) lexicographic; encode into u128 for the sort key.
        ((a as u128) << 96) | ((b as u128) << 64) | ((c as u128) << 32) | (d as u128)
    });

    let fourgram_file =
        std::fs::File::create(out_dir.join("fourgram.fst")).context("create fourgram.fst")?;
    let mut fb = fst::MapBuilder::new(std::io::BufWriter::new(fourgram_file))
        .context("fourgram MapBuilder")?;
    let mut n_fourgrams: u64 = 0;
    for &((w0, w1, w2, w3), count) in &quads {
        let c3 = *tri_counts.get(&(w0, w1, w2)).unwrap_or(&0) as f64;
        // Lower-order term: the SMOOTHED trigram P(w3|w1,w2) (uses the raw c3(w1,w2,w3) count).
        let p_tri = tri_p(w1, w2, w3, *tri_counts.get(&(w1, w2, w3)).unwrap_or(&0) as f64);
        let p_cond = if c3 > 0.0 {
            let n1 = *four_distinct.get(&(w0, w1, w2)).unwrap_or(&0) as f64;
            let lambda = FOURGRAM_DISCOUNT * n1 / c3;
            ((count as f64 - FOURGRAM_DISCOUNT).max(0.0)) / c3 + lambda * p_tri
        } else {
            p_tri
        };
        let p_uni = uni_p(w3);
        let cost = log_ratio_cost(p_cond, p_uni);
        fb.insert(
            pyime_core::format::fourgram_key(w0, w1, w2, w3),
            pyime_core::lm::encode_cost(cost),
        )
        .context("fourgram insert")?;
        n_fourgrams += 1;
    }
    fb.finish().context("fourgram finish")?;

    Ok((n_bigrams, n_trigrams, n_fourgrams, heldout))
}

/// Read review sentences from the online_shopping CSV (`cat,label,review`).
fn read_corpus_sentences(csv_path: &Path) -> Result<Vec<String>> {
    let raw = std::fs::read_to_string(csv_path).context("read corpus csv")?;
    let mut out = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        if i == 0 {
            continue;
        }
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
        for clause in review.split(is_sentence_split) {
            let clause = clause.trim();
            let cjk = clause.chars().filter(|&c| pinyin::is_cjk(c)).count();
            if cjk >= 2 && clause.chars().count() <= 40 {
                out.push(clause.to_string());
            }
        }
    }
    Ok(out)
}

/// Read general-domain sentences from the Toutiao news-title dataset.
fn read_toutiao_sentences(path: &Path) -> Result<Vec<String>> {
    let raw = std::fs::read_to_string(path).context("read toutiao corpus")?;
    let mut out = Vec::new();
    let push_text = |text: &str, out: &mut Vec<String>| {
        for clause in text.split(is_sentence_split) {
            let clause = clause.trim();
            let cjk = clause.chars().filter(|&c| pinyin::is_cjk(c)).count();
            if cjk >= 2 && clause.chars().count() <= 40 {
                out.push(clause.to_string());
            }
        }
    };
    for line in raw.lines() {
        let mut fields = line.split("_!_");
        let _id = fields.next();
        let _code = fields.next();
        let _cat = fields.next();
        if let Some(title) = fields.next() {
            push_text(title, &mut out);
        }
        if let Some(keywords) = fields.next() {
            push_text(keywords, &mut out);
        }
    }
    Ok(out)
}

/// Shared CJK/ASCII sentence/segment splitter (CJK punctuation + ASCII terminators).
#[inline]
fn is_sentence_split(c: char) -> bool {
    matches!(
        c,
        '。' | '！'
            | '？'
            | '；'
            | '，'
            | '、'
            | '\n'
            | '\r'
            | '\t'
            | '!'
            | '?'
            | ';'
            | ','
            | '：'
            | ':'
            | '“'
            | '”'
            | '‘'
            | '’'
            | '（'
            | '）'
            | '('
            | ')'
            | '《'
            | '》'
            | '【'
            | '】'
            | '|'
            | ' '
            | '~'
            | '—'
            | '…'
    )
}

/// Max-munch segmenter over dictionary surfaces.
struct Segmenter {
    max_len: usize,
}

impl Segmenter {
    fn new(surface_to_id: &FxHashMap<String, u32>) -> Self {
        let max_len = surface_to_id
            .keys()
            .map(|s| s.chars().count())
            .max()
            .unwrap_or(1)
            .min(8);
        Segmenter { max_len }
    }

    /// Greedy longest-match segmentation of a CJK run into known word ids.
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
                i += 1;
            }
        }
        out
    }
}

// ===========================================================================
// english.fst
// ===========================================================================

fn write_english(out_dir: &Path, english_raw: &[u8], english_freq_raw: &[u8]) -> Result<u64> {
    let mut rank: FxHashMap<String, u64> = FxHashMap::default();
    let is_word = |w: &str| !w.is_empty() && w.bytes().all(|b| b.is_ascii_lowercase());

    if let Ok(freq_text) = std::str::from_utf8(english_freq_raw) {
        for (i, line) in freq_text.lines().enumerate() {
            let w = line.trim().to_ascii_lowercase();
            if is_word(&w) {
                rank.entry(w).or_insert(i as u64);
            }
        }
    }
    let freq_count = rank.len() as u64;

    if let Ok(text) = std::str::from_utf8(english_raw) {
        for line in text.lines() {
            let w = line.trim().to_ascii_lowercase();
            if is_word(&w) && !rank.contains_key(&w) {
                let r = freq_count + (w.len() as u64) * 1_000_000;
                rank.insert(w, r);
            }
        }
    }

    let mut ranked: Vec<(u64, String)> = rank.into_iter().map(|(w, r)| (r, w)).collect();
    ranked.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    ranked.truncate(MAX_ENGLISH);
    let mut words: Vec<String> = ranked.into_iter().map(|(_, w)| w).collect();
    words.sort_unstable();
    words.dedup();

    let file = std::fs::File::create(out_dir.join("english.fst")).context("create english.fst")?;
    let mut sb =
        fst::SetBuilder::new(std::io::BufWriter::new(file)).context("english SetBuilder")?;
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
        "trigram.fst",
        "fourgram.fst",
        "english.fst",
        "word_pinyin.tsv",
        "hanzi_pinyin.tsv",
    ];
    files
        .iter()
        .filter_map(|f| std::fs::metadata(out_dir.join(f)).ok())
        .map(|m| m.len())
        .sum()
}

#[allow(clippy::too_many_arguments)]
fn write_meta(
    out_dir: &Path,
    num_words: u64,
    num_readings: u64,
    num_bigrams: u64,
    num_trigrams: u64,
    num_fourgrams: u64,
    source_notes: &str,
) -> Result<()> {
    let bytes_total = dir_data_bytes(out_dir);
    // We extend the core `Meta` JSON with extra `num_trigrams` / `num_fourgrams` keys. The core
    // `Meta` deserializer ignores unknown fields, so this stays format-compatible
    // while exposing the higher-order n-gram counts. Build the JSON object explicitly.
    let json = serde_json::json!({
        "version": pyime_core::format::FORMAT_VERSION,
        "log_base": LOG_BASE as f32,
        "num_words": num_words,
        "num_readings": num_readings,
        "num_bigrams": num_bigrams,
        "num_trigrams": num_trigrams,
        "num_fourgrams": num_fourgrams,
        "bytes_total": bytes_total,
        "source_notes": source_notes,
    });
    let txt = serde_json::to_string_pretty(&json).context("serialize meta")?;
    std::fs::write(out_dir.join("meta.json"), txt).context("write meta.json")?;
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
    let meta: pyime_core::format::Meta =
        serde_json::from_str(&meta_txt).context("parse meta.json")?;
    // also pull num_trigrams out of the raw JSON (not in the core Meta struct).
    let raw_meta: serde_json::Value = serde_json::from_str(&meta_txt).unwrap_or_default();
    let meta_trigrams = raw_meta
        .get("num_trigrams")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let meta_fourgrams = raw_meta
        .get("num_fourgrams")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let words_bytes = std::fs::read(out_dir.join("words.bin")).context("read words.bin")?;
    let archived = rkyv::check_archived_root::<Vec<pyime_core::format::WordEntry>>(&words_bytes)
        .map_err(|e| anyhow::anyhow!("validate words.bin: {e}"))?;
    let num_words = archived.len();

    let lex_bytes = std::fs::read(out_dir.join("lexicon.fst")).context("read lexicon.fst")?;
    let lex = Map::new(lex_bytes).context("open lexicon.fst")?;
    let num_readings = lex.len() as u64;

    let postings = std::fs::read(out_dir.join("postings.bin")).context("read postings.bin")?;

    let bi_bytes = std::fs::read(out_dir.join("bigram.fst")).context("read bigram.fst")?;
    let bi = Map::new(bi_bytes).context("open bigram.fst")?;
    let num_bigrams = bi.len() as u64;

    // trigram.fst (NEW)
    let tri_bytes = std::fs::read(out_dir.join("trigram.fst")).context("read trigram.fst")?;
    let tri = Map::new(tri_bytes).context("open trigram.fst")?;
    let num_trigrams = tri.len() as u64;

    // fourgram.fst (NEW)
    let four_bytes = std::fs::read(out_dir.join("fourgram.fst")).context("read fourgram.fst")?;
    let four = Map::new(four_bytes).context("open fourgram.fst")?;
    let num_fourgrams = four.len() as u64;

    let en_bytes = std::fs::read(out_dir.join("english.fst")).context("read english.fst")?;
    let en = Set::new(en_bytes).context("open english.fst")?;
    let num_english = en.len();

    // word_pinyin.tsv (NEW): count lines, sanity-check tab structure.
    let wp_txt =
        std::fs::read_to_string(out_dir.join("word_pinyin.tsv")).context("read word_pinyin.tsv")?;
    let mut wp_lines = 0u64;
    for line in wp_txt.lines() {
        if line.is_empty() {
            continue;
        }
        anyhow::ensure!(
            line.split('\t').count() == 2,
            "word_pinyin.tsv malformed line: {line}"
        );
        wp_lines += 1;
    }

    let bytes_total = dir_data_bytes(out_dir);

    eprintln!("=== verify {} ===", out_dir.display());
    eprintln!("  format version : {}", meta.version);
    eprintln!("  log_base       : {}", meta.log_base);
    eprintln!("  words.bin      : {num_words} entries (meta {})", meta.num_words);
    eprintln!("  lexicon.fst    : {num_readings} readings (meta {})", meta.num_readings);
    eprintln!("  postings.bin   : {} bytes", postings.len());
    eprintln!("  bigram.fst     : {num_bigrams} pairs (meta {})", meta.num_bigrams);
    eprintln!("  trigram.fst    : {num_trigrams} triples (meta {meta_trigrams})");
    eprintln!("  fourgram.fst   : {num_fourgrams} quads (meta {meta_fourgrams})");
    eprintln!("  english.fst    : {num_english} terms");
    eprintln!("  word_pinyin.tsv: {wp_lines} entries");
    eprintln!("  bytes_total    : {bytes_total} ({:.2} MB)", bytes_total as f64 / 1e6);

    anyhow::ensure!(num_words as u64 == meta.num_words, "words count mismatch");
    anyhow::ensure!(num_readings == meta.num_readings, "readings count mismatch");
    anyhow::ensure!(num_bigrams == meta.num_bigrams, "bigram count mismatch");
    anyhow::ensure!(num_trigrams == meta_trigrams, "trigram count mismatch");
    anyhow::ensure!(num_fourgrams == meta_fourgrams, "fourgram count mismatch");
    anyhow::ensure!(wp_lines == num_words as u64, "word_pinyin.tsv count mismatch");
    anyhow::ensure!(!postings.is_empty(), "postings.bin empty");

    if let Some((kbytes, off)) = lex.stream_first() {
        let off = off as usize;
        anyhow::ensure!(off + 2 <= postings.len(), "posting offset OOB");
        let n = u16::from_le_bytes([postings[off], postings[off + 1]]) as usize;
        anyhow::ensure!(off + 2 + n * 6 <= postings.len(), "posting record OOB");
        let key = String::from_utf8_lossy(&kbytes);
        eprintln!("  spot-check key '{key}' -> {n} candidate(s)");
    }

    // trigram spot-check: first key decodes to a valid 12-byte structure.
    if let Some((kbytes, _)) = tri.stream_first() {
        anyhow::ensure!(kbytes.len() == 12, "trigram key not 12 bytes");
    }

    // fourgram spot-check: keys are 16 bytes and values decode to the signed cost band.
    if let Some((kbytes, val)) = four.stream_first() {
        anyhow::ensure!(kbytes.len() == 16, "fourgram key not 16 bytes");
        let cost = pyime_core::lm::decode_cost(val);
        anyhow::ensure!(
            cost.unsigned_abs() <= LM_COST_CLAMP as u32,
            "fourgram cost {cost} outside ±{LM_COST_CLAMP} band"
        );
    }

    Ok(meta)
}

// Small fst helper: first key+value of a Map.
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

    #[test]
    fn opencc_traditional_to_simplified() {
        // char-level: trad -> simp; phrase-level applied first.
        let chars = "這\t这\n個\t个\n中\t中\n國\t国\n了\t了\n".as_bytes();
        // phrase that overrides a naive char mapping (不瞭解 -> 不了解).
        let phrases = "不瞭解\t不了解\n".as_bytes();
        let cc = pinyin::OpenCc::from_raw(chars, phrases);
        assert_eq!(cc.convert("這個"), "这个");
        assert_eq!(cc.convert("中國"), "中国");
        // already simplified -> unchanged
        assert_eq!(cc.convert("这个"), "这个");
        // phrase wins over per-char
        assert_eq!(cc.convert("不瞭解"), "不了解");
        // mixed / passthrough non-mapped chars
        assert_eq!(cc.convert("a這b"), "a这b");
        // self-mapping entries are dropped (中->中 not stored) but identity holds
        assert!(!cc.is_empty());
    }

    #[test]
    fn rime_reading_normalization() {
        assert_eq!(
            pinyin::normalize_reading("zhong guo").as_deref(),
            Some("zhong'guo")
        );
        assert_eq!(pinyin::normalize_reading("nǐ hǎo").as_deref(), Some("ni'hao"));
        assert_eq!(pinyin::normalize_reading("  ").as_deref(), None);
    }

    /// If a built data dir exists, confirm key curated readings are correct.
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
        check("wo'shi", "我是");
        check("zhong'guo", "中国");
    }
}
