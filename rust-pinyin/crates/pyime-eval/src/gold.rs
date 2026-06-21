//! Gold-set generation from a held-out Chinese sentence corpus.
//!
//! Two conversion paths turn each held-out sentence into the canonical pinyin a user would type:
//!
//!   * **per-character** ([`generate`]): one syllable per Han char from the `hanzi_pinyin.tsv`
//!     first-reading table. Cheap, but wrong for polyphones / 词组 — e.g. 提高 becomes `digao`
//!     (提's first reading is `di`) instead of the real `tigao`, 系统 becomes `jitong` not
//!     `xitong`, 重要 becomes `chongyao` not `zhongyao`. The engine is then unfairly penalized
//!     for "failing" on pinyin nobody would ever type.
//!
//!   * **longest-match word tokenization** ([`generate_correct`]): greedily segment the sentence
//!     into the longest known words from the curated lexicon (`word_pinyin.tsv`) and concatenate
//!     THEIR canonical readings (correct, including polyphones), falling back to the per-char
//!     table only for chars absent from the lexicon. This yields the pinyin a real user types,
//!     so the gold set is *fair*.
//!
//! Both paths feed the same bucket synthesis (`full`, `abbr`, `fuzzy`, `typo`, `english`,
//! `mixed`, `long_sentence`, `short_word`) so the two gold sets are directly comparable.
//!
//! Reproducibility: a seeded xorshift RNG drives every stochastic choice (sampling, fuzzy
//! selection, typo injection). Regenerating with the same seed yields the same gold set.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::GoldCase;

/// Tiny dependency-free xorshift64* RNG (deterministic, seedable).
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid the zero fixed-point of xorshift.
        Rng { state: seed ^ 0x9E37_79B9_7F4A_7C15 | 1 }
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [0, n). n must be > 0.
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }
    /// True with probability p.
    #[inline]
    pub fn chance(&mut self, p: f64) -> bool {
        (self.next_u64() as f64 / u64::MAX as f64) < p
    }
    /// Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}

/// All scenario buckets we generate.
pub const BUCKETS: &[&str] = &[
    "full",
    "abbr",
    "fuzzy",
    "typo",
    "english",
    "mixed",
    "long_sentence",
    "short_word",
];

/// Load the `汉<TAB>pinyin` first-reading table.
fn load_hanzi_pinyin(path: &Path) -> Result<HashMap<char, String>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut map = HashMap::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        let mut it = line.splitn(2, '\t');
        let (Some(han), Some(py)) = (it.next(), it.next()) else { continue };
        let mut chars = han.chars();
        if let (Some(c), None) = (chars.next(), chars.next()) {
            // tone-less, ü→v already per contract; lowercase ascii expected
            map.entry(c).or_insert_with(|| py.trim().to_string());
        }
    }
    Ok(map)
}

/// A valid canonical syllable string: non-empty, ascii-lowercase only.
fn syllable_ok(py: &str) -> bool {
    !py.is_empty() && py.bytes().all(|b| b.is_ascii_lowercase())
}

/// Convert a Chinese phrase to per-CHARACTER pinyin (one first-reading syllable per Han char).
/// Returns None if any Han char is missing from the table (so we skip such sentences).
fn phrase_to_syllables(phrase: &str, table: &HashMap<char, String>) -> Option<Vec<String>> {
    let mut syls = Vec::new();
    for c in phrase.chars() {
        // Only handle CJK-ish chars; reject if not found.
        let py = table.get(&c)?;
        if !syllable_ok(py) {
            return None;
        }
        syls.push(py.clone());
    }
    if syls.is_empty() {
        return None;
    }
    Some(syls)
}

/// The curated word→reading lexicon for longest-match tokenization.
///
/// `words[k]` maps a `k`-character word (Han chars) to its canonical reading, split into
/// syllables. We index by char-length so the greedy matcher can probe longest-first without
/// scanning the whole map. `max_len` is the longest word length present.
pub struct Lexicon {
    /// `by_len[k]` = words of `(k+1)` chars (so `by_len[0]` is single chars), value = syllables.
    by_len: Vec<HashMap<String, Vec<String>>>,
    max_len: usize,
}

impl Lexicon {
    /// Load `word<TAB>reading` (reading = syllables joined by `'`). The FIRST reading seen for a
    /// word wins (the file lists the canonical/most-frequent reading first). Words whose reading
    /// has a non-ascii-lowercase syllable are skipped.
    pub fn load(path: &Path) -> Result<Lexicon> {
        let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
        // Cap word length we index; lexicons rarely exceed ~12 chars and very long "words"
        // (idioms/sentences) only hurt the greedy segmentation.
        const MAX_INDEXED_LEN: usize = 12;
        let mut by_len: Vec<HashMap<String, Vec<String>>> =
            (0..MAX_INDEXED_LEN).map(|_| HashMap::new()).collect();
        let mut max_len = 0usize;
        for line in BufReader::new(f).lines() {
            let line = line?;
            let mut it = line.splitn(2, '\t');
            let (Some(word), Some(reading)) = (it.next(), it.next()) else { continue };
            let nchars = word.chars().count();
            if nchars == 0 || nchars > MAX_INDEXED_LEN {
                continue;
            }
            let syls: Vec<String> = reading.trim().split('\'').map(|s| s.to_string()).collect();
            if syls.is_empty() || syls.iter().any(|s| !syllable_ok(s)) {
                continue;
            }
            // The reading must have exactly one syllable per character to be usable for
            // initials/abbr alignment; skip mismatches (rare, e.g. erhua collapses).
            if syls.len() != nchars {
                continue;
            }
            let slot = &mut by_len[nchars - 1];
            slot.entry(word.to_string()).or_insert(syls);
            if nchars > max_len {
                max_len = nchars;
            }
        }
        Ok(Lexicon { by_len, max_len })
    }

    /// Look up a word (by its char slice already materialized into a String of `len` chars).
    fn get(&self, word: &str, len: usize) -> Option<&Vec<String>> {
        if len == 0 || len > self.by_len.len() {
            return None;
        }
        self.by_len[len - 1].get(word)
    }
}

/// Longest-match tokenization of a phrase over the lexicon, concatenating each matched word's
/// CANONICAL reading (correct polyphones), falling back to the per-char `table` for any char not
/// covered by a lexicon word. Returns None if a fallback char is also missing from the table
/// (so we skip such sentences, exactly like the per-char path).
fn phrase_to_syllables_correct(
    phrase: &str,
    lex: &Lexicon,
    table: &HashMap<char, String>,
) -> Option<Vec<String>> {
    let chars: Vec<char> = phrase.chars().collect();
    let mut syls: Vec<String> = Vec::with_capacity(chars.len());
    let mut i = 0usize;
    while i < chars.len() {
        // Try the longest word starting at i.
        let max_try = lex.max_len.min(chars.len() - i);
        let mut matched = false;
        for len in (2..=max_try).rev() {
            let candidate: String = chars[i..i + len].iter().collect();
            if let Some(reading) = lex.get(&candidate, len) {
                syls.extend(reading.iter().cloned());
                i += len;
                matched = true;
                break;
            }
        }
        if matched {
            continue;
        }
        // Single char: prefer a single-char lexicon reading, else the per-char table.
        let one: String = chars[i].to_string();
        if let Some(reading) = lex.get(&one, 1) {
            syls.extend(reading.iter().cloned());
        } else {
            let py = table.get(&chars[i])?;
            if !syllable_ok(py) {
                return None;
            }
            syls.push(py.clone());
        }
        i += 1;
    }
    if syls.is_empty() {
        return None;
    }
    Some(syls)
}

/// Apply one or two fuzzy substitutions to a list of syllables, returning the fuzzed
/// concatenated pinyin. Returns None if no applicable rule fires.
fn apply_fuzzy(syls: &[String], rng: &mut Rng) -> Option<String> {
    // (find prefix/suffix, replacement) candidate transforms, applied to one syllable each.
    // Initial swaps then final swaps; these mirror pyime-core fuzzy rules.
    const INIT: &[(&str, &str)] = &[("zh", "z"), ("ch", "c"), ("sh", "s"), ("n", "l"), ("f", "h"), ("r", "l")];
    const FIN: &[(&str, &str)] = &[("ang", "an"), ("eng", "en"), ("ing", "in"), ("iang", "ian"), ("uang", "uan")];

    let mut out: Vec<String> = syls.to_vec();
    // Build a list of (syllable_index, kind, from, to) applicable edits.
    let mut edits: Vec<(usize, bool, &str, &str)> = Vec::new();
    for (i, s) in out.iter().enumerate() {
        for (a, b) in INIT {
            if s.starts_with(a) {
                edits.push((i, true, a, b));
            }
        }
        for (a, b) in FIN {
            if s.ends_with(a) && s.len() > a.len() {
                edits.push((i, false, a, b));
            }
        }
    }
    if edits.is_empty() {
        return None;
    }
    rng.shuffle(&mut edits);
    let n_apply = if edits.len() >= 2 && rng.chance(0.5) { 2 } else { 1 };
    let mut applied_idx: Vec<usize> = Vec::new();
    for &(i, is_init, a, b) in edits.iter() {
        if applied_idx.contains(&i) {
            continue; // one edit per syllable
        }
        let s = &out[i];
        let news = if is_init {
            format!("{}{}", b, &s[a.len()..])
        } else {
            format!("{}{}", &s[..s.len() - a.len()], b)
        };
        out[i] = news;
        applied_idx.push(i);
        if applied_idx.len() >= n_apply {
            break;
        }
    }
    Some(out.concat())
}

/// Inject exactly one edit (transposition / drop / adjacent-key substitution) into a string.
fn inject_typo(full: &str, rng: &mut Rng) -> Option<String> {
    let bytes: Vec<u8> = full.bytes().collect();
    if bytes.len() < 2 {
        return None;
    }
    // QWERTY adjacency for fat-finger substitution.
    const ADJ: &[(u8, &[u8])] = &[
        (b'a', b"sqzw"), (b'b', b"vghn"), (b'c', b"xdfv"), (b'd', b"serfcx"),
        (b'e', b"wrsdf"), (b'f', b"drtgvc"), (b'g', b"ftyhbv"), (b'h', b"gyujnb"),
        (b'i', b"ujko"), (b'j', b"huikmn"), (b'k', b"jiolm"), (b'l', b"kop"),
        (b'm', b"njk"), (b'n', b"bhjm"), (b'o', b"iklp"), (b'p', b"ol"),
        (b'q', b"wa"), (b'r', b"edft"), (b's', b"awedxz"), (b't', b"rfgy"),
        (b'u', b"yhij"), (b'v', b"cfgb"), (b'w', b"qase"), (b'x', b"zsdc"),
        (b'y', b"tghu"), (b'z', b"asx"),
    ];
    let mut b = bytes.clone();
    match rng.below(3) {
        0 => {
            // transposition of two adjacent distinct chars
            let mut tries = 0;
            loop {
                let i = rng.below(b.len() - 1);
                if b[i] != b[i + 1] {
                    b.swap(i, i + 1);
                    break;
                }
                tries += 1;
                if tries > 8 {
                    b.swap(i, i + 1); // give up; harmless if equal (no-op avoided below)
                    if b[i] == b[i + 1] {
                        return drop_one(&bytes, rng);
                    }
                    break;
                }
            }
        }
        1 => return drop_one(&bytes, rng),
        _ => {
            // adjacent-key substitution
            let i = rng.below(b.len());
            let lookup = ADJ.iter().find(|(c, _)| *c == b[i]).map(|(_, n)| *n);
            if let Some(neigh) = lookup {
                if !neigh.is_empty() {
                    b[i] = neigh[rng.below(neigh.len())];
                }
            } else {
                return drop_one(&bytes, rng);
            }
        }
    }
    let s = String::from_utf8(b).ok()?;
    if s == full {
        None
    } else {
        Some(s)
    }
}

fn drop_one(bytes: &[u8], rng: &mut Rng) -> Option<String> {
    let i = rng.below(bytes.len());
    let mut b = bytes.to_vec();
    b.remove(i);
    String::from_utf8(b).ok()
}

/// A small built-in English word list for the `english` and `mixed` buckets.
pub const ENGLISH_WORDS: &[&str] = &[
    "github", "hello", "world", "rust", "python", "computer", "internet", "google",
    "windows", "linux", "code", "data", "model", "search", "engine", "design",
    "system", "network", "server", "client", "email", "video", "music", "phone",
    "apple", "android", "browser", "keyboard", "mouse", "screen", "online", "office",
    "project", "version", "update", "download", "upload", "message", "account", "password",
];

/// Read non-empty trimmed lines from the corpus file.
fn load_sentences(corpus: &Path) -> Result<Vec<String>> {
    let f = File::open(corpus).with_context(|| format!("open {}", corpus.display()))?;
    let mut sentences = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        let t = line.trim();
        if !t.is_empty() {
            sentences.push(t.to_string());
        }
    }
    Ok(sentences)
}

/// Generate the gold set using the PER-CHARACTER first-reading table and write JSONL to `out`.
///
/// This is the original (v1) path; it can mis-read polyphones/词组 (see module docs). Kept for
/// comparison with the fair [`generate_correct`] path.
pub fn generate(
    corpus: &Path,
    hanzi_pinyin: &Path,
    out: &Path,
    seed: u64,
    per_bucket: usize,
) -> Result<()> {
    let table = load_hanzi_pinyin(hanzi_pinyin)?;
    let mut sentences = load_sentences(corpus)?;

    let mut rng = Rng::new(seed);
    // Deterministic order independent of file iteration: shuffle the sentence pool once.
    rng.shuffle(&mut sentences);

    // Pre-compute syllable decompositions (skip sentences with missing chars).
    let mut decomp: Vec<(String, Vec<String>)> = Vec::new();
    for s in &sentences {
        if let Some(syls) = phrase_to_syllables(s, &table) {
            decomp.push((s.clone(), syls));
        }
    }

    let cases = synthesize_buckets(&decomp, &mut rng, per_bucket);
    write_jsonl(out, &cases)
}

/// Generate the FAIR gold set using longest-match word tokenization over the lexicon (correct
/// canonical readings, incl. polyphones), falling back to the per-char `hanzi_pinyin` table for
/// chars absent from the lexicon. Writes JSONL to `out`.
///
/// All buckets are derived from these correct readings, so e.g. the `abbr` bucket uses the
/// initials of the CORRECT syllables and the `typo`/`fuzzy` buckets perturb the CORRECT pinyin.
pub fn generate_correct(
    corpus: &Path,
    word_pinyin: &Path,
    hanzi_pinyin: &Path,
    out: &Path,
    seed: u64,
    per_bucket: usize,
) -> Result<()> {
    let table = load_hanzi_pinyin(hanzi_pinyin)?;
    let lex = Lexicon::load(word_pinyin)?;
    let mut sentences = load_sentences(corpus)?;

    let mut rng = Rng::new(seed);
    rng.shuffle(&mut sentences);

    let mut decomp: Vec<(String, Vec<String>)> = Vec::new();
    for s in &sentences {
        if let Some(syls) = phrase_to_syllables_correct(s, &lex, &table) {
            decomp.push((s.clone(), syls));
        }
    }

    let cases = synthesize_buckets(&decomp, &mut rng, per_bucket);
    write_jsonl(out, &cases)
}

/// Synthesize all scenario buckets from a list of `(phrase, syllables)` decompositions, using the
/// seeded `rng` for every stochastic choice. Shared by both the per-char and word-based paths so
/// the two gold sets are directly comparable (same buckets, caps, and RNG discipline).
fn synthesize_buckets(
    decomp: &[(String, Vec<String>)],
    rng: &mut Rng,
    per_bucket: usize,
) -> Vec<GoldCase> {
    let mut cases: Vec<GoldCase> = Vec::new();
    let push = |bucket: &str, input: String, expected: String, cases: &mut Vec<GoldCase>| {
        if input.is_empty() || expected.is_empty() {
            return;
        }
        cases.push(GoldCase { bucket: bucket.to_string(), input, expected });
    };

    // ---- full: full pinyin of the whole phrase ----
    for (phrase, syls) in decomp.iter().take(per_bucket) {
        push("full", syls.concat(), phrase.clone(), &mut cases);
    }

    // ---- long_sentence: full pinyin of >=6 char sentences ----
    {
        let mut n = 0;
        for (phrase, syls) in decomp.iter() {
            if syls.len() >= 6 {
                push("long_sentence", syls.concat(), phrase.clone(), &mut cases);
                n += 1;
                if n >= per_bucket {
                    break;
                }
            }
        }
    }

    // ---- short_word: 1-2 char words ----
    {
        let mut n = 0;
        for (phrase, syls) in decomp.iter() {
            if syls.len() >= 1 && syls.len() <= 2 {
                push("short_word", syls.concat(), phrase.clone(), &mut cases);
                n += 1;
                if n >= per_bucket {
                    break;
                }
            }
        }
    }

    // ---- abbr: initials only (first letter of each syllable). Keep short phrases so abbr
    // input is unambiguous-ish (long abbr inputs are nearly impossible to convert exactly). ----
    {
        let mut n = 0;
        for (phrase, syls) in decomp.iter() {
            if syls.len() >= 2 && syls.len() <= 4 {
                let abbr: String = syls.iter().filter_map(|s| s.chars().next()).collect();
                push("abbr", abbr, phrase.clone(), &mut cases);
                n += 1;
                if n >= per_bucket {
                    break;
                }
            }
        }
    }

    // ---- fuzzy: 1-2 fuzzy substitutions on the full pinyin ----
    {
        let mut n = 0;
        for (phrase, syls) in decomp.iter() {
            if let Some(fz) = apply_fuzzy(syls, rng) {
                if fz != syls.concat() {
                    push("fuzzy", fz, phrase.clone(), &mut cases);
                    n += 1;
                    if n >= per_bucket {
                        break;
                    }
                }
            }
        }
    }

    // ---- typo: inject one edit into the full pinyin ----
    {
        let mut n = 0;
        for (phrase, syls) in decomp.iter() {
            let full = syls.concat();
            if let Some(t) = inject_typo(&full, rng) {
                push("typo", t, phrase.clone(), &mut cases);
                n += 1;
                if n >= per_bucket {
                    break;
                }
            }
        }
    }

    // ---- english: passthrough, expected == input ----
    {
        let mut words: Vec<&str> = ENGLISH_WORDS.to_vec();
        rng.shuffle(&mut words);
        for w in words.iter().take(per_bucket) {
            push("english", w.to_string(), w.to_string(), &mut cases);
        }
        // If more requested than the built-in list, cycle with combinations.
        let mut i = 0;
        while cases.iter().filter(|c| c.bucket == "english").count() < per_bucket.min(ENGLISH_WORDS.len()) {
            let w = ENGLISH_WORDS[i % ENGLISH_WORDS.len()];
            push("english", w.to_string(), w.to_string(), &mut cases);
            i += 1;
            if i > ENGLISH_WORDS.len() {
                break;
            }
        }
    }

    // ---- mixed: Chinese phrase + an English token, e.g. "wo用github" → expected "我github" ----
    // We build input = <pinyin of short phrase> + <english>, expected = <hanzi> + <english>.
    {
        let mut n = 0;
        for (phrase, syls) in decomp.iter() {
            if syls.len() < 1 || syls.len() > 3 {
                continue;
            }
            let w = ENGLISH_WORDS[rng.below(ENGLISH_WORDS.len())];
            // pinyin then english token (engine should segment CN run then EN run)
            let input = format!("{}{}", syls.concat(), w);
            let expected = format!("{}{}", phrase, w);
            push("mixed", input, expected, &mut cases);
            n += 1;
            if n >= per_bucket {
                break;
            }
        }
    }

    cases
}

/// Write a list of gold cases as JSONL to `out` (creating parent dirs).
fn write_jsonl(out: &Path, cases: &[GoldCase]) -> Result<()> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let f = File::create(out).with_context(|| format!("create {}", out.display()))?;
    let mut w = BufWriter::new(f);
    for c in cases {
        serde_json::to_writer(&mut w, c)?;
        w.write_all(b"\n")?;
    }
    w.flush()?;
    Ok(())
}

/// Load a gold set from JSONL.
pub fn load(path: &Path) -> Result<Vec<GoldCase>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut out = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let c: GoldCase = serde_json::from_str(&line)?;
        out.push(c);
    }
    Ok(out)
}
