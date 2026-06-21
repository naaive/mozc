//! Gold-set generation from a held-out Chinese sentence corpus.
//!
//! Each held-out sentence is converted to canonical pinyin via the `hanzi_pinyin.tsv`
//! first-reading table (one syllable per Han char). From that we synthesize `(input, expected)`
//! pairs bucketed by scenario (see DESIGN.md "Evaluation system").
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

/// Convert a Chinese phrase to (per-syllable) pinyin. Returns None if any Han char is missing
/// from the table (so we skip such sentences as required).
fn phrase_to_syllables(phrase: &str, table: &HashMap<char, String>) -> Option<Vec<String>> {
    let mut syls = Vec::new();
    for c in phrase.chars() {
        // Only handle CJK-ish chars; reject if not found.
        let py = table.get(&c)?;
        if py.is_empty() || !py.bytes().all(|b| b.is_ascii_lowercase()) {
            return None;
        }
        syls.push(py.clone());
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

/// Generate the gold set and write it as JSONL to `out`.
pub fn generate(
    corpus: &Path,
    hanzi_pinyin: &Path,
    out: &Path,
    seed: u64,
    per_bucket: usize,
) -> Result<()> {
    let table = load_hanzi_pinyin(hanzi_pinyin)?;
    let f = File::open(corpus).with_context(|| format!("open {}", corpus.display()))?;
    let mut sentences: Vec<String> = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        let t = line.trim();
        if !t.is_empty() {
            sentences.push(t.to_string());
        }
    }

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
            if let Some(fz) = apply_fuzzy(syls, &mut rng) {
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
            if let Some(t) = inject_typo(&full, &mut rng) {
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

    // Write JSONL.
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let f = File::create(out).with_context(|| format!("create {}", out.display()))?;
    let mut w = BufWriter::new(f);
    for c in &cases {
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
