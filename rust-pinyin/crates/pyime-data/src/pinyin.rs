//! Pinyin normalization: parse the hanzi/phrase tables, strip tone marks, and
//! convert words to their canonical reading key (syllables joined by `'`).

use anyhow::{Context, Result};
use rustc_hash::FxHashMap;

/// char -> canonical (toneless, lowercase, ü→v) FIRST reading.
pub type HanziTable = FxHashMap<char, String>;

/// True if `c` is a CJK Unified Ideograph (incl. common extension A).
#[inline]
pub fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3400..=0x4DBF      // Ext A
        | 0x4E00..=0x9FFF    // BMP
        | 0xF900..=0xFAFF    // Compatibility Ideographs
        | 0x20000..=0x2A6DF  // Ext B
    )
}

/// True if every char in `s` is CJK (and `s` non-empty).
pub fn is_all_cjk(s: &str) -> bool {
    let mut any = false;
    for c in s.chars() {
        if !is_cjk(c) {
            return false;
        }
        any = true;
    }
    any
}

/// Strip a toned pinyin syllable to plain canonical letters: remove tone marks,
/// map ü/v → `v`, lowercase, drop trailing tone digits.
pub fn normalize_syllable(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        let lc = ch.to_ascii_lowercase();
        if lc.is_ascii_digit() {
            continue; // tone number
        }
        let mapped = match ch {
            'ā' | 'á' | 'ǎ' | 'à' | 'a' => 'a',
            'ē' | 'é' | 'ě' | 'è' | 'ê' | 'e' => 'e',
            'ī' | 'í' | 'ǐ' | 'ì' | 'i' => 'i',
            'ō' | 'ó' | 'ǒ' | 'ò' | 'o' => 'o',
            'ū' | 'ú' | 'ǔ' | 'ù' | 'u' => 'u',
            'ǖ' | 'ǘ' | 'ǚ' | 'ǜ' | 'ü' | 'v' => 'v',
            // 'ń','ň','ǹ','ḿ' (interjections) keep base letter
            'ń' | 'ň' | 'ǹ' => 'n',
            'ḿ' => 'm',
            other => other.to_ascii_lowercase(),
        };
        if mapped.is_ascii_alphabetic() {
            out.push(mapped);
        }
    }
    out
}

/// Parse mozillazg/pinyin-data `pinyin.txt`:
///   `U+3007: líng,yuán,xīng  # 〇`
/// Take the FIRST reading per char as canonical.
pub fn parse_hanzi_table(raw: &[u8]) -> Result<HanziTable> {
    let text = std::str::from_utf8(raw).context("pinyin-data utf8")?;
    let mut map: HanziTable = FxHashMap::default();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // U+XXXX: r1,r2  # 汉
        let (code, rest) = match line.split_once(':') {
            Some(x) => x,
            None => continue,
        };
        let code = code.trim();
        if !code.starts_with("U+") {
            continue;
        }
        let cp = match u32::from_str_radix(&code[2..], 16) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let ch = match char::from_u32(cp) {
            Some(c) => c,
            None => continue,
        };
        // readings are before the '#'
        let readings = rest.split('#').next().unwrap_or("").trim();
        let first = match readings.split(',').next() {
            Some(r) => r.trim(),
            None => continue,
        };
        let syl = normalize_syllable(first);
        if !syl.is_empty() {
            map.insert(ch, syl);
        }
    }
    Ok(map)
}

/// Parse mozillazg/phrase-pinyin-data `pinyin.txt`:
///   `你好: nǐ hǎo`  →  ("你好", ["ni","hao"])
/// Only keep phrases whose syllable count matches char count (sane alignment).
pub fn parse_phrase_table(raw: &[u8]) -> Result<FxHashMap<String, Vec<String>>> {
    let text = std::str::from_utf8(raw).context("phrase-pinyin-data utf8")?;
    let mut map: FxHashMap<String, Vec<String>> = FxHashMap::default();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (word, py) = match line.split_once(':') {
            Some(x) => x,
            None => continue,
        };
        let word = word.trim();
        if !is_all_cjk(word) {
            continue;
        }
        let sylls: Vec<String> = py
            .split_whitespace()
            .map(normalize_syllable)
            .filter(|s| !s.is_empty())
            .collect();
        if sylls.len() != word.chars().count() {
            continue; // misaligned; fall back to per-char readings
        }
        map.insert(word.to_string(), sylls);
    }
    Ok(map)
}

/// Convert a CJK word to its canonical reading key (syllables joined by `'`).
/// Prefers a phrase-pinyin-data override; otherwise composes per-char readings.
/// Returns None if EVERY char lacks a reading (word is unusable).
pub fn word_to_key(
    word: &str,
    hanzi: &HanziTable,
    phrases: &FxHashMap<String, Vec<String>>,
) -> Option<String> {
    if let Some(sylls) = phrases.get(word) {
        return Some(sylls.join("'"));
    }
    let mut sylls: Vec<&str> = Vec::new();
    let mut any = false;
    for c in word.chars() {
        match hanzi.get(&c) {
            Some(s) => {
                sylls.push(s.as_str());
                any = true;
            }
            None => return None, // a char with no reading → can't form a clean key
        }
    }
    if !any {
        return None;
    }
    Some(sylls.join("'"))
}

/// Normalize a space-separated rime-ice pinyin reading (e.g. `zhong guo`,
/// `ni hao`) into the canonical key with syllables joined by `'` (e.g.
/// `zhong'guo`). Returns None if no usable syllable survives. Each syllable is
/// passed through [`normalize_syllable`] (tone-less, ü/v→`v`, lowercase).
pub fn normalize_reading(py: &str) -> Option<String> {
    let sylls: Vec<String> = py
        .split_whitespace()
        .map(normalize_syllable)
        .filter(|s| !s.is_empty())
        .collect();
    if sylls.is_empty() {
        None
    } else {
        Some(sylls.join("'"))
    }
}

/// Parse a rime-ice single-character dict (e.g. `8105.dict.yaml`,
/// `41448.dict.yaml`) into a char→canonical-reading table. Lines are
/// `char<TAB>pinyin[<TAB>weight]`; only single-CJK-char entries are kept, and
/// the FIRST reading seen for a char wins (files are weight-sorted, so this is
/// the most common reading). Comment/`---`/`...`/header lines are skipped.
pub fn parse_rime_char_table(raw: &[u8]) -> Result<HanziTable> {
    let text = std::str::from_utf8(raw).context("rime char table utf8")?;
    let mut map: HanziTable = FxHashMap::default();
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
        // single CJK char only
        let mut chs = word.chars();
        let ch = match (chs.next(), chs.next()) {
            (Some(c), None) if is_cjk(c) => c,
            _ => continue,
        };
        let py = match it.next() {
            Some(p) => p.trim(),
            None => continue,
        };
        // a single-char reading should be one syllable
        let syl = normalize_syllable(py.split_whitespace().next().unwrap_or(""));
        if !syl.is_empty() {
            map.entry(ch).or_insert(syl);
        }
    }
    Ok(map)
}

// ===========================================================================
// OpenCC Traditional → Simplified normalization
// ===========================================================================

/// Traditional → Simplified converter built from the OpenCC `TSPhrases.txt`
/// (phrase-level) and `TSCharacters.txt` (char-level) dictionaries.
///
/// Conversion applies the phrase map first (longest match at each position),
/// then per-character mapping for any remaining (un-phrase-matched) characters.
/// This mirrors how real simplified IMEs normalize their lexicon: 繁体 surfaces
/// collapse onto their 简体 canonical form while the READING is left untouched.
///
/// For each OpenCC entry the FIRST value (space-separated) is taken as the
/// canonical simplified form. Self-mapping entries (trad == simp) are dropped so
/// `is_noop()`-style fast paths stay cheap and a word that is already simplified
/// is returned unchanged.
pub struct OpenCc {
    /// trad char -> canonical simp char (only entries that actually change).
    chars: FxHashMap<char, char>,
    /// trad phrase -> canonical simp phrase (only entries that actually change).
    phrases: FxHashMap<String, String>,
    /// longest phrase key length in chars (0 if no phrases).
    max_phrase_chars: usize,
}

impl OpenCc {
    /// Build from raw OpenCC `TSCharacters.txt` and `TSPhrases.txt` bytes. Either
    /// may be empty (e.g. download failed); conversion then degrades gracefully
    /// (an empty converter is an identity map).
    pub fn from_raw(ts_chars_raw: &[u8], ts_phrases_raw: &[u8]) -> Self {
        let mut chars: FxHashMap<char, char> = FxHashMap::default();
        let mut phrases: FxHashMap<String, String> = FxHashMap::default();
        let mut max_phrase_chars = 0usize;

        // Pass 0: collect the set of chars that appear as a canonical simplified
        // VALUE (RHS first token of TSCharacters). OpenCC's TSCharacters is a
        // *variant-folding* table, so some KEYS are themselves perfectly valid
        // simplified characters (e.g. `坏→坯`); a real simplified IME must NOT fold
        // those away. We treat any char that is a canonical simplified value as
        // "simplified-valid" and refuse to convert it (or any phrase containing
        // only such chars). This restricts conversion to genuinely traditional-only
        // characters.
        let mut simp_values: rustc_hash::FxHashSet<char> = rustc_hash::FxHashSet::default();
        if let Ok(text) = std::str::from_utf8(ts_chars_raw) {
            for line in text.lines() {
                let line = line.trim_end_matches(['\r', '\n']);
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let vals = match line.split_once('\t') {
                    Some((_, v)) => v,
                    None => continue,
                };
                if let Some(c) = vals.split_whitespace().next().and_then(|v| v.chars().next()) {
                    simp_values.insert(c);
                }
            }
        }

        // char-level: `trad<TAB>simp [alt...]`, single CJK char key.
        if let Ok(text) = std::str::from_utf8(ts_chars_raw) {
            for line in text.lines() {
                let line = line.trim_end_matches(['\r', '\n']);
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let (key, vals) = match line.split_once('\t') {
                    Some(x) => x,
                    None => continue,
                };
                let mut kc = key.chars();
                let (trad, second) = (kc.next(), kc.next());
                let trad = match (trad, second) {
                    (Some(c), None) => c,
                    _ => continue, // not a single-char key
                };
                let simp = match vals.split_whitespace().next().and_then(|v| v.chars().next()) {
                    Some(c) => c,
                    None => continue,
                };
                // Skip self-maps and keys that are themselves a canonical simplified
                // character (those are NOT traditional-only; converting them mangles
                // legitimate simplified text).
                if trad != simp && !simp_values.contains(&trad) {
                    chars.insert(trad, simp);
                }
            }
        }

        // phrase-level: `trad<TAB>simp [alt...]`, multi-char key.
        if let Ok(text) = std::str::from_utf8(ts_phrases_raw) {
            for line in text.lines() {
                let line = line.trim_end_matches(['\r', '\n']);
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let (key, vals) = match line.split_once('\t') {
                    Some(x) => x,
                    None => continue,
                };
                let key = key.trim();
                let simp = match vals.split_whitespace().next() {
                    Some(v) => v.trim(),
                    None => continue,
                };
                if key.is_empty() || simp.is_empty() || key == simp {
                    continue;
                }
                // Only keep a phrase mapping if the KEY contains at least one
                // genuinely-traditional char (one that is not itself a canonical
                // simplified value). This skips variant-fold phrases whose key is
                // already all-simplified (e.g. `坏子→坯子`), preventing mangling of
                // legitimate simplified compounds.
                let has_trad = key.chars().any(|c| !simp_values.contains(&c));
                if !has_trad {
                    continue;
                }
                let klen = key.chars().count();
                if klen >= 1 {
                    max_phrase_chars = max_phrase_chars.max(klen);
                    phrases.insert(key.to_string(), simp.to_string());
                }
            }
        }

        OpenCc {
            chars,
            phrases,
            max_phrase_chars,
        }
    }

    /// Number of (changing) char + phrase mappings loaded.
    pub fn len(&self) -> usize {
        self.chars.len() + self.phrases.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chars.is_empty() && self.phrases.is_empty()
    }

    /// Convert a Traditional surface into Simplified: longest-match phrase
    /// substitution first, then per-character mapping for the rest. Non-mapped
    /// characters (including non-CJK) pass through unchanged. A fully-simplified
    /// input is returned identical (modulo allocation).
    pub fn convert(&self, s: &str) -> String {
        let chars: Vec<char> = s.chars().collect();
        let mut out = String::with_capacity(s.len());
        let mut i = 0;
        while i < chars.len() {
            // Try the longest phrase match starting at i.
            let mut matched = false;
            if self.max_phrase_chars >= 2 {
                let upper = (i + self.max_phrase_chars).min(chars.len());
                let mut j = upper;
                while j > i + 1 {
                    let cand: String = chars[i..j].iter().collect();
                    if let Some(simp) = self.phrases.get(&cand) {
                        out.push_str(simp);
                        i = j;
                        matched = true;
                        break;
                    }
                    j -= 1;
                }
            }
            if matched {
                continue;
            }
            // Fall back to per-character mapping.
            let c = chars[i];
            match self.chars.get(&c) {
                Some(&simp) => out.push(simp),
                None => out.push(c),
            }
            i += 1;
        }
        out
    }
}

/// Coarse part-of-speech tag → small u8 code (0 = generic). Just a few buckets.
pub fn pos_tag(pos: &str) -> u8 {
    match pos.chars().next() {
        Some('n') => 1, // noun-ish
        Some('v') => 2, // verb
        Some('a') => 3, // adjective
        Some('r') => 4, // pronoun
        Some('m') => 5, // numeral
        Some('t') => 6, // time
        Some('s') => 7, // place
        _ => 0,
    }
}
