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
