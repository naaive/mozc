//! Syllable lattice construction (full / abbrev / fuzzy / correction expansion).
//!
//! The input is first *normalized* into a sequence of lowercase ASCII letters, with separator
//! positions recorded so syllable edges never cross an explicit boundary (`'`, space, digit).
//! Over the normalized letters we build a lattice: for every start index we enumerate candidate
//! edges, each carrying a canonical syllable (or an abbreviation initial token) and an edit
//! cost. The decoder later walks the lexicon FST along lattice paths.

use crate::consts::{ABBR_PEN, FUZZY_PEN, TYPO_PEN};
use crate::fuzzy::{self, FuzzySet};
use crate::syllable;
use crate::EngineConfig;
use rustc_hash::FxHashSet;

/// A single lattice edge over the normalized letter string `[start, end)`.
#[derive(Debug, Clone)]
pub struct Edge {
    pub start: usize,
    pub end: usize,
    /// Canonical syllable text. For abbreviation edges this is just the initial (e.g. "b", "zh"),
    /// and `abbrev` is set.
    pub syllable: String,
    /// Integer edit cost contributed by this edge (fuzzy/abbr/typo penalties summed).
    pub cost: i32,
    /// True if this is an abbreviation (initial-only) token; lexicon match treats it as
    /// "any syllable starting with this initial".
    pub abbrev: bool,
    /// True if this edge crosses an explicit syllable boundary at `end` (separator follows).
    pub boundary_after: bool,
}

/// Normalized input: lowercase letters plus a map from letter index → original byte offset,
/// and a set of letter indices after which an explicit separator occurred.
pub struct Normalized {
    /// Lowercased ASCII letters only.
    pub letters: String,
    /// `orig_byte[i]` = byte offset in the original input where letter `i` started.
    pub orig_byte: Vec<usize>,
    /// `orig_end[i]` = byte offset in original input where letter i ended (exclusive).
    pub orig_end: Vec<usize>,
    /// letter indices (0-based, value = letter index) that are forced syllable starts because a
    /// separator preceded them. Always includes 0.
    pub hard_starts: FxHashSet<usize>,
    /// letter indices `i` such that a separator occurs immediately after letter `i-1` /
    /// before letter `i`; equivalently boundaries at letter position. We store boundary
    /// positions as "letter index at which a hard boundary exists to its left".
    pub boundary_at: FxHashSet<usize>,
}

/// Split the raw input into normalized letter runs. Separators are `'`, ASCII whitespace, and
/// ASCII digits. Non-ASCII (e.g. CJK) characters also act as separators here — the decoder
/// handles CJK / mixed input at a higher level, but for the latin pinyin lattice they bound runs.
pub fn normalize(input: &str) -> Normalized {
    let mut letters = String::new();
    let mut orig_byte = Vec::new();
    let mut orig_end = Vec::new();
    let mut boundary_at = FxHashSet::default();
    let mut hard_starts = FxHashSet::default();
    let mut pending_boundary = true; // start of string is a boundary

    for (b, ch) in input.char_indices() {
        if ch.is_ascii_alphabetic() {
            let idx = letters.len();
            if pending_boundary {
                hard_starts.insert(idx);
                boundary_at.insert(idx);
                pending_boundary = false;
            }
            letters.push(ch.to_ascii_lowercase());
            orig_byte.push(b);
            orig_end.push(b + ch.len_utf8());
        } else {
            // separator (apostrophe, space, digit, punctuation, CJK, ...)
            pending_boundary = true;
        }
    }
    // boundary at end
    boundary_at.insert(letters.len());

    Normalized {
        letters,
        orig_byte,
        orig_end,
        hard_starts,
        boundary_at,
    }
}

/// Adjacent-key (fat-finger) map for QWERTY: each key → neighbors used for substitution typos.
fn adjacent_keys(c: u8) -> &'static [u8] {
    match c {
        b'q' => b"wa",
        b'w' => b"qes",
        b'e' => b"wrd",
        b'r' => b"etf",
        b't' => b"ryg",
        b'y' => b"tuh",
        b'u' => b"yij",
        b'i' => b"uok",
        b'o' => b"ipl",
        b'p' => b"ol",
        b'a' => b"qsz",
        b's' => b"awdz",
        b'd' => b"serf",
        b'f' => b"drtg",
        b'g' => b"fyth",
        b'h' => b"gjy",
        b'j' => b"hkun",
        b'k' => b"jlim",
        b'l' => b"kop",
        b'z' => b"asx",
        b'x' => b"zsdc",
        b'c' => b"xdfv",
        b'v' => b"cfgb",
        b'b' => b"vghn",
        b'n' => b"bhjm",
        b'm' => b"njk",
        _ => b"",
    }
}

/// Generate edit-distance-1 (and small-budget) typo correction syllables for the slice
/// `letters[start..]`. Returns `(canonical_syllable, consumed_len, n_edits)`.
///
/// We attempt: single substitution (fat-finger neighbor), single deletion (user typed an extra
/// letter), single insertion (user missed a letter), and adjacent transposition. We test each
/// modified prefix against the canonical syllable table. To stay bounded we only correct over a
/// window of length ≤ MAX_SYL_LEN+1 from `start`.
fn correction_syllables(letters: &str, start: usize, max_edits: u8) -> Vec<(String, usize, u8)> {
    if max_edits == 0 {
        return Vec::new();
    }
    let bytes = letters.as_bytes();
    let n = bytes.len();
    let window_end = (start + syllable::MAX_SYL_LEN + 1).min(n);
    let window = &letters[start..window_end];
    let wb = window.as_bytes();
    let wlen = wb.len();
    let mut out: Vec<(String, usize, u8)> = Vec::new();
    let mut seen: FxHashSet<(String, usize)> = FxHashSet::default();

    let try_push =
        |s: &str, consumed: usize, edits: u8, out: &mut Vec<(String, usize, u8)>, seen: &mut FxHashSet<(String, usize)>| {
            if consumed == 0 {
                return;
            }
            if syllable::is_syllable(s) && seen.insert((s.to_string(), consumed)) {
                out.push((s.to_string(), consumed, edits));
            }
        };

    // 1) Substitution: replace one window char with a neighbor; the corrected string is the
    //    same length, so consumed = candidate length.
    for i in 0..wlen.min(syllable::MAX_SYL_LEN) {
        for &nb in adjacent_keys(wb[i]) {
            let mut buf: Vec<u8> = wb[..(i + 1).max(0)].to_vec();
            // build candidates for each possible consumed length L (i < L <= wlen)
            for l in (i + 1)..=wlen.min(syllable::MAX_SYL_LEN) {
                buf.clear();
                buf.extend_from_slice(&wb[..l]);
                buf[i] = nb;
                if let Ok(s) = std::str::from_utf8(&buf) {
                    let s = s.to_string();
                    try_push(&s, l, 1, &mut out, &mut seen);
                }
            }
        }
    }

    // 2) Deletion: user typed an extra letter — drop window char i, the canonical syllable is
    //    shorter but we consumed l input letters.
    for i in 0..wlen.min(syllable::MAX_SYL_LEN + 1) {
        for l in (i + 1)..=wlen.min(syllable::MAX_SYL_LEN + 1) {
            let mut buf: Vec<u8> = Vec::with_capacity(l - 1);
            buf.extend_from_slice(&wb[..i]);
            buf.extend_from_slice(&wb[i + 1..l]);
            if let Ok(s) = std::str::from_utf8(&buf) {
                let s = s.to_string();
                try_push(&s, l, 1, &mut out, &mut seen);
            }
        }
    }

    // 3) Insertion: user missed a letter — insert one letter; canonical longer than consumed.
    for i in 0..=wlen.min(syllable::MAX_SYL_LEN) {
        for &ins in b"abcdefghijklmnopqrstuvwxyz" {
            for l in i..=wlen.min(syllable::MAX_SYL_LEN) {
                let mut buf: Vec<u8> = Vec::with_capacity(l + 1);
                buf.extend_from_slice(&wb[..i]);
                buf.push(ins);
                buf.extend_from_slice(&wb[i..l]);
                if let Ok(s) = std::str::from_utf8(&buf) {
                    if s.len() <= syllable::MAX_SYL_LEN {
                        let s = s.to_string();
                        try_push(&s, l, 1, &mut out, &mut seen);
                    }
                }
            }
        }
    }

    // 4) Adjacent transposition.
    for i in 0..wlen.saturating_sub(1).min(syllable::MAX_SYL_LEN) {
        for l in (i + 2)..=wlen.min(syllable::MAX_SYL_LEN) {
            let mut buf: Vec<u8> = wb[..l].to_vec();
            buf.swap(i, i + 1);
            if let Ok(s) = std::str::from_utf8(&buf) {
                let s = s.to_string();
                try_push(&s, l, 1, &mut out, &mut seen);
            }
        }
    }

    let _ = (n, max_edits);
    out
}

/// Cap on edges produced per start position (keeps expansion bounded).
const MAX_EDGES_PER_POS: usize = 24;

/// Build the syllable lattice over the normalized letters. Returns edges grouped by start
/// position: `edges_from[i]` lists all edges leaving letter index `i`.
pub fn build_lattice(norm: &Normalized, cfg: &EngineConfig) -> Vec<Vec<Edge>> {
    let letters = &norm.letters;
    let n = letters.len();
    let mut edges_from: Vec<Vec<Edge>> = vec![Vec::new(); n + 1];
    let fuzzy_set: FuzzySet = cfg.fuzzy;

    for start in 0..n {
        let rest = &letters[start..];
        let mut seen: FxHashSet<(usize, String, bool)> = FxHashSet::default();
        let mut local: Vec<Edge> = Vec::new();

        let add = |end: usize, syl: String, cost: i32, abbrev: bool, local: &mut Vec<Edge>, seen: &mut FxHashSet<(usize, String, bool)>| {
            let key = (end, syl.clone(), abbrev);
            if seen.insert(key) {
                local.push(Edge {
                    start,
                    end,
                    syllable: syl,
                    cost,
                    abbrev,
                    boundary_after: norm.boundary_at.contains(&end),
                });
            }
        };

        // (a) exact max-munch syllables
        for (syl, len) in syllable::prefix_syllables(rest) {
            add(start + len, syl.to_string(), 0, false, &mut local, &mut seen);
        }

        // (b) fuzzy variants of each exact-length prefix (only when fuzzy enabled)
        if !fuzzy_set.is_empty() {
            // For each prefix length that forms a syllable OR could form one under fuzzy, try.
            let max = syllable::MAX_SYL_LEN.min(rest.len());
            for len in 1..=max {
                let cand = &rest[..len];
                for (variant, was_fuzzy) in fuzzy::fuzzy_variants(cand, fuzzy_set) {
                    if was_fuzzy {
                        add(start + len, variant, FUZZY_PEN, false, &mut local, &mut seen);
                    }
                }
            }
        }

        // (c) abbreviation: a single leading initial (1 or 2 chars) as an initial-match token
        for (init, len) in syllable::prefix_initials(rest) {
            add(start + len, init.to_string(), ABBR_PEN, true, &mut local, &mut seen);
        }

        // (d) correction: edit-distance typo fixes
        if cfg.enable_correction {
            for (syl, consumed, edits) in
                correction_syllables(letters, start, cfg.correction_max_edits)
            {
                let cost = TYPO_PEN * edits as i32;
                add(start + consumed, syl, cost, false, &mut local, &mut seen);
            }
        }

        // bound expansion: keep cheapest edges, preferring longer spans on ties
        if local.len() > MAX_EDGES_PER_POS {
            local.sort_by(|a, b| {
                a.cost
                    .cmp(&b.cost)
                    .then_with(|| (b.end - b.start).cmp(&(a.end - a.start)))
            });
            local.truncate(MAX_EDGES_PER_POS);
        }

        edges_from[start] = local;
    }

    edges_from
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EngineConfig;

    #[test]
    fn normalize_separators() {
        let n = normalize("ni'hao");
        assert_eq!(n.letters, "nihao");
        assert!(n.boundary_at.contains(&2)); // boundary before 'h'
    }

    #[test]
    fn lattice_full_pinyin() {
        let cfg = EngineConfig::default();
        let n = normalize("nihao");
        let lat = build_lattice(&n, &cfg);
        // edge ni from 0..2
        assert!(lat[0].iter().any(|e| e.syllable == "ni" && e.end == 2 && !e.abbrev));
        // edge hao from 2..5
        assert!(lat[2].iter().any(|e| e.syllable == "hao" && e.end == 5));
    }

    #[test]
    fn lattice_abbrev() {
        let cfg = EngineConfig::default();
        let n = normalize("bj");
        let lat = build_lattice(&n, &cfg);
        assert!(lat[0].iter().any(|e| e.syllable == "b" && e.abbrev));
        assert!(lat[1].iter().any(|e| e.syllable == "j" && e.abbrev));
    }

    #[test]
    fn lattice_fuzzy() {
        let cfg = EngineConfig::default();
        let n = normalize("zongguo");
        let lat = build_lattice(&n, &cfg);
        // zong should produce zhong as a fuzzy variant
        assert!(lat[0]
            .iter()
            .any(|e| e.syllable == "zhong" && e.cost >= FUZZY_PEN));
    }
}
