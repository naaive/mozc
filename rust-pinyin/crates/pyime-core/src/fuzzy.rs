//! Fuzzy-pinyin rule set: bidirectional initial/final variant rules implemented as bit flags.
//!
//! Given an input syllable we yield canonical syllable variants reachable under the enabled
//! rules. Each variant is paired with a flag indicating whether a fuzzy substitution was
//! actually applied (so the caller can charge `FUZZY_PEN`). Only variants that are themselves
//! valid canonical syllables are returned.

use crate::syllable::is_syllable;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FuzzySet {
    pub bits: u32,
}

// Initial fuzzy classes.
pub const F_ZH_Z: u32 = 1 << 0; // zh <-> z
pub const F_CH_C: u32 = 1 << 1; // ch <-> c
pub const F_SH_S: u32 = 1 << 2; // sh <-> s
pub const F_N_L: u32 = 1 << 3; // n <-> l
pub const F_F_H: u32 = 1 << 4; // f <-> h
pub const F_R_L: u32 = 1 << 5; // r <-> l
pub const F_L_R: u32 = 1 << 6; // l <-> r (kept distinct flag; symmetric application)
// Final fuzzy classes.
pub const F_AN_ANG: u32 = 1 << 7; // an <-> ang
pub const F_EN_ENG: u32 = 1 << 8; // en <-> eng
pub const F_IN_ING: u32 = 1 << 9; // in <-> ing
pub const F_IAN_IANG: u32 = 1 << 10; // ian <-> iang
pub const F_UAN_UANG: u32 = 1 << 11; // uan <-> uang

const ALL_BITS: u32 = F_ZH_Z
    | F_CH_C
    | F_SH_S
    | F_N_L
    | F_F_H
    | F_R_L
    | F_L_R
    | F_AN_ANG
    | F_EN_ENG
    | F_IN_ING
    | F_IAN_IANG
    | F_UAN_UANG;

impl FuzzySet {
    pub fn none() -> Self {
        FuzzySet { bits: 0 }
    }
    pub fn all() -> Self {
        FuzzySet { bits: ALL_BITS }
    }
    #[inline]
    pub fn has(self, flag: u32) -> bool {
        self.bits & flag != 0
    }
    pub fn with(mut self, flag: u32) -> Self {
        self.bits |= flag;
        self
    }
    pub fn is_empty(self) -> bool {
        self.bits & ALL_BITS == 0
    }
}

impl Default for FuzzySet {
    fn default() -> Self {
        FuzzySet::all()
    }
}

/// An initial-substitution rule: if a syllable starts with `a`, it may also start with `b`
/// (and vice versa), gated by `flag`.
struct InitialRule {
    a: &'static str,
    b: &'static str,
    flag: u32,
}

const INITIAL_RULES: &[InitialRule] = &[
    InitialRule { a: "zh", b: "z", flag: F_ZH_Z },
    InitialRule { a: "ch", b: "c", flag: F_CH_C },
    InitialRule { a: "sh", b: "s", flag: F_SH_S },
    InitialRule { a: "n", b: "l", flag: F_N_L },
    InitialRule { a: "f", b: "h", flag: F_F_H },
    InitialRule { a: "r", b: "l", flag: F_R_L },
    InitialRule { a: "l", b: "r", flag: F_L_R },
];

/// A final-substitution rule: if a syllable ends with `a`, it may also end with `b`.
struct FinalRule {
    a: &'static str,
    b: &'static str,
    flag: u32,
}

const FINAL_RULES: &[FinalRule] = &[
    // longer finals first so we don't mis-trigger on a shorter suffix
    FinalRule { a: "iang", b: "ian", flag: F_IAN_IANG },
    FinalRule { a: "uang", b: "uan", flag: F_UAN_UANG },
    FinalRule { a: "ang", b: "an", flag: F_AN_ANG },
    FinalRule { a: "eng", b: "en", flag: F_EN_ENG },
    FinalRule { a: "ing", b: "in", flag: F_IN_ING },
];

fn swap_initial<'a>(syl: &str, rule: &InitialRule) -> Option<String> {
    if let Some(rest) = syl.strip_prefix(rule.a) {
        // Avoid empty-rest replacing a standalone initial token here (handled elsewhere).
        return Some(format!("{}{}", rule.b, rest));
    }
    if let Some(rest) = syl.strip_prefix(rule.b) {
        return Some(format!("{}{}", rule.a, rest));
    }
    let _ = syl;
    None
}

fn swap_final(syl: &str, rule: &FinalRule) -> Option<String> {
    if let Some(head) = syl.strip_suffix(rule.a) {
        if !head.is_empty() {
            return Some(format!("{}{}", head, rule.b));
        }
    }
    if let Some(head) = syl.strip_suffix(rule.b) {
        if !head.is_empty() {
            return Some(format!("{}{}", head, rule.a));
        }
    }
    None
}

/// Yield canonical syllable variants of `syl` reachable under the enabled fuzzy rules.
/// Each result is `(variant, fuzzy_used)`. The exact input (if itself canonical) is always
/// included with `fuzzy_used = false`. Applies at most one initial and one final substitution
/// (combinations included) to keep the fan-out bounded.
pub fn fuzzy_variants(syl: &str, set: FuzzySet) -> Vec<(String, bool)> {
    let mut out: Vec<(String, bool)> = Vec::new();
    let mut seen: rustc_hash::FxHashSet<String> = rustc_hash::FxHashSet::default();

    let push = |s: String, fuzzy: bool, out: &mut Vec<(String, bool)>, seen: &mut rustc_hash::FxHashSet<String>| {
        if is_syllable(&s) && seen.insert(s.clone()) {
            out.push((s, fuzzy));
        }
    };

    // base
    push(syl.to_string(), false, &mut out, &mut seen);

    if set.is_empty() {
        return out;
    }

    // Collect a small working set of "initial-substituted" forms (each tagged whether fuzzy).
    let mut bases: Vec<(String, bool)> = vec![(syl.to_string(), false)];
    for rule in INITIAL_RULES {
        if !set.has(rule.flag) {
            continue;
        }
        if let Some(v) = swap_initial(syl, rule) {
            bases.push((v, true));
        }
    }

    // For each base form, apply final rules (and include the base itself).
    for (base, init_fuzzy) in bases {
        push(base.clone(), init_fuzzy, &mut out, &mut seen);
        for rule in FINAL_RULES {
            if !set.has(rule.flag) {
                continue;
            }
            if let Some(v) = swap_final(&base, rule) {
                push(v, true, &mut out, &mut seen);
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zh_z_bidirectional() {
        let v = fuzzy_variants("zhong", FuzzySet::all());
        assert!(v.iter().any(|(s, _)| s == "zhong"));
        assert!(v.iter().any(|(s, f)| s == "zong" && *f));
        let v2 = fuzzy_variants("zong", FuzzySet::all());
        assert!(v2.iter().any(|(s, f)| s == "zhong" && *f));
    }

    #[test]
    fn final_an_ang() {
        let v = fuzzy_variants("fan", FuzzySet::all());
        assert!(v.iter().any(|(s, f)| s == "fang" && *f));
    }

    #[test]
    fn none_set_only_identity() {
        let v = fuzzy_variants("zhong", FuzzySet::none());
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].0, "zhong");
    }
}
