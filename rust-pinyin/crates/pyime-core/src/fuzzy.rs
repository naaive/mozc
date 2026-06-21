//! Fuzzy-pinyin rule set. CORE agent: implement the full bidirectional rule table
//! (zh/z, ch/c, sh/s, n/l, f/h, r/l, an/ang, en/eng, in/ing, ian/iang, uan/uang, ...).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FuzzySet {
    pub bits: u32,
}

impl FuzzySet {
    pub fn none() -> Self {
        FuzzySet { bits: 0 }
    }
    pub fn all() -> Self {
        FuzzySet { bits: u32::MAX }
    }
}

impl Default for FuzzySet {
    fn default() -> Self {
        FuzzySet::all()
    }
}
