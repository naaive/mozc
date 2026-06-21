//! Word unigram + bigram language model (bigram.fst, backoff to unigram).
//!
//! `bigram.fst` is an `fst::Map` keyed by `bigram_key(prev_id, id)` (8 bytes BE), value =
//! bigram cost. Missing pairs back off to a fixed penalty so the decoder still has a
//! transition cost.

use crate::format::bigram_key;
use fst::raw::Fst;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// Backoff penalty (in LOG_BASE cost units) added when a bigram is absent. Roughly a moderately
/// unlikely transition; the unigram cost of the target word still dominates ranking.
pub const BIGRAM_BACKOFF: u32 = 2400;

pub struct LanguageModel {
    _mmap: Mmap,
    fst: Fst<&'static [u8]>,
}

impl LanguageModel {
    pub fn load(data_dir: &Path) -> anyhow::Result<LanguageModel> {
        let path = data_dir.join("bigram.fst");
        let f = File::open(&path)
            .map_err(|e| anyhow::anyhow!("open {}: {e}", path.display()))?;
        // SAFETY: read-only mmap held for the lifetime of this struct.
        let mmap = unsafe { Mmap::map(&f) }
            .map_err(|e| anyhow::anyhow!("mmap {}: {e}", path.display()))?;
        let slice: &'static [u8] =
            unsafe { std::mem::transmute::<&[u8], &'static [u8]>(&mmap[..]) };
        let fst = Fst::new(slice).map_err(|e| anyhow::anyhow!("invalid bigram.fst: {e}"))?;
        Ok(LanguageModel { _mmap: mmap, fst })
    }

    /// Bigram cost for `(prev_id, id)`, or `None` if absent (caller applies backoff).
    pub fn bigram_cost(&self, prev_id: u32, id: u32) -> Option<u32> {
        let key = bigram_key(prev_id, id);
        self.fst.get(key).map(|o| o.value() as u32)
    }

    /// Transition cost from `prev` to `id`: bigram if present, else backoff penalty.
    pub fn transition_cost(&self, prev_id: Option<u32>, id: u32) -> u32 {
        match prev_id {
            Some(p) => self.bigram_cost(p, id).unwrap_or(BIGRAM_BACKOFF),
            None => 0,
        }
    }
}
